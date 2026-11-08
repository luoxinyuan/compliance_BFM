"""Reuse actual ComplianceEnv, frozen native motor primitives, original 50Hz BFM/PD and Tether.
No future partner data enters the expert; episode config is only environment state.
"""
import copy,json,math,functools,gzip,os
from pathlib import Path
import numpy as np,torch
from m8_stage import reset_common,advance
from m9_learning import old_context,load_net,inputs,policy,read,sha
from m11_history import FastHistory as ContactHistory
from m10_contact import install_partner
from locomotion_rollout import UnsafePhysics,spherical
from clean_policy import structured_command, direct_command
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m11-privileged-expert-20261101'
PRIOR=P/'bridge-v2-owned/runs/m10-task-appropriate-20261101/train1-attempt-01/outputs/student-final.pt'
PRIOR_SHA='1742ba74e9ec871abf8577eb9a276d7a1e614cd3184330a008dd9c4a7031bcd8'

class Task:
    def __init__(self,e,model,contract,seed):
        self.e=e;self.m=model;self.c=contract;self.rng=np.random.default_rng(seed)
        self.prior=None;self.old=None;self.ends=None
        ref=read(Path(__file__).with_name('narrow_protocol.json'))['native_reference'];assert sha(Path(ref['path']))==ref['sha256']
        from m10_reference import load_variants
        self.refs=load_variants(e,model)[0]['task']
        import m8_stage
        m8_stage.read=functools.lru_cache(maxsize=8)(m8_stage.read)
        oldrun=P/'bridge-v2-owned/runs/m8-contact-body-20261101A';a=[a for a in read(oldrun/'compute-ledger.json')['attempts'] if a['stage']=='qualification' and a.get('result_status')=='completed'][0]
        self.free=read(Path(a['path'])/'outputs/standing.json.gz')['physics']
        self.scale=torch.tensor(contract['q_PD_scale'],device=e.standing.device);self.off=torch.tensor(contract['q_PD_offset'],device=e.standing.device)
        self.kind=os.environ.get('CLEAN_KIND','structured');self.raw=torch.zeros((1,258 if self.kind=='structured' else 256),device=e.standing.device);self.nominal_state=None
        self.total_prefill=0;self.total_steps=0;self.total_controls=0;self.total_episodes=0
        self.record_nominal_diagnostic=os.environ.get('BFM_RECORD_NOMINAL_DIAGNOSTIC','1')=='1'
    def qof(self,a):return self.off+(a*5).clamp(-self.c['clip'],self.c['clip'])*self.scale/5
    def sample_spec(self):
        plan=read(Path(__file__).with_name('narrow_protocol.json'))
        spec=copy.deepcopy(plan['single'])
        if getattr(self,'neighborhood',False):
            spec['distance']=float(self.rng.uniform(.76,.88));spec['ramp']=float(self.rng.uniform(7.4,8.2))
            spec['onset']=float(self.rng.uniform(4.05,4.45));spec['release']=spec['onset']+spec['ramp']+2;spec['end']=spec['release']+6
        return spec
    def reset(self,spec=None,record=False,label='train',query_reference=False):
        e=self.e;self.spec=copy.deepcopy(spec or self.sample_spec());s=self.spec;self.record=record;self.label=label;self.track_reference=query_reference
        self.alpha=0.;self.phase=0.;self.phase_rate=0.;self.last_command=e.standing.clone();self.bootstrap_rows=[];self.decision_rows=[];self.raw.zero_()
        self.compare_mode=getattr(self,'requested_mode','learned')
        reset_common(e,self.m,self.m.act);self.total_prefill+=600
        rotation=None
        if s.get('heading_deg'):
            from m9_heading import rotate_ready
            rotation,pivot,_=rotate_ready(e,self.m,s['heading_deg']);self.total_prefill+=80
        rr=min(self.free,key=lambda r:abs(r['time']-s['onset']));anchor=np.array(rr['wrist'])
        if s.get('family')=='old':
            from m6_reference import wrist_at_time
            oldfree=read(P/'bridge-v2-owned/runs/m5-contact-lpp-20261031A/qualification-attempt-01/outputs/free__teacher.json.gz')['physics'];anchor=np.array(wrist_at_time(oldfree,s['onset']))
        if rotation is not None:anchor=pivot+rotation@(anchor-pivot)
        for c in [e.cfg,e.tether.cfg]:c.enabled=s['enabled'];c.activation_time=s['onset'];c.ramp_duration=s['ramp'];c.release_time=s['release']
        install_partner(e.tether,s,anchor);self.anchor=anchor;self.initial=copy.deepcopy(e.trace[-1]);self.initial['wrist']=e.last_point.tolist()
        self.h=ContactHistory();self.tick=0;self.raw.zero_();self.prevz=e.standing.clone();self.lastq=self.qof(e.previous_action);self.done=False;self.boundary=None;self.physics=[];self.controls=[];self.stats=[];self.execution_stats=[];self.root_max=0.;self.episode_return=0.;e.trace=[];e.qualification_active=True
        self.prepared=False;self.base=self.prepare()
        # Legitimate visited deviations are obtained by actual frozen M10 rollout,
        # not by arbitrary qpos edits. Same prefix used in paired evaluation.
        until=s.get('prefix_until',0.)
        while e.tether.time<until-1e-8:
            _,_,done,_=self.step(torch.zeros_like(self.raw),prefix=True)
            if done:break
        self.total_episodes+=1
        return self.features()
    def prepare(self):
        if self.prepared:return self.base
        e=self.e;t=float(e.tether.time);R=e.sim.data.xmat[1].reshape(3,3);f=e.tether.force.copy()
        self.h.append(f,'right_wrist_yaw_link',[0,0,0],R)
        h,h8=inputs(self.m,e.obs,e.keys,self.h,R)
        # No force-to-alpha, force-to-phase, contact gate, or online Student.
        self.base=e.standing.detach()
        self.pro=h[:,:465];self.prepared=True
        self.nominal_state=dict(alpha=self.alpha,phase=self.phase,phase_rate=self.phase_rate,source='policy_own_command_memory')
        return self.base
    def read_point_state(self):
        # Measurement must not run another dynamics solve or alter warmstart.
        # Explicit legacy mode exists only for paired regression of old runs.
        e=self.e;t=e.tether
        if not self.spec.get('cached_observation_reads',True):return t.point_state()
        import mujoco
        point=t.data.xpos[t.body]+t.data.xmat[t.body].reshape(3,3)@np.asarray(t.cfg.local_point)
        mujoco.mj_jac(t.model,t.data,t.jac,None,point,t.body)
        return point.copy(),t.jac@t.data.qvel
    def features(self):
        e=self.e;self.prepare();R=e.sim.data.xmat[1].reshape(3,3);point,vel=self.read_point_state();vp=getattr(e.tether,'partner_velocity',np.zeros(3));active=bool(e.tether.active)
        err=R.T@(e.tether.anchor-point) if active else np.zeros(3);rv=R.T@(vp-vel) if active else np.zeros(3)
        extra=np.r_[err/.25,rv/.5,R.T@e.tether.force/5,R.T@e.sim.data.qvel[:3]/.5,e.sim.data.qvel[3:6]/2,R.T@(point-e.sim.data.qpos[:3]),e.sim.data.qpos[2],R[2,:],e.last_support,float(active)]
        hist=self.h.features(R)[-25:,:3].reshape(-1) # actual past 0.5s, rotated to current body frame
        memory=[self.alpha,np.sin(2*np.pi*self.phase),np.cos(2*np.pi*self.phase),self.phase_rate]
        return torch.cat([self.pro.clamp(-10,10),torch.tensor(extra,device=e.standing.device,dtype=torch.float32)[None].clamp(-10,10),torch.tensor(hist,device=e.standing.device,dtype=torch.float32)[None],self.last_command/16,torch.tensor([memory],device=e.standing.device,dtype=torch.float32)],-1)
    def native_expert(self):
        # Frozen M10 reference+projected-feedback comparator, no new rules.
        from m9_cyclic import CyclicFeedback,sample
        from m9c_feedback import ArmFeedback,project_action
        if self.tick==0 or not hasattr(self,'fb'):
            self.fb=CyclicFeedback({k:r['period'] for k,r in self.refs.items()},True);self.arm=ArmFeedback(self.e,self.c)
        e=self.e;R=e.sim.data.xmat[1].reshape(3,3);guide=getattr(e.tether,'partner_velocity',np.zeros(3))+.5*(e.tether.anchor-e.last_point)
        st=self.fb.step(e.tether.force,R,guide=guide);self.teacher_state=st.copy();turn=st['turn'];z=spherical(sample(self.refs['straight'],st['phase']),sample(self.refs['left' if turn>=0 else 'right'],st['phase']),abs(turn));z=spherical(e.standing,z,st['alpha'])
        if self.spec['family']=='old':return self.base
        target,_=self.arm.step(self.m.act(e.obs,z,mean=True),float(e.tether.time),self.tick+1)
        return project_action(self.m,e.obs,z,target,self.c)[0]
    def step(self,raw,prefix=False,mode='learned'):
        e=self.e;s=self.spec;decision_obs=self.features().detach().clone();self.raw=raw.detach().clone();re=0.;rows=[];terms=[];control=[];teacher_labels=[];start_z=self.last_command.clone()
        for substep in range(5):
            if self.done:break
            self.prepare();t=float(e.tether.time)
            if self.kind=='structured':
                z,base,self.alpha,self.phase,self.phase_rate,norm=structured_command(self.m,e.standing,self.refs['straight'],self.raw,self.alpha,self.phase)
            else:
                target=direct_command(self.raw,e.standing)
                z=spherical(start_z,target,(substep+1)/5);base=e.standing;norm=(target-e.standing).norm(dim=-1,keepdim=True)
                self.alpha=0.;self.phase=0.;self.phase_rate=0.
            oracle=self.native_expert() if self.track_reference else None
            if mode=='standing':z=e.standing
            elif mode=='m10_expert':
                z=oracle
                if self.kind=='structured':
                    self.alpha=self.teacher_state['alpha'];self.phase=self.teacher_state['phase'];self.phase_rate=1. if self.alpha>.05 else 0.
                from m9_cyclic import sample
                tb=spherical(e.standing,sample(self.refs['straight'],self.teacher_state['phase']),self.teacher_state['alpha'])
                cosine=(z*tb).sum(-1,keepdim=True)/256
                tangent=z/cosine.clamp_min(1e-5)-tb
                teacher_labels.append(dict(alpha=self.teacher_state['alpha'],phase=self.teacher_state['phase'],rate=1. if self.teacher_state['alpha']>.05 else 0.,raw_residual=(tangent/16)[0].tolist(),z=z[0].tolist(),angle_deg=float(torch.rad2deg(torch.acos(cosine.clamp(-1,1)))[0,0]),tangent_norm=float(tangent.norm())))
            self.last_command=z.detach().clone()
            self.nominal_state=dict(alpha=self.alpha,phase=self.phase,phase_rate=self.phase_rate,source='teacher_data_only' if mode=='m10_expert' else 'policy')
            a=self.m.act(e.obs,z,mean=True);q=self.qof(a);nom=self.qof(self.m.act(e.obs,base,mean=True)) if self.record_nominal_diagnostic else None;dq=q-self.lastq
            assert torch.isfinite(z).all() and abs(float(z.norm())-16)<1e-4
            execution_diag=dict(raw_norm=float(self.raw.norm()),residual_norm=float(norm),latent_angle_deg=float(torch.rad2deg(torch.acos(((z*base).sum()/256).clamp(-1,1)))),qPD_step_max_deg=float(torch.rad2deg(dq.abs()).max()))
            if nom is not None:execution_diag['qPD_vs_prior_rms_deg']=float(torch.rad2deg(q-nom).square().mean().sqrt())
            c=dict(bfm_obs={k:v.tolist() for k,v in e.obs.items()},time=t,executed_z=z.tolist(),action=a[0].tolist(),q_PD=q[0].tolist(),q_PD_step_max_rad=float(dq.abs().max()),pre_qpos=e.sim.data.qpos.tolist(),pre_qvel=e.sim.data.qvel.tolist(),measured_force_xyz=e.tether.force.tolist(),action_saturation_fraction=float(((a*5).abs()>=self.c['clip']).float().mean()),residual_uncapped_norm=float(norm),prior_z=base.tolist()) if self.record else None
            start=len(e.trace)
            try:
                if advance(e,a):self.boundary=dict(kind='native_termination',time=float(e.tether.time))
            except UnsafePhysics as exc:self.boundary=dict(kind='physics_review',time=float(e.tether.time),physics=exc.row)
            rs=e.trace[start:];rows.extend(rs);self.root_max=max(self.root_max,max(float(np.linalg.norm(np.array(r['root'])-self.initial['root'])) for r in rs));self.total_steps+=len(rs);self.total_controls+=1
            point,vel=self.read_point_state();vp=getattr(e.tether,'partner_velocity',np.zeros(3));active=bool(e.tether.active);err=e.tether.anchor-point;demand=80*err+8*(vp-vel) if active else np.zeros(3)
            F=np.array([r['force'] for r in rs]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in rs]);vroot=e.sim.data.qvel[:3];R=e.sim.data.xmat[1].reshape(3,3);yaw=math.atan2(R[1,0],R[0,0]);direction=math.atan2(vp[1],vp[0]);motion_weight=min(np.linalg.norm(vroot[:2])/.12,1.) if active and np.linalg.norm(vp[:2])>.02 else 0.
            term=dict(force=1.5*float(np.mean(np.sum(F*F,1))/25),uncapped=.15*float(min(np.log1p(np.sum(demand*demand)/25),8)),tracking=.15*float(min(np.mean(np.sum(E*E,1))/.04,16)) if active else 0.,relative_velocity=.1*float(min(np.sum((vp-vel)**2)/.25,16)) if active else 0.,heading=.1*motion_weight*(1-math.cos(yaw-direction)),slip=.05*min((max(r['contact_tangent_speed_max_mps'] for r in rs)/.15)**2,16),tilt=.1*min((max(0,rs[-1]['tilt_deg']-15)/15)**2,16),continuity=.05*float(min(dq.square().mean()/.01,16)),soft_prior=0.,idle=.2*float(min(np.sum(vroot*vroot)/.04,16)) if not active else 0.)
            re-=.02*sum(term.values());terms.append(term);term_diag=execution_diag
            if not hasattr(self,'execution_stats'):self.execution_stats=[]
            self.execution_stats.append(term_diag)
            if self.record:
                c['uncapped_demand_xyz']=demand.tolist();c['reward_terms']=term.copy();c['reward_integral']=-.02*sum(term.values());c['nominal_q_PD']=nom[0].tolist() if nom is not None else None;c['nominal_diagnostic_recorded']=nom is not None;c['post_point']=point.tolist();c['post_velocity']=vel.tolist();c['post_partner_velocity']=vp.tolist();c['post_active']=active;c['post_yaw']=yaw;c['post_root_velocity']=vroot.tolist();c['native_memory']=dict(self.nominal_state);control.append(c)
            self.lastq=q.clone();self.prevz=z.clone();self.tick+=1;self.prepared=False
            self.done=self.boundary is not None or e.tether.time>=s['end']-1e-8
            if self.boundary:
                # Absorbing maximum-cost tail: falling cannot buy force relief.
                remaining=max(0,math.ceil((s['end']-e.tether.time)*10));re-=10+(1-.99**remaining)/.01
            if self.done:break
        e.trace=[]
        if teacher_labels:
            # Endpoint labels align the next 100ms command with the actual teacher; no future partner observations.
            last=teacher_labels[-1]
            target=[last['alpha'],last['rate']]+last['raw_residual'] if self.kind=='structured' else (np.array(last['z'])/16).tolist()
            self.bootstrap_rows.append(dict(obs=decision_obs[0].cpu().tolist(),target=target,teacher=last))
        if self.record:
            self.physics.extend(rows);self.controls.extend(control)
            self.decision_rows.append(dict(time=float(e.tether.time),obs=decision_obs[0].cpu().tolist(),raw=self.raw[0].cpu().tolist()))
        stat=dict(force_squared=sum(np.dot(r['force'],r['force']) for r in rows),active_steps=sum(r['active'] for r in rows),active_force_squared=sum(np.dot(r['force'],r['force']) for r in rows if r['active']),cap_steps=sum(np.linalg.norm(r['force'])>=4.999 for r in rows),error_squared=sum(r['error']**2 for r in rows if r['active']),physics_steps=len(rows),control_steps=len(terms),prefix=prefix,reward=re,execution={k:float(np.mean([d[k] for d in self.execution_stats[-len(terms):]])) for k in execution_diag},execution_max={k:max(d[k] for d in self.execution_stats[-len(terms):]) for k in execution_diag},terms={k:float(np.mean([v[k] for v in terms])) for k in terms[0]},boundary=self.boundary)
        self.stats.append(stat);self.episode_return+=re
        if self.done:e.qualification_active=False
        return self.features(),re,self.done,stat
    def summary(self):
        s=self.stats;active=sum(x['active_steps'] for x in s);return dict(label=self.label,condition=self.spec,root_max_displacement=self.root_max,boundary=self.boundary,complete=self.done and self.boundary is None,force_rms=(sum(x['active_force_squared'] for x in s)/max(active,1))**.5,error_cm=100*(sum(x['error_squared'] for x in s)/max(active,1))**.5,cap_fraction=sum(x['cap_steps'] for x in s)/max(active,1),episode_return=self.episode_return,physics_steps=sum(x['physics_steps'] for x in s),rl_decisions=sum(not x['prefix'] for x in s),prefill_steps=self.total_prefill)
    def save(self,out):
        e=dict(label=self.label,initial=self.initial,physics=self.physics,control=self.controls,boundary=self.boundary,condition=self.spec,anchor=self.anchor.tolist(),phases={'pre':[3,self.spec['onset']],'move':[self.spec['onset'],self.spec['onset']+self.spec['ramp']],'hold':[self.spec['onset']+self.spec['ramp'],self.spec['release']],'recover':[self.spec['release'],self.spec['end']]})
        e['rl_stats']=self.stats;e['scope']='learned high-level motor commands; no online contact-to-motion rules';e['kind']=self.kind;e['bootstrap_rows']=self.bootstrap_rows;e['decisions']=self.decision_rows;
        with gzip.open(out/(self.label+'.json.gz'),'wt') as f:json.dump(e,f,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else (_ for _ in ()).throw(TypeError(type(x).__name__)))
        return self.summary()
