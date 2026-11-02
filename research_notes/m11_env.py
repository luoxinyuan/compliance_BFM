"""Reuse actual ComplianceEnv, M10 deployed prior, original 50Hz BFM/PD and Tether.
No future partner data enters the expert; episode config is only environment state.
"""
import copy,json,math,functools,gzip
from pathlib import Path
import numpy as np,torch
from m8_stage import reset_common,advance
from m9_learning import old_context,load_net,inputs,policy,read,sha
from m11_history import FastHistory as ContactHistory
from contact_projector import ContactGate
from m10_contact import install_partner
from locomotion_rollout import UnsafePhysics,spherical
from m11_policy import legal_command
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m11-privileged-expert-20261101'
PRIOR=P/'bridge-v2-owned/runs/m10-task-appropriate-20261101/train1-attempt-01/outputs/student-final.pt'
PRIOR_SHA='1742ba74e9ec871abf8577eb9a276d7a1e614cd3184330a008dd9c4a7031bcd8'

class Task:
    def __init__(self,e,model,contract,seed):
        self.e=e;self.m=model;self.c=contract;self.rng=np.random.default_rng(seed)
        assert sha(PRIOR)==PRIOR_SHA
        self.prior=load_net(PRIOR,e.standing.device);self.old,self.ends=old_context(e)
        import m8_stage
        m8_stage.read=functools.lru_cache(maxsize=8)(m8_stage.read)
        oldrun=P/'bridge-v2-owned/runs/m8-contact-body-20261101A';a=[a for a in read(oldrun/'compute-ledger.json')['attempts'] if a['stage']=='qualification' and a.get('result_status')=='completed'][0]
        self.free=read(Path(a['path'])/'outputs/standing.json.gz')['physics']
        self.scale=torch.tensor(contract['q_PD_scale'],device=e.standing.device);self.off=torch.tensor(contract['q_PD_offset'],device=e.standing.device)
        self.raw=torch.zeros((1,256),device=e.standing.device);self.refs=None
        self.total_prefill=0;self.total_steps=0;self.total_controls=0;self.total_episodes=0
    def qof(self,a):return self.off+(a*5).clamp(-self.c['clip'],self.c['clip'])*self.scale/5
    def sample_spec(self):
        r=self.rng;p=r.random();local=p<.25;enabled=p>=.05
        if local:dist=float(r.uniform(.04,.20));ramp=float(r.uniform(3.5,5.5));curve=0.;yaw=float(r.uniform(-12,12))
        else:dist=float(r.uniform(.60,1.05));ramp=float(r.uniform(6.8,9.5));curve=float(r.uniform(-.24,.24));yaw=float(r.uniform(-14,14))
        onset=round(float(r.uniform(4.1,5.1)),2);release=round(onset+ramp+2,2)
        spec=dict(enabled=enabled,distance=dist,ramp=ramp,onset=onset,release=release,end=release+6.,curvature=curve,heading_deg=yaw,family='walking' if not local else 'old')
        if not local and r.random()<.25:spec['prefix_until']=round(onset+float(r.uniform(2.,4.5)),2)
        return spec
    def reset(self,spec=None,record=False,label='train',query_reference=False):
        e=self.e;self.spec=copy.deepcopy(spec or self.sample_spec());s=self.spec;self.record=record;self.label=label;self.track_reference=query_reference
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
        self.h=ContactHistory();self.gate=ContactGate();self.tick=0;self.raw.zero_();self.prevz=e.standing.clone();self.lastq=self.qof(e.previous_action);self.done=False;self.boundary=None;self.physics=[];self.controls=[];self.stats=[];self.episode_return=0.;e.trace=[];e.qualification_active=True
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
        if self.tick%5==0:self.gate.update(t,f)
        h,h8=inputs(self.m,e.obs,e.keys,self.h,R)
        self.base=policy(self.m,self.old,self.ends,self.prior,h,h8,self.gate.value(t)).detach()
        self.pro=h[:,:465];self.prepared=True
        return self.base
    def features(self):
        e=self.e;self.prepare();R=e.sim.data.xmat[1].reshape(3,3);point,vel=e.tether.point_state();vp=getattr(e.tether,'partner_velocity',np.zeros(3));active=bool(e.tether.active)
        err=R.T@(e.tether.anchor-point) if active else np.zeros(3);rv=R.T@(vp-vel) if active else np.zeros(3)
        extra=np.r_[err/.25,rv/.5,R.T@e.tether.force/5,R.T@e.sim.data.qvel[:3]/.5,e.sim.data.qvel[3:6]/2,R.T@(point-e.sim.data.qpos[:3]),e.sim.data.qpos[2],R[2,:],e.last_support,float(active)]
        return torch.cat([self.pro.clamp(-10,10),torch.tensor(extra,device=e.standing.device,dtype=torch.float32)[None].clamp(-10,10),self.base/16,self.raw.detach()],-1)
    def native_expert(self):
        # Frozen M10 reference+projected-feedback comparator, no new rules.
        from m9_cyclic import CyclicFeedback,sample
        from m9c_feedback import ArmFeedback,project_action
        if self.refs is None:
            from m10_reference import load_variants
            self.refs=load_variants(self.e,self.m)[0]['task']
        if self.tick==0 or not hasattr(self,'fb'):
            self.fb=CyclicFeedback({k:r['period'] for k,r in self.refs.items()},True);self.arm=ArmFeedback(self.e,self.c)
        e=self.e;R=e.sim.data.xmat[1].reshape(3,3);guide=getattr(e.tether,'partner_velocity',np.zeros(3))+.5*(e.tether.anchor-e.last_point)
        st=self.fb.step(e.tether.force,R,guide=guide);turn=st['turn'];z=spherical(sample(self.refs['straight'],st['phase']),sample(self.refs['left' if turn>=0 else 'right'],st['phase']),abs(turn));z=spherical(e.standing,z,st['alpha'])
        if self.spec['family']=='old':return self.base
        target,_=self.arm.step(self.m.act(e.obs,z,mean=True),float(e.tether.time),self.tick+1)
        return project_action(self.m,e.obs,z,target,self.c)[0]
    def step(self,raw,prefix=False,mode='learned'):
        e=self.e;s=self.spec;self.raw=raw.detach().clone();re=0.;rows=[];terms=[];control=[]
        for _ in range(5):
            if self.done:break
            base=self.prepare();t=float(e.tether.time);z,norm=legal_command(self.m,base,self.raw)
            oracle=self.native_expert() if self.track_reference else None
            if mode=='prior' or prefix:z=base
            elif mode=='standing':z=e.standing
            elif mode=='m10_expert':z=oracle
            a=self.m.act(e.obs,z,mean=True);q=self.qof(a);nom=self.qof(self.m.act(e.obs,base,mean=True));dq=q-self.lastq
            assert torch.isfinite(z).all() and abs(float(z.norm())-16)<1e-4
            c=dict(bfm_obs={k:v.tolist() for k,v in e.obs.items()},time=t,executed_z=z.tolist(),action=a[0].tolist(),q_PD=q[0].tolist(),q_PD_step_max_rad=float(dq.abs().max()),pre_qpos=e.sim.data.qpos.tolist(),pre_qvel=e.sim.data.qvel.tolist(),measured_force_xyz=e.tether.force.tolist(),action_saturation_fraction=float(((a*5).abs()>=self.c['clip']).float().mean()),residual_uncapped_norm=float(norm),prior_z=base.tolist()) if self.record else None
            start=len(e.trace)
            try:
                if advance(e,a):self.boundary=dict(kind='native_termination',time=float(e.tether.time))
            except UnsafePhysics as exc:self.boundary=dict(kind='physics_review',time=float(e.tether.time),physics=exc.row)
            rs=e.trace[start:];rows.extend(rs);self.total_steps+=len(rs);self.total_controls+=1
            point,vel=e.tether.point_state();vp=getattr(e.tether,'partner_velocity',np.zeros(3));active=bool(e.tether.active);err=e.tether.anchor-point;demand=80*err+8*(vp-vel) if active else np.zeros(3)
            F=np.array([r['force'] for r in rs]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in rs]);vroot=e.sim.data.qvel[:3];R=e.sim.data.xmat[1].reshape(3,3);yaw=math.atan2(R[1,0],R[0,0]);direction=math.atan2(vp[1],vp[0]);motion_weight=min(np.linalg.norm(vroot[:2])/.12,1.) if active and np.linalg.norm(vp[:2])>.02 else 0.
            term=dict(force=1.5*float(np.mean(np.sum(F*F,1))/25),uncapped=.15*float(min(np.log1p(np.sum(demand*demand)/25),8)),tracking=.15*float(min(np.mean(np.sum(E*E,1))/.04,16)) if active else 0.,relative_velocity=.1*float(min(np.sum((vp-vel)**2)/.25,16)) if active else 0.,heading=.1*motion_weight*(1-math.cos(yaw-direction)),slip=.05*min((max(r['contact_tangent_speed_max_mps'] for r in rs)/.15)**2,16),tilt=.1*min((max(0,rs[-1]['tilt_deg']-15)/15)**2,16),continuity=.05*float(min(dq.square().mean()/.01,16)),soft_prior=.02*float(min((q-nom).square().mean()/.04,16)),idle=.2*float(min(np.sum(vroot*vroot)/.04,16)) if not active else 0.)
            re-=.02*sum(term.values());terms.append(term)
            if self.record:
                c['uncapped_demand_xyz']=demand.tolist();control.append(c)
            self.lastq=q.clone();self.prevz=z.clone();self.tick+=1;self.prepared=False
            self.done=self.boundary is not None or e.tether.time>=s['end']-1e-8
            if self.boundary:
                # Absorbing maximum-cost tail: falling cannot buy force relief.
                remaining=max(0,math.ceil((s['end']-e.tether.time)*10));re-=10+(1-.99**remaining)/.01
            if self.done:break
        e.trace=[]
        if self.record:self.physics.extend(rows);self.controls.extend(control)
        stat=dict(force_squared=sum(np.dot(r['force'],r['force']) for r in rows),active_steps=sum(r['active'] for r in rows),active_force_squared=sum(np.dot(r['force'],r['force']) for r in rows if r['active']),cap_steps=sum(np.linalg.norm(r['force'])>=4.999 for r in rows),error_squared=sum(r['error']**2 for r in rows if r['active']),physics_steps=len(rows),control_steps=len(terms),prefix=prefix,reward=re,terms={k:float(np.mean([v[k] for v in terms])) for k in terms[0]},boundary=self.boundary)
        self.stats.append(stat);self.episode_return+=re
        if self.done:e.qualification_active=False
        return self.features(),re,self.done,stat
    def summary(self):
        s=self.stats;active=sum(x['active_steps'] for x in s);return dict(label=self.label,condition=self.spec,boundary=self.boundary,complete=self.done and self.boundary is None,force_rms=(sum(x['active_force_squared'] for x in s)/max(active,1))**.5,error_cm=100*(sum(x['error_squared'] for x in s)/max(active,1))**.5,cap_fraction=sum(x['cap_steps'] for x in s)/max(active,1),episode_return=self.episode_return,physics_steps=sum(x['physics_steps'] for x in s),rl_decisions=sum(not x['prefix'] for x in s),prefill_steps=self.total_prefill)
    def save(self,out):
        e=dict(label=self.label,initial=self.initial,physics=self.physics,control=self.controls,boundary=self.boundary,condition=self.spec,anchor=self.anchor.tolist(),phases={'pre':[3,self.spec['onset']],'move':[self.spec['onset'],self.spec['onset']+self.spec['ramp']],'hold':[self.spec['onset']+self.spec['ramp'],self.spec['release']],'recover':[self.spec['release'],self.spec['end']]})
        with gzip.open(out/(self.label+'.json.gz'),'wt') as f:json.dump(e,f)
        return self.summary()
