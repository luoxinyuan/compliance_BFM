"""New v2 bridge to the qualified single-MuJoCo physics; public Goal and own option memory.
Legacy physics stays unchanged. No learned old expert is loaded.
"""
import copy,math
import numpy as np
import torch
from dual_track_task import Task,global_rng
from dual_track_native_data import NativeActor,cpu_obs
from locomotion_rollout import spherical
from clean_policy import direct_command
from .schema import KEY

class GoalTask(Task):
    def sample_spec(self):raise RuntimeError('Only prelocked v2 manifests')
    def step(self,*args,native_limit=5,**kw):
        # Administrative 1024-row cuts may split a real five-control command.
        # Resume from its REAL original interpolation start/slot, not a new 100ms ramp.
        import clean_env
        original=clean_env.spherical;ctx=getattr(self,'v2_interpolation',None)
        if ctx is not None and not kw.get('prefix',False):
            start,offset=ctx
            def continue_interpolation(unused,target,alpha):
                k=round(float(alpha)*5);return original(start,target,(offset+k)/5)
            clean_env.spherical=continue_interpolation
        begin=len(self.physics);cb=len(self.controls)
        try:x,r,done,st=super().step(*args,native_limit=native_limit,**kw)
        finally:clean_env.spherical=original
        # Source prefixes retain exact old FOLLOW reward; they are never PPO rows.
        if kw.get('prefix',False) or getattr(self,'v2_goal_mode',0)==0:return x,r,done,st
        ph=self.physics[begin:];controls=self.controls[cb:];old_total=r;new_rewards=[]
        for i,c in enumerate(controls):
            end=controls[i+1]['time'] if i+1<len(controls) else float(self.e.tether.time)+1e-8
            rows=[p for p in ph if c['time']-1e-8<=p['force_time']<end-1e-8]
            assert rows
            hold=float(np.mean([4*np.sum((np.asarray(p['wrist'])-self.v2_goal_position_world)**2)/.04**2+.2*np.sum(np.asarray(p['contacts'][0]['velocity'])**2)/.10**2 if p['active'] else 0. for p in rows]))
            old=c['reward_terms'];removed=sum(old[k] for k in ('tracking','relative_velocity','heading'))
            nr=st['native_rewards'][i]+len(rows)*.005*(removed-hold)
            c['v2_old_native_reward']=st['native_rewards'][i];c['v2_goal_cost']=hold;c['v2_goal_mode']='HOLD';c['reward_integral']=nr
            c['reward_terms']=dict({k:v for k,v in old.items() if k not in ('tracking','relative_velocity','heading')},hold=hold)
            new_rewards.append(nr)
        r=sum(new_rewards);self.episode_return+=r-old_total;st=dict(st,native_rewards=new_rewards,reward=r,v2_goal_mode='HOLD');self.stats[-1]=st
        return x,r,done,st

class GoalProxy(NativeActor):
    def __init__(self,env,task,original):
        super().__init__(task,original);self.env=env;self.agent=None;self.slot=0;self.start=None;self.target=None;self.actions=[];self.raw_records=[];self.stochastic=True;self.replay=None
    def enrich(self,o):
        dev=self.task.e.standing.device
        before=self.previous_z.to(dev)
        goal,mask,mode=self.env.public_goal()
        own=torch.zeros(1,5,device=dev);own[0,self.env.option]=1;own[0,4]=self.env.option_age/5
        return dict(o,_previous_z=before,_target_z=(before if self.target is None else self.target),_slot=torch.tensor([[self.slot]],device=dev),_goal_y=goal,_goal_mask=mask,_goal_mode=mode,_option_memory=own,_first_interpolated_z=(before if self.env.macro_start is None else spherical(self.env.macro_start,self.env.macro_target,.2)))
    def finish_pending(self):
        if self.pending is not None:self.pending['next_observation']=cpu_obs(self.enrich(self.task.extended_observation()));self.pending=None
    def act(self,obs,z,mean=True):
        o=self.enrich(self.task.extended_observation())
        if self.pending is not None:self.pending['next_observation']=cpu_obs(o)
        if self.source_mode:
            a=self.original.act(obs,z,mean=True);u=a;mu=a;lp=torch.zeros(1,device=a.device)
        elif self.replay is not None:
            rr=next(self.replay);a=rr['action'].to(z.device);u=rr['raw_u'].to(z.device);mu=rr['old_mu'].to(z.device);lp=rr['old_logp'].to(z.device)
        else:
            op=torch.tensor([self.env.option],device=z.device)
            u,a,mu,lp=self.agent.sample(o,z,op,self.stochastic)
        row=dict(observation=cpu_obs(o),action=a.detach().cpu(),raw_u=u.detach().cpu(),old_mu=mu.detach().cpu(),old_logp=lp.detach().cpu(),z=z.detach().cpu(),zi=torch.zeros(1,16),previous_z=self.previous_z.clone(),target_z=(z if self.target is None else self.target).detach().cpu(),time=float(self.task.e.tether.time),source=self.task.spec['source'])
        self.previous_z=z.detach().cpu();self.pending=row
        (self.source_pairs if self.source_mode else self.records).append(row)
        if not self.source_mode:self.actions.append(a.detach().cpu());self.slot+=1
        return a

class GoalEnvironment:
    def __init__(self,e,m,contract,parents):
        self.e=e;self.model=m;self.contract=contract;self.parents=parents;self.cases=[]
        for p in parents:
            for s in p['specs']:self.cases.append(dict(spec=copy.deepcopy(s),goal_stream=p['goal_stream'],execution_source_parent_id=p['execution_source_parent_id']))
        self.task=GoalTask(e,m,contract,660301);self.proxy=GoalProxy(self,self.task,m);self.task.m=self.proxy
        self.source_cache={};self.rows=[];self.native=[];self.physical_stream=[];self.episodes=[];self.instructions=[];self.next_cursor=0;self.reset_cursor=0
        self.collection_role='qualification';self.agent=None;self.option=0;self.option_age=0;self.command_slot=0;self.offset=0;self.macro_start=None;self.macro_target=None;self.motion_phase=None;self.motion_steps=0;self.spec_id=0;self.phase='P2';self.goal_stream='HOLD';self.goal_changed=False;self.handoff=0.;self.goal_mode=1;self.goal_world=None;self.reference_trace=[]
    def public_goal(self):
        t=self.task;e=self.e;dev=e.standing.device
        R=np.asarray(e.sim.data.xmat[1]).reshape(3,3)
        g=np.zeros(24,dtype=np.float32);mask=np.zeros(24,dtype=np.float32);mask[:3]=1;mask[6:9]=1;mask[21]=1;g[21]=1
        if self.goal_mode==0:
            f=np.asarray(e.tether.force);norm=np.linalg.norm(f);delta=R.T@f/norm*.10 if norm>1e-6 else np.zeros(3)
        else:
            point=t.read_point_state()[0];goal=point if self.goal_world is None else self.goal_world;delta=R.T@(goal-point)
        g[:3]=delta;g[6:9]=delta
        return torch.tensor(g,device=dev)[None],torch.tensor(mask,device=dev)[None],torch.tensor([[float(self.goal_mode==0),float(self.goal_mode==1)]],device=dev)
    def issue(self,mode):
        self.goal_mode=0 if mode=='FOLLOW' else 1
        if self.goal_mode:self.goal_world=self.task.read_point_state()[0].copy()
        self.task.v2_goal_mode=self.goal_mode;self.task.v2_goal_position_world=None if self.goal_world is None else self.goal_world.copy()
    def update_public_goal(self):
        if not self.goal_changed and '_TO_' in self.goal_stream and float(self.e.tether.time)>=self.handoff+1.5-1e-7:
            self.issue(self.goal_stream.split('_TO_')[1]);self.goal_changed=True
    def reset(self,index):
        from dual_track_runtime import TrainingBranch
        from diagnostic_frozen import portable
        from f1_mjdata_contract import canonical,metadata,exact
        p=self.proxy;t=self.task;case=self.cases[index];self.spec_id=index;self.goal_stream=case['goal_stream'];self.goal_world=None;self.goal_changed=False
        self.option=0;self.option_age=0;self.command_slot=0;self.offset=0;self.motion_phase=None;self.motion_steps=0
        self.goal_mode=0;self.macro_start=None;self.macro_target=None;t.v2_goal_mode=0;t.v2_goal_position_world=None;t.v2_interpolation=None
        p.begin_source();p.slot=0;p.start=None;p.target=None
        t.reset(case['spec'],record=True,label='v2-'+str(index));p.finish_pending()
        self.handoff=float(self.e.tether.time);self.native.extend(copy.deepcopy(p.source_pairs))
        key=case['spec']['parent_id']
        if key not in self.source_cache:
            before=canonical(portable(t),metadata(self.e.sim.model,self.e.sim.data));snap=TrainingBranch(t)
            fields={k:copy.deepcopy(getattr(p,k)) for k in ('source_mode','source_pairs','records','pending','previous_z','slot','start','target','actions')}
            first=len(t.physics)
            while float(self.e.tether.time)<t.spec['onset']-1e-7:
                t.step(t.native(t.spec['source'],float(self.e.tether.time))/16,prefix=True)
                assert t.boundary is None
            t.prepare();anchor=t.anchor.copy()
            self.source_cache[key]=dict(anchor=anchor.copy(),reference_trace=copy.deepcopy(t.physics[first:]),source_key=case['execution_source_parent_id'])
            snap.restore();assert exact(canonical(portable(t),metadata(self.e.sim.model,self.e.sim.data)),before),'reference-to-handoff full restore'
            for k,v in fields.items():setattr(p,k,v)
        # Modify the explicit partner-frame origin only, after full handoff restore.
        cache=self.source_cache[key];anchor=cache['anchor'].copy();measure=self.e.tether.measure.__func__
        cells=dict(zip(measure.__code__.co_freevars,measure.__closure__));assert 'base' in cells
        cells['base'].cell_contents=anchor.copy();t.anchor=anchor.copy();t.geometry_origin=anchor.copy();t.spec['anchor_mode']='v2_common_native_reference';self.e.tether.reach_aligned=True
        self.e.tether.anchor=anchor.copy()
        for c in self.e.tether.geometry_rows:c['anchor']=anchor.tolist()
        self.reference_trace=cache['reference_trace']
        self.issue(self.goal_stream.split('_TO_')[0]);self.macro_start=t.last_command.clone();self.macro_target=self.macro_start.clone();p.start=self.macro_start;p.target=self.macro_target;p.begin_online();p.actions=[]
        assert not self.e.tether.active
    def targets(self):
        from m9_cyclic import sample
        t=self.task;ref=t.refs['straight'];last=t.last_command
        phase=self.motion_phase
        if phase is None:phase=min((j/8 for j in range(8)),key=lambda x:float((sample(ref,x)-last).square().sum()))
        mix=float(np.interp(self.motion_steps/29,[0,.5,1],[.25,.5,.75]))
        zs=[self.e.standing,t.native_z['goal_arm5'],t.native_z['goal_arm15'],spherical(self.e.standing,sample(ref,phase),mix)]
        raw=[z/16 for z in zs];zs=[direct_command(r,self.e.standing) for r in raw]
        return raw,torch.cat(zs),phase
    def collect(self,native_limit=5):
        assert self.agent is not None
        if self.task.done:
            self.episodes.append(dict(spec=copy.deepcopy(self.task.spec),index=self.spec_id,boundary=self.task.boundary,online_rows=sum(r['episode']==self.reset_cursor for r in self.rows)))
            if self.task.boundary:raise RuntimeError('HARD_PHYSICS_BOUNDARY:'+str(self.task.boundary))
            self.reset_cursor+=1;self.reset((self.spec_id+1)%len(self.cases))
        p=self.proxy;t=self.task;device=self.e.standing.device;cat_boundary=False;old_logits=None;candidate_first=None
        if self.offset==0:
            self.update_public_goal();self.macro_start=t.last_command.clone();p.start=self.macro_start
            raw,zs,phase=self.targets();candidate_first=torch.stack([spherical(self.macro_start,z[None],.2)[0] for z in zs])[None]
            o=p.enrich(t.extended_observation())
            if self.phase=='P3' and self.command_slot%5==0:
                old_logits=self.agent.logits(o,candidate_first);dist=torch.distributions.Categorical(logits=old_logits)
                forced=getattr(self,'replay_option',None)
                self.option=(int(forced) if forced is not None else (int(dist.sample()) if p.stochastic else int(old_logits.argmax(-1))));self.option_age=0;cat_boundary=True
            elif self.phase=='P2':
                ref=getattr(self,'reference_constant_option',None)
                self.option=(int(ref) if ref is not None else (0 if self.goal_mode else 2))
            self.raw_target=raw[self.option];self.macro_target=zs[self.option:self.option+1];p.target=self.macro_target
            if self.option==3:
                self.motion_phase=(phase+.1*.5/t.refs['straight']['period'])%1;self.motion_steps+=1
        n=min(native_limit,5-self.offset);p.slot=self.offset;t.v2_interpolation=(self.macro_start.clone(),self.offset)
        if cat_boundary:
            # selector condition is the actual pre-draw history, distinct from post-draw own option memory.
            selector_obs=cpu_obs(o)
        else:selector_obs=None
        stream_begin=len(self.physical_stream);first=len(t.physics);rows,done,stats=p.take_step(self.raw_target,n)
        self.physical_stream.extend(copy.deepcopy(t.physics[first:]));self.instructions.append(dict(raw=self.raw_target.detach().cpu(),offset=self.offset,start=self.macro_start.detach().cpu(),target=self.macro_target.detach().cpu(),n=len(rows),option=self.option,phase=self.phase,cat_boundary=cat_boundary,episode=self.reset_cursor,time=float(rows[0]['provenance']['time']),goal_mode=self.goal_mode,goal_position_world=None if self.goal_world is None else self.goal_world.copy()))
        for j,(r,src) in enumerate(zip(rows,p.records)):
            begin=stream_begin+round((float(r['provenance']['time'])-float(rows[0]['provenance']['time']))/.005)
            end=begin+round(float(r['duration'])/.005)
            assert 0<=begin<end<=len(self.physical_stream)
            r.update(physics_begin=begin,physics_end=end)
            r.update(collection_role=self.collection_role,eligible_for_joint_PPO=bool(self.collection_role in ('engineering_on_policy','formal_on_policy') and p.stochastic and p.replay is None and getattr(self,'reference_constant_option',None) is None),behavior_version=int(self.agent.accepted),behavior_phase=self.phase,raw_u=src['raw_u'],old_mu=src['old_mu'],old_logp=src['old_logp'],option=self.option,cat_boundary=bool(cat_boundary and j==0),old_logits=old_logits.detach().cpu() if cat_boundary and j==0 else None,selector_obs=selector_obs if cat_boundary and j==0 else None,candidate_first_z=candidate_first.detach().cpu() if cat_boundary and j==0 else None,episode=self.reset_cursor,goal_mode=self.goal_mode,macro_slot=self.offset+j)
            self.rows.append(r)
        self.offset+=len(rows);self.next_cursor+=len(rows)
        if self.offset==5:
            self.offset=0;self.command_slot+=1;self.option_age=(self.option_age+1)%5
            self.update_public_goal()
            p.slot=0;p.start=t.last_command.clone();p.target=p.start.clone()
            self.rows[-1]['next']['observation']=cpu_obs(p.enrich(t.extended_observation()))
        if self.task.boundary:raise RuntimeError('HARD_PHYSICS_BOUNDARY:'+str(self.task.boundary))
        return rows,stats
