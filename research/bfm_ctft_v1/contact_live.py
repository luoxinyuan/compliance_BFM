"""One isolated real contact task for S0 throughput; original physics and PD.

No learned high-level expert, source ID, schedule or partner truth reaches the
actor. This module deliberately does not claim batched physics capability.
"""
import copy
import gzip
from pathlib import Path
import pickle
import types
import numpy as np
import torch
import mujoco
from agent import ACTOR_KEY,CRITIC_KEY
from contact_scale_qualify import restore_source
from contact_interface_probe import measure,partner
from features import ForceHistory,actor_fields,critic_fields
from reward import ForceEMA,contact_cost
from quality import OnlineSupport
from source_qualify import capture_state
from learning_checkpoint import cpu,rng,set_rng
from locomotion_rollout import UnsafePhysics


class ContactLive:
    def __init__(self,e,stage,coefficients):
        self.e=e
        self.stage=Path(stage)
        self.coefficients=coefficients
        self.hook=e.sim.simulate_at_each_physics_step
        self.native_measure=e.tether.measure.__func__
        self.pelvis=mujoco.mj_name2id(e.sim.model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
        self.done=True
        self.guard=None
        self.cache={}

    def start(self,spec):
        e=self.e
        learner_rng=rng()
        e.qualification_active=False
        if spec['mode']==1:
            name='force' if spec['force'] else 'no_force'
            key=('contact',name)
            if key not in self.cache:
                with gzip.open(self.stage/'runs/s0-contact-interface-01'/(name+'.pkl.gz'),'rb') as f:
                    x=pickle.load(f)
                self.cache[key]=x['snapshots'][0]
                del x
            saved=self.cache[key]
            contract=saved['tether']['ctft_partner_spec']
            for cfg in [e.cfg,e.tether.cfg]:
                cfg.enabled=spec['force'];cfg.body='right_wrist_yaw_link';cfg.local_point=[0.,0.,0.]
                cfg.stiffness=80.;cfg.damping=8.;cfg.max_force=10.
                cfg.activation_time=contract['onset_s'];cfg.release_time=contract['release_s']
            restore_source(e,saved)
            e.tether.measure=types.MethodType(measure,e.tether)
            self.history=ForceHistory(saved['ctft_force_history'])
            self.ema=ForceEMA(np.asarray(saved['ctft_force_ema']).copy())
            self.controls=round((contract['end_s']-float(e.tether.time))/.02)
            self.native_commands=None
        else:
            key=('native',spec['case'])
            if key not in self.cache:
                with gzip.open(self.stage/'runs/s0-native-sources-03'/f"source-{spec['case']:02d}.pkl.gz",'rb') as f:
                    x=pickle.load(f)
                assert x['boundary'] is None
                i=25
                prior=x['physics'][(i-10)*4:i*4:4]
                assert len(prior)==10
                self.cache[key]=dict(state=x['states'][i],commands=[r['z'].clone() for r in x['controls'][i:625]],
                    history=np.array([np.asarray(r['pelvis_matrix']).T@np.asarray(r['force']) for r in prior]))
                del x
            source=self.cache[key]
            saved=source['state']
            e.cfg.enabled=e.tether.cfg.enabled=False
            e.tether.measure=types.MethodType(self.native_measure,e.tether)
            restore_source(e,saved)
            self.history=ForceHistory(source['history'])
            self.ema=ForceEMA()
            self.native_commands=source['commands']
            self.controls=len(self.native_commands)
        self.guard=OnlineSupport(**saved['online_support'])
        self.last_z=saved['actual_last_z'].to(e.standing.device)
        e.trace=[]
        e.sim.simulate_at_each_physics_step=self.guarded
        e.qualification_active=True
        self.spec=copy.deepcopy(spec)
        self.i=0
        self.done=False
        # Physical source RNGs/history have been restored. The explicitly owned
        # learner/sampling RNG continues, not rewound by a source replay reset.
        set_rng(learner_rng)

    def guarded(self):
        self.hook()
        state=self.guard.update(self.e.contact_sample['foot_normal_force_n'])
        self.e.trace[-1]['ctft_online_support']=state
        if self.spec['mode']==1:
            self.e.trace[-1]['ctft_raw_force']=self.e.tether.ctft_raw_force.tolist()
        if state['stop']:
            raise UnsafePhysics(dict(self.e.trace[-1],ctft_stop_reason='continuous_no_effective_support_gt_50ms'))

    def observation(self):
        e=self.e;m,d=e.sim.model,e.sim.data;t=e.tether
        point=d.xpos[t.body].copy()
        jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,t.body)
        R=d.xmat[self.pelvis].reshape(3,3)
        a=actor_fields(force_history=self.history,known_mount=[0.,0.,0.],wrist_world=point,
            root_world=d.qpos[:3],wrist_velocity_world=jac@d.qvel,root_velocity_world=d.qvel[:3],world_from_body=R,public_mode=self.spec['mode'])
        if self.spec['mode']==1:
            p,v=partner(t.ctft_partner_spec,float(t.time));anchor=t.ctft_partner_origin+p
            active=t.ctft_partner_spec['apply_force'] and t.ctft_partner_spec['onset_s']<=t.time<t.ctft_partner_spec['release_s']
            raw=80.*(anchor-point)+8.*(v-jac@d.qvel) if active else np.zeros(3)
            c=critic_fields(current_partner_world=anchor,current_partner_velocity_world=v,current_raw_force_world=raw,wrist_world=point,world_from_body=R)
        else:c=np.zeros(9,dtype=np.float32)
        return dict(e.obs,**{ACTOR_KEY:torch.tensor(a[None],device=e.standing.device),CRITIC_KEY:torch.tensor(c[None],device=e.standing.device)})

    def command(self):
        return self.e.standing if self.native_commands is None else self.native_commands[self.i].to(self.e.standing.device)

    def step(self,action,before,z):
        from m8_stage import advance
        # The optional wrapper time tensor aliases episode_length_buf. Freeze
        # the actual pre-action input before physics mutates that metadata.
        # Clock is NOT in the normalizer/Actor/F/B/discriminator input filters.
        frozen_before=cpu(before)
        e=self.e;m,d=e.sim.model,e.sim.data;t=e.tether
        e.previous_z=z.clone()
        native_done=advance(e,action)
        assert not native_done,'Native hard terminal/truncation inside predeclared S0 episode'
        self.last_z=z.clone()
        self.history.append_actual(t.force,d.xmat[self.pelvis].reshape(3,3))
        filtered=self.ema.update(t.force)
        point=d.xpos[t.body].copy();jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,t.body)
        cost=contact_cost(filtered_force_n=filtered,applied_force_n=t.force,wrist_velocity_mps=jac@d.qvel,
            root_velocity_mps=d.qvel[:3],guidance_mode=self.spec['mode'])
        raw=e.base.extras['aux_rewards']
        reward=sum(float(raw[k].detach().cpu().reshape(-1)[0])*v for k,v in self.coefficients.items())+cost['contact_native_reward']
        self.i+=1;self.done=self.i>=self.controls
        return dict(observation=frozen_before,action=cpu(action),z=cpu(z),
            next=dict(observation=cpu(self.observation()),terminated=torch.zeros(1,1,dtype=torch.bool),truncated=torch.tensor([[self.done]],dtype=torch.bool)),
            aux_rewards=dict(ctft_total_native=torch.tensor([[reward]],dtype=torch.float32)))

    def snapshot(self):
        saved=capture_state(self.e,self.last_z,dict(scope='S0_actual_online_task',time_s=float(self.e.tether.time)))
        saved.update(ctft_force_history=self.history.state(),ctft_force_ema=self.ema.value.copy(),online_support=copy.deepcopy(vars(self.guard)),
            live_spec=copy.deepcopy(self.spec),live_control=self.i,live_controls=self.controls,
            native_commands=cpu(self.native_commands),done=self.done)
        return saved

    def restore(self,saved):
        """Restore this callback graph, not just body coordinates or data fields."""
        learner_rng=rng();e=self.e
        spec=copy.deepcopy(saved['live_spec'])
        if spec['mode']==1:
            partner_spec=saved['tether']['ctft_partner_spec']
            for cfg in [e.cfg,e.tether.cfg]:
                cfg.enabled=spec['force'];cfg.body='right_wrist_yaw_link'
                cfg.local_point=[0.,0.,0.];cfg.stiffness=80.;cfg.damping=8.;cfg.max_force=10.
                cfg.activation_time=partner_spec['onset_s'];cfg.release_time=partner_spec['release_s']
        else:e.cfg.enabled=e.tether.cfg.enabled=False
        restore_source(e,saved)
        e.tether.measure=types.MethodType(measure if spec['mode']==1 else self.native_measure,e.tether)
        self.spec=spec;self.history=ForceHistory(saved['ctft_force_history'])
        self.ema=ForceEMA(np.asarray(saved['ctft_force_ema']).copy())
        self.guard=OnlineSupport(**saved['online_support']);self.last_z=saved['actual_last_z'].to(e.standing.device)
        self.native_commands=copy.deepcopy(saved['native_commands']);self.i=saved['live_control']
        self.controls=saved['live_controls'];self.done=saved['done']
        e.sim.simulate_at_each_physics_step=self.guarded;e.qualification_active=True
        e.trace=[];set_rng(learner_rng)
