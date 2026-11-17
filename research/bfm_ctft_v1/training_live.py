"""Actual source-state contact task; no body pose reset or hidden actor input."""
import copy
import gzip
from pathlib import Path
import pickle
import types
import numpy as np
import torch
import mujoco
from contact_live import ContactLive
from contact_scale_qualify import restore_source,measure
from contact_interface_probe import partner
from features import ForceHistory,actor_fields,critic_fields
from reward import ForceEMA,contact_cost
from quality import OnlineSupport
from learning_checkpoint import rng,set_rng
from agent import ACTOR_KEY,CRITIC_KEY


class TrainingLive(ContactLive):
    """Same native Actor/PD/physical hook with arbitrary qualified real source."""
    def __init__(self,e,stage,coefficients,source_records):
        super().__init__(e,stage,coefficients)
        self.sources={r['case']:r for r in source_records}

    def start(self,spec):
        e=self.e;learner_rng=rng();record=self.sources[spec['source']['case']]
        key=('formal_native_source',record['case'])
        if key not in self.cache:
            path=self.stage/'runs/s0-native-sources-03'/record['evidence_file']
            import hashlib
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record['source_sha256']
            with gzip.open(path,'rb') as f:self.cache[key]=pickle.load(f)
        x=self.cache[key];index=spec['source']['control'];saved=x['states'][index]
        restore_source(e,saved)
        prior=x['physics'][(index-10)*4:index*4:4]
        assert len(prior)==10 and all(np.linalg.norm(p['force'])==0 for p in prior)
        self.history=ForceHistory(np.array([np.asarray(p['pelvis_matrix']).T@np.asarray(p['force']) for p in prior]))
        self.ema=ForceEMA();self.guard=OnlineSupport(**saved['online_support'])
        self.last_z=saved['actual_last_z'].to(e.standing.device)
        t=e.tether;t.body=mujoco.mj_name2id(e.sim.model,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link')
        self.spec=copy.deepcopy(spec);self.i=0;self.done=False
        for cfg in [e.cfg,t.cfg]:
            cfg.body='right_wrist_yaw_link';cfg.local_point=[0.,0.,0.];cfg.damping=8.;cfg.max_force=10.
        if spec['mode']==1:
            p=copy.deepcopy(spec['partner']);p['apply_force']=spec['force']
            p['onset_s']=float(t.time)
            p['release_s']=p['onset_s']+p['distance_m']/p['plateau_speed_mps']+p['accel_s']+p['hold_s']
            p['end_s']=p['release_s']+4.
            assert p['end_s']<e.base.max_episode_length*e.dt,'Native time-limit would truncate recovery'
            for cfg in [e.cfg,t.cfg]:
                cfg.enabled=spec['force'];cfg.stiffness=p['stiffness'];cfg.activation_time=p['onset_s'];cfg.release_time=p['release_s']
            t.ctft_partner_spec=p;t.ctft_partner_origin=t.point_state()[0].copy()
            t.measure=types.MethodType(measure,t);t.measure()
            self.controls=int(np.ceil((p['end_s']-float(t.time))/.02-1e-9));self.native_commands=None
        else:
            e.cfg.enabled=t.cfg.enabled=False;t.measure=types.MethodType(self.native_measure,t);t.measure()
            full=len(x['controls'])-(1 if x['boundary'] is not None else 0)
            self.native_commands=[c['z'].clone() for c in x['controls'][index:min(index+250,full)]]
            self.controls=len(self.native_commands);assert self.controls>0
        e.trace=[];e.sim.simulate_at_each_physics_step=self.guarded;e.qualification_active=True
        set_rng(learner_rng)

    def observation(self):
        e=self.e;m,d=e.sim.model,e.sim.data;t=e.tether
        point=d.xpos[t.body].copy();jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,t.body)
        R=d.xmat[self.pelvis].reshape(3,3)
        a=actor_fields(force_history=self.history,known_mount=[0.,0.,0.],wrist_world=point,
            root_world=d.qpos[:3],wrist_velocity_world=jac@d.qvel,root_velocity_world=d.qvel[:3],world_from_body=R,public_mode=self.spec['mode'])
        if self.spec['mode']==1:
            p,v=partner(t.ctft_partner_spec,float(t.time));angle=np.deg2rad(t.ctft_partner_spec['direction_deg'])
            rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
            p,v=rotation@p,rotation@v;anchor=t.ctft_partner_origin+p
            active=t.ctft_partner_spec['apply_force'] and t.ctft_partner_spec['onset_s']<=t.time<t.ctft_partner_spec['release_s']
            raw=t.ctft_partner_spec['stiffness']*(anchor-point)+8.*(v-jac@d.qvel) if active else np.zeros(3)
            c=critic_fields(current_partner_world=anchor,current_partner_velocity_world=v,current_raw_force_world=raw,wrist_world=point,world_from_body=R)
        else:c=np.zeros(9,dtype=np.float32)
        return dict(e.obs,**{ACTOR_KEY:torch.tensor(a[None],device=e.standing.device),CRITIC_KEY:torch.tensor(c[None],device=e.standing.device)})

    def step(self,action,before,z):
        row=super().step(action,before,z)
        if self.spec['mode']==0:phase=0
        else:
            p=self.e.tether.ctft_partner_spec;t=float(self.e.tether.time)
            moving_end=p['onset_s']+p['distance_m']/p['plateau_speed_mps']+p['accel_s']
            phase=1 if t<moving_end else 2 if t<p['release_s'] else 3
        row['ctft_replay_phase']=torch.tensor([[phase]],dtype=torch.int64)
        raw=self.e.base.extras['aux_rewards']
        native=sum(float(raw[k].detach().cpu().reshape(-1)[0])*v for k,v in self.coefficients.items())
        e=self.e;m,d=e.sim.model,e.sim.data;point=d.xpos[e.tether.body].copy()
        jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,e.tether.body)
        cost=contact_cost(filtered_force_n=self.ema.value,applied_force_n=e.tether.force,
            wrist_velocity_mps=jac@d.qvel,root_velocity_mps=d.qvel[:3],guidance_mode=self.spec['mode'])
        assert torch.equal(row['aux_rewards']['ctft_total_native'],torch.tensor([[native+cost['contact_native_reward']]],dtype=torch.float32))
        row['ctft_reward_components']=dict(native_aux=torch.tensor([[native]],dtype=torch.float64),
            contact_aux=torch.tensor([[cost['contact_native_reward']]],dtype=torch.float64))
        return row

    def restore(self,saved):
        # Parent restores the complete data/RNG/solver graph, then explicitly
        # bind the generalized K/direction observer (never a saved closure).
        super().restore(saved)
        if self.spec['mode']==1:
            p=self.e.tether.ctft_partner_spec
            for cfg in [self.e.cfg,self.e.tether.cfg]:cfg.stiffness=p['stiffness']
            self.e.tether.measure=types.MethodType(measure,self.e.tether)
