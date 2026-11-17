"""Bounded frozen-native contact/causal-field qualification, no learning.

Two paired conditions share the actual native reset prefix. They constitute
one physical parent, not two independent source lineages. No probe replay or
model is admitted into formal U/T training by this script.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def partner(spec,t):
    """Continuous position/velocity; speed is plateau speed, not episode mean."""
    import numpy as np
    v=spec['plateau_speed_mps'];L=spec['distance_m'];r=spec['accel_s'];x=t-spec['onset_s']
    assert L/v>=r
    if x<=0:s,ds=0.,0.
    elif x<r:
        u=x/r;s=v*r*(u**3-.5*u**4);ds=v*(3*u*u-2*u**3)
    elif x<L/v:s=.5*v*r+v*(x-r);ds=v
    elif x<L/v+r:
        u=(x-L/v)/r;s=L-.5*v*r+v*r*(u-u**3+.5*u**4);ds=v*(1-3*u*u+2*u**3)
    else:s,ds=L,0.
    direction=np.array([1.,0.,0.]);return direction*s,direction*ds


def measure(self):
    import numpy as np
    from humanoidverse.force_compliance.interaction import spring_force
    spec=self.ctft_partner_spec;p,v=partner(spec,self.time)
    self.anchor=self.ctft_partner_origin+p;self.partner_velocity=v
    point,pv=self.point_state()
    self.active=bool(spec['apply_force'] and spec['onset_s']<=self.time<spec['release_s'])
    self.ctft_raw_force=80.*(self.anchor-point)+8.*(v-pv) if self.active else np.zeros(3)
    self.force=spring_force(self.anchor,point,v,pv,80.,8.,10.) if self.active else np.zeros(3)
    assert np.allclose(self.force,self.ctft_raw_force*min(1.,10./max(float(np.linalg.norm(self.ctft_raw_force)),1e-12)),atol=0,rtol=0)
    return point,self.force.copy()


def study(e,model,contract,out):
    import numpy as np
    import torch
    import mujoco
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from locomotion_rollout import UnsafePhysics
    from source_qualify import capture_state
    from features import ForceHistory,actor_fields,critic_fields
    from reward import ForceEMA,contact_cost
    from quality import OnlineSupport
    speed(e);(out/'backend-prefix').mkdir();backend(e,model,out/'backend-prefix')
    plan=json.loads((out/'protocol.json').read_text());results=[]
    m=e.sim.model;d=e.sim.data;pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    off=torch.tensor(contract['q_PD_offset'],device=e.standing.device);scale=torch.tensor(contract['q_PD_scale'],device=e.standing.device)
    for spec in plan['conditions']:
        e.cfg.max_force=5. # exact legacy reset-prefix config; no force is active
        reset_common(e,model,model.act);t=e.tether
        t.body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link')
        t.cfg.local_point=[0.,0.,0.];t.cfg.stiffness=80.;t.cfg.damping=8.;t.cfg.max_force=10.
        t.cfg.enabled=spec['apply_force'];e.cfg.enabled=spec['apply_force'];e.cfg.max_force=10.
        t.cfg.activation_time=spec['onset_s'];t.cfg.release_time=spec['release_s']
        t.ctft_partner_spec=copy.deepcopy(spec);t.ctft_partner_origin=t.point_state()[0].copy()
        t.measure=types.MethodType(measure,t);t.measure()
        # Prefix is genuinely executed and already includes ten force samples.
        prior=e.trace[-37::4];assert len(prior)==10 and all(np.linalg.norm(r['force'])==0 for r in prior)
        history=ForceHistory(np.array([np.asarray(r['pelvis_matrix']).T@np.asarray(r['force']) for r in prior]))
        ema=ForceEMA();guard=OnlineSupport(float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity)))
        for r in e.trace:guard.update(r['foot_normal_force_n'])
        native_hook=e.sim.simulate_at_each_physics_step
        def guarded():
            native_hook();s=guard.update(e.contact_sample['foot_normal_force_n'])
            e.trace[-1]['ctft_online_support']=s
            e.trace[-1]['ctft_raw_force']=t.ctft_raw_force.tolist()
            if s['stop']:raise UnsafePhysics(dict(e.trace[-1],ctft_stop_reason='continuous_no_effective_support_gt_50ms'))
        e.sim.simulate_at_each_physics_step=guarded;e.qualification_active=True
        rows=[];snapshots=[];start=len(e.trace);boundary=None;last_z=e.standing.clone()
        try:
            while t.time<spec['end_s']-1e-8:
                assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC']),'Contact probe local wall guard'
                point=d.xpos[t.body].copy();jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,t.body)
                pv=jac@d.qvel;R=d.xmat[pelvis].reshape(3,3)
                fields=actor_fields(force_history=history,known_mount=[0,0,0],wrist_world=point,
                   root_world=d.qpos[:3],wrist_velocity_world=pv,root_velocity_world=d.qvel[:3],world_from_body=R,public_mode=1)
                now=float(t.time)
                delta,velocity=partner(spec,now);current_anchor=t.ctft_partner_origin+delta
                active=bool(spec['apply_force'] and spec['onset_s']<=now<spec['release_s'])
                current_raw=80*(current_anchor-point)+8*(velocity-pv) if active else np.zeros(3)
                privileged=critic_fields(current_partner_world=current_anchor,current_partner_velocity_world=velocity,
                   current_raw_force_world=current_raw,wrist_world=point,world_from_body=R)
                assert fields.shape==(46,) and privileged.shape==(9,)
                # Data-only full snapshots at planned causal boundaries.
                if any(abs(now-mark)<.005 for mark in spec['snapshot_times_s']):
                    state=capture_state(e,last_z,dict(probe_condition=spec['name'],time_s=now))
                    state.update(ctft_force_history=history.state(),ctft_force_ema=ema.value.copy(),online_support=copy.deepcopy(vars(guard)))
                    snapshots.append(state)
                action=model.act(e.obs,e.standing,mean=True);assert torch.isfinite(action).all()
                qpd=off+(action*5).clamp(-contract['clip'],contract['clip'])*scale/5
                before_obs={k:v.detach().cpu().clone() for k,v in e.obs.items()}
                e.previous_z=e.standing.clone()
                e.obs,_,native_terminal,native_truncation,_=e.wrapped.step(action,to_numpy=False)
                e.steps+=1;e.previous_action=action.clone()
                e.history=np.roll(e.history,-1,axis=0);e.history[-1]=t.force
                terminated=bool(e.failed or torch.as_tensor(native_terminal).any())
                truncated=bool(torch.as_tensor(native_truncation).any());done=terminated or truncated
                history.append_actual(t.force,d.xmat[pelvis].reshape(3,3));filtered=ema.update(t.force)
                post_point=d.xpos[t.body].copy();post_jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,post_jac,None,post_point,t.body)
                wvel=post_jac@d.qvel
                cost=contact_cost(filtered_force_n=filtered,applied_force_n=t.force,wrist_velocity_mps=wvel,
                    root_velocity_mps=d.qvel[:3],guidance_mode=1)
                native_aux={k:float(v.detach().cpu().reshape(-1)[0]) for k,v in e.base.extras['aux_rewards'].items()}
                rows.append(dict(time_s=now,observation=before_obs,actor46=fields,privileged9=privileged,
                    z=e.standing.detach().cpu().clone(),previous_z=last_z.detach().cpu().clone(),action=action.detach().cpu().clone(),
                    q_PD=qpd.detach().cpu().clone(),contact_reward=cost,native_aux_raw=native_aux,
                    terminated=terminated,truncated=truncated,identity='frozen_native_mean_interface_probe_not_formal_replay'))
                last_z=e.standing.clone()
                if done:boundary=dict(kind='native_termination',time_s=float(t.time));break
        except UnsafePhysics as exc:boundary=dict(kind='hard_physics',row=exc.row)
        finally:e.qualification_active=False
        if rows and boundary is None:rows[-1]['truncated']=True
        physics=copy.deepcopy(e.trace[start:]);assert spec['apply_force'] or all(np.linalg.norm(r['force'])==0 for r in physics)
        with gzip.open(out/(spec['name']+'.pkl.gz'),'wb',compresslevel=1) as f:
            pickle.dump(dict(spec=spec,controls=rows,physics=physics,snapshots=snapshots,boundary=boundary),f,protocol=5)
        summary=dict(name=spec['name'],complete=boundary is None,boundary=boundary,control_rows=len(rows),physics_rows=len(physics),
            snapshots=len(snapshots),max_applied_force_n=max((float(np.linalg.norm(r['force'])) for r in physics),default=0.),
            schema_only=True,stochastic_execution_qualified=False,task_learner_restore_qualified=False)
        results.append(summary);write(out/'progress.json',results);print('CTFT_CONTACT_INTERFACE',json.dumps(summary),flush=True)
        if boundary:break
    result=dict(status='FROZEN_CONTACT_INTERFACE_RECORDED' if all(r['complete'] for r in results) else 'CONTACT_INTERFACE_PHYSICS_BOUNDARY',
        conditions=results,independent_physical_parents=1,formal_training=False,calibration_scale_locked=False,
        actual_actor_is_original_frozen=True,actual_optimized_reward_logged_separately=True,student_inputs_not_used=True)
    write(out/'interface-result.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    onset=3.;release=onset+.6/.12+.5+1.;end=release+4.
    common=dict(distance_m=.6,plateau_speed_mps=.12,accel_s=.5,onset_s=onset,release_s=release,end_s=end,
                snapshot_times_s=[3.,4.,release,release+1.])
    write(a.output/'protocol.json',dict(scope='train-only frozen causal-interface qualification; no learning or final scale',
        conditions=[dict(common,name='force',apply_force=True),dict(common,name='no_force',apply_force=False)],
        source='same verified actual native reset/standing prefix; one parent',max_physics_ticks=8000))
    write(a.output/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+900))
    import mujoco
    original=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));counts['reserved']+=n;assert counts['reserved']<=8000
        x=original(*args,**kw);counts['actual']+=n;return x
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output;started=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'cost.json',dict(wall_s=time.monotonic()-started,physics=counts,new_agent_updates=0,formal_transitions=0))


if __name__=='__main__':main()
