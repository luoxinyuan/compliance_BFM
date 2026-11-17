"""S0 contact Task + native learner exact fresh-process continuation probe.

This is disposable correctness work. The reference branch and fresh-process
branch use one immutable complete snapshot, actual sampled 20ms actions, real
Tether physics and one subsequent complete FB/CPR/aux update. No formal data.
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

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def study(e,original,contract,out):
    import numpy as np
    import torch
    import mujoco
    from agent import build,ACTOR_KEY,CRITIC_KEY
    from contact_scale_qualify import restore_source
    from contact_interface_probe import measure,partner
    from contact_replay import build_buffers,cat
    from learning_checkpoint import atomic_save,fingerprint,one_update,file_sha,digest,cpu,rng,set_rng
    from source_qualify import capture_state
    from features import ForceHistory,actor_fields,critic_fields
    from reward import ForceEMA,contact_cost
    from quality import OnlineSupport
    from locomotion_rollout import UnsafePhysics
    from m8_stage import reset_common,advance
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import metadata,canonical

    args=json.loads((out/'protocol.json').read_text())
    scale=json.loads((STAGE/'runs/s0-aux-calibration-01/scale-result.json').read_text())
    assert scale['status']=='S0_SAMPLING_AND_FIXED_SCALE_QUALIFIED' and scale['scale_locked']
    # Preserve the exact FP32 scale actually stored in the model buffer.
    fixed_scale=scale['scalar_float32']
    speed(e)
    (out/'backend-prefix').mkdir()
    backend(e,original,out/'backend-prefix')
    e.cfg.max_force=5.
    reset_common(e,original,original.act)
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(True)
    agent,identity=build(aux_scale=fixed_scale,compile_execution=False)
    torch.set_grad_enabled(False)
    write(out/'agent-identity.json',identity)
    updates=[]
    if args['mode']=='reference':
        buffers,groups,sources=build_buffers(ROOT,identity['original_training_config']['aux_rewards_scaling'])
        write(out/'replay-provenance.json',sources)
        # Exactly the previously checked disposable wiring pattern, not a
        # substitute for the formal 8192/64 warm-up or formal online sampling.
        for i in range(4):
            agent.actor_enabled=i>=2
            with torch.enable_grad():
                updates.append(one_update(agent,buffers,i))
        (out/'warm-updates.json').write_text(json.dumps(updates,indent=2)+'\n')
        learner_rng=rng()
        with gzip.open(STAGE/'runs/s0-contact-interface-01/force.pkl.gz','rb') as f:
            archived=pickle.load(f)
        saved=next(s for s in archived['snapshots'] if abs(s['native_source_progress']['time_s']-4.)<.005)
        del archived
    else:
        from restore_diagnose import restore
        package=Path(args['package'])
        joint=json.loads((package/'manifest.json').read_text())
        assert joint['complete']
        for name,record in joint['files'].items():
            assert file_sha(package/name)==record['sha256']
        learner=package/'learner'
        lm=json.loads((learner/'manifest.json').read_text())
        for name,record in lm['files'].items():
            assert file_sha(learner/name)==record['sha256']
        state=torch.load(learner/'learner.pt',map_location='cpu',weights_only=False)
        model=torch.load(learner/'model.pt',map_location='cpu',weights_only=False)
        assert state['aux_scale']==fixed_scale
        buffers=restore(agent,state,model)
        groups=state['expert_groups']
        saved=torch.load(package/'task.pt',weights_only=False)
        learner_rng=state['rng']
        del state,model

    spec=copy.deepcopy(saved['tether']['ctft_partner_spec'])
    # Callback identity and its spec binding are part of the explicit snapshot
    # contract; data-only state_of intentionally does not pickle bound methods.
    for cfg in [e.cfg,e.tether.cfg]:
        cfg.enabled=spec['apply_force']
        cfg.body='right_wrist_yaw_link'
        cfg.local_point=[0.,0.,0.]
        cfg.stiffness=80.
        cfg.damping=8.
        cfg.max_force=10.
        cfg.activation_time=spec['onset_s']
        cfg.release_time=spec['release_s']
    restore_source(e,saved)
    e.tether.measure=types.MethodType(measure,e.tether)
    history=ForceHistory(saved['ctft_force_history'])
    ema=ForceEMA(np.asarray(saved['ctft_force_ema']).copy())
    guard=OnlineSupport(**saved['online_support'])
    native_hook=e.sim.simulate_at_each_physics_step
    def guarded():
        native_hook()
        s=guard.update(e.contact_sample['foot_normal_force_n'])
        e.trace[-1]['ctft_online_support']=s
        e.trace[-1]['ctft_raw_force']=e.tether.ctft_raw_force.tolist()
        if s['stop']:
            raise UnsafePhysics(dict(e.trace[-1],ctft_stop_reason='continuous_no_effective_support_gt_50ms'))
    e.sim.simulate_at_each_physics_step=guarded
    e.qualification_active=True
    e.trace=[]
    set_rng(learner_rng)
    last_z=saved['actual_last_z'].to(e.standing.device)
    m,d=e.sim.model,e.sim.data
    pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    off=torch.tensor(contract['q_PD_offset'],device=e.standing.device)
    qscale=torch.tensor(contract['q_PD_scale'],device=e.standing.device)

    def observation():
        point=d.xpos[e.tether.body].copy()
        jac=np.zeros((3,m.nv))
        mujoco.mj_jac(m,d,jac,None,point,e.tether.body)
        R=d.xmat[pelvis].reshape(3,3)
        a=actor_fields(force_history=history,known_mount=[0.,0.,0.],wrist_world=point,
            root_world=d.qpos[:3],wrist_velocity_world=jac@d.qvel,root_velocity_world=d.qvel[:3],world_from_body=R,public_mode=1)
        delta,v=partner(spec,float(e.tether.time))
        anchor=e.tether.ctft_partner_origin+delta
        active=spec['apply_force'] and spec['onset_s']<=e.tether.time<spec['release_s']
        raw=80.*(anchor-point)+8.*(v-jac@d.qvel) if active else np.zeros(3)
        c=critic_fields(current_partner_world=anchor,current_partner_velocity_world=v,
            current_raw_force_world=raw,wrist_world=point,world_from_body=R)
        return dict(e.obs,**{ACTOR_KEY:torch.tensor(a[None],device=e.standing.device),CRITIC_KEY:torch.tensor(c[None],device=e.standing.device)})

    if args['mode']=='reference':
        package=out/'joint-checkpoint'
        pending=out/'joint-checkpoint.pending'
        pending.mkdir()
        state=capture_state(e,last_z,dict(scope='actual_learner_and_contact_boundary',time_s=float(e.tether.time)))
        state.update(ctft_force_history=history.state(),ctft_force_ema=ema.value.copy(),online_support=copy.deepcopy(vars(guard)))
        torch.save(state,pending/'task.pt')
        lm=atomic_save(agent,buffers,groups,pending/'learner',step=4)
        files={'task.pt':dict(sha256=file_sha(pending/'task.pt'),bytes=(pending/'task.pt').stat().st_size),
            'learner/manifest.json':dict(sha256=file_sha(pending/'learner/manifest.json'),bytes=(pending/'learner/manifest.json').stat().st_size)}
        joint=dict(complete=True,scope='native_eager_learner_and_actual_contact_Task',files=files,
            runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            actor_mean=False,controls_after_resume=12,then_actual_agent_update=1,
            callback='contact_interface_probe.measure + local guarded with explicit spec binding',
            frozen_aux_scale=fixed_scale,initial_learner=lm['expected_initial'],initial_observation=digest(observation()),
            bytes_including_learner=sum(p.stat().st_size for p in pending.rglob('*') if p.is_file()))
        write(pending/'manifest.json',joint)
        for path in pending.rglob('*'):
            if path.is_file():
                with path.open('rb') as f:os.fsync(f.fileno())
        os.rename(pending,package)
    else:
        assert fingerprint(agent)==joint['initial_learner'],'Joint initial learner mismatch'
        assert digest(observation())==joint['initial_observation'],'Joint initial causal observation mismatch'
    rows=[]
    for i in range(12):
        assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
        before=observation()
        with torch.no_grad():action=agent.act(before,e.standing,mean=False)
        qpd=off+(action*5).clamp(-contract['clip'],contract['clip'])*qscale/5
        e.previous_z=e.standing.clone()
        assert not advance(e,action),'Native termination in actual contact continuation'
        last_z=e.standing.clone()
        history.append_actual(e.tether.force,d.xmat[pelvis].reshape(3,3))
        filtered=ema.update(e.tether.force)
        point=d.xpos[e.tether.body].copy()
        jac=np.zeros((3,m.nv))
        mujoco.mj_jac(m,d,jac,None,point,e.tether.body)
        contact=contact_cost(filtered_force_n=filtered,applied_force_n=e.tether.force,
            wrist_velocity_mps=jac@d.qvel,root_velocity_mps=d.qvel[:3],guidance_mode=1)
        raw=e.base.extras['aux_rewards']
        reward=sum(float(raw[k].detach().cpu().reshape(-1)[0])*v for k,v in identity['original_training_config']['aux_rewards_scaling'].items())+contact['contact_native_reward']
        row=dict(observation=cpu(before),action=cpu(action),z=cpu(e.standing),
            next=dict(observation=cpu(observation()),terminated=torch.zeros(1,1,dtype=torch.bool),truncated=torch.zeros(1,1,dtype=torch.bool)),
            aux_rewards=dict(ctft_total_native=torch.tensor([[reward]],dtype=torch.float32)))
        rows.append(row)
        write(out/'physical-progress.json',dict(completed_controls=i+1,physics_rows=len(e.trace)))
    buffers['train'].extend(cat(rows))
    with torch.enable_grad():updated=one_update(agent,buffers,4)
    state=capture_state(e,last_z,dict(scope='after_actual_continuation',time_s=float(e.tether.time)))
    state.update(ctft_force_history=history.state(),ctft_force_ema=ema.value.copy(),online_support=copy.deepcopy(vars(guard)))
    # Raw bytes stay in task.pt; only the previously proved zero-valid-length
    # solver arrays are masked for comparison, exactly as original S0 restore.
    state=canonical(state,metadata(m,d))
    answer=dict(actual_physics_sha256=digest(e.trace),actual_12_rows_sha256=digest(rows),
        final_complete_Task_sha256=digest(state),
        learner={k:updated[k] for k in ['state','sample_trace','metrics']},
        new_continuation_controls=12,new_continuation_ticks=len(e.trace),frozen_aux_scale=fixed_scale)
    write(out/'continuation.json',answer)
    if args['mode']=='resume':
        expected=json.loads((Path(args['package']).parent/'continuation.json').read_text())
        different=[k for k in expected if expected[k]!=answer[k]]
        write(out/'comparison.json',dict(status='PASS' if not different else 'FAIL',different=different,
            initial_learner_exact=True,initial_causal_observation_exact=True,
            actual_sampled_actions_and_physics_exact=answer['actual_12_rows_sha256']==expected['actual_12_rows_sha256'] and answer['actual_physics_sha256']==expected['actual_physics_sha256'],
            complete_Task_exact=answer['final_complete_Task_sha256']==expected['final_complete_Task_sha256'],
            subsequent_full_agent_update_exact=answer['learner']==expected['learner']))
        assert not different,('Fresh-process joint continuation mismatch',different)
    e.qualification_active=False
    return dict(status='CONTACT_JOINT_REFERENCE_COMPLETE' if args['mode']=='reference' else 'CONTACT_JOINT_RESTORE_EXACT',
        disposable=True,formal_training=False,actual_new_rows=12,updates=5 if args['mode']=='reference' else 1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'protocol.json',dict(mode='resume' if a.package else 'reference',package=str(a.package) if a.package else None,
        scope='S0 exact same-process configuration, actual 12 sampled controls then full native update; no formal U/T',
        max_physics_ticks=3000,wall_cap_s=1200))
    write(a.output/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1200))
    import mujoco
    original=mujoco.mj_step
    count=dict(actual=0,reserved=0)
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));count['reserved']+=n
        assert count['reserved']<=3000
        r=original(*args,**kw);count['actual']+=n;return r
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output
    start=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'cost.json',dict(wall_s=time.monotonic()-start,physics=count,formal_transitions=0))


if __name__=='__main__':main()
