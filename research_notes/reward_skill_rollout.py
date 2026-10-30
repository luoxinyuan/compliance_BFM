"""Fixed standing -> native reward z -> standing; no teacher or learner."""
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path


class UnsafePhysics(RuntimeError):
    def __init__(self, row):
        self.row = copy.deepcopy(row)
        super().__init__('Pre-registered physical safety boundary at ' + str(row['time']))


def coefficient(t):
    if t < .75:
        x = max(0., t/.75)
    elif t < 3.25:
        return 1.
    elif t < 4.:
        x = (4.-t)/.75
    else:
        return 0.
    return x*x*(3.-2.*x)


def spherical(z0, z1, alpha):
    import torch
    if alpha <= 0: return z0
    if alpha >= 1: return z1
    theta = torch.acos(((z0*z1).sum()/256.).clamp(-1.,1.))
    if float(theta) < 1e-5:
        z = (1-alpha)*z0 + alpha*z1
    else:
        assert float(theta) < math.pi-1e-4, 'Antipodal native command has no unique shortest path'
        z = (torch.sin((1-alpha)*theta)*z0 + torch.sin(alpha*theta)*z1)/torch.sin(theta)
    return z * (16./z.norm())


def snapshot(env):
    import numpy as np
    import torch
    import mujoco
    m = env.sim.model
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(m,spec))
    mujoco.mj_getState(m,env.sim.data,state,spec)
    tensors = lambda obj: {k:v.tolist() for k,v in vars(obj).items() if torch.is_tensor(v)}
    return dict(mj_integration=state.tolist(),base_tensors=tensors(env.base),
        sim_tensors=tensors(env.sim),wrapped_tensors=tensors(env.wrapped),
        bfm_obs={k:v.tolist() for k,v in env.obs.items()},previous_z=env.previous_z.tolist(),
        previous_action=env.previous_action.tolist(),force_history=env.history.tolist(),steps=env.steps,
        common_step_counter=env.base.common_step_counter,
        history={k:v.tolist() for k,v in env.base.history_handler.history.items()},
        env_bookkeeping=dict(done=bool(env.done),failed=bool(env.failed),last_qpos=env.last_qpos.tolist(),
                             last_point=env.last_point.tolist(),last_support=env.last_support),
        tether={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(env.tether).items()
                if k in ['origin','delta','time','force','anchor','active','body','jac']})


def run_study(env, model, contract, out):
    import numpy as np
    import torch
    import mujoco
    root = Path(__file__).resolve().parents[1]
    runs = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
    read = lambda p: json.loads(Path(p).read_text())
    def save(name, obj):
        (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
    protocol = read(root/'research_notes/reward_skill_protocol.json')
    inference = Path(os.environ['BFM_REWARD_INFERENCE_OUTPUT'])
    ir = read(inference/'result.json')
    assert ir['status']=='completed' and ir['bfm_frozen']
    assert ir['model_sha256']==protocol['model_sha256']
    candidates={t:torch.tensor(read(inference/(t+'-latent.json'))['z'],device='cuda',dtype=torch.float32)
                for t in protocol['tasks']}
    for z in candidates.values():
        assert z.shape==(1,256) and torch.isfinite(z).all() and abs(float(z.norm())-16)<1e-4
    zero = read(runs/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics']
    initial0 = read(runs/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json')
    common = read(runs/protocol['physical_protocol']['initial_snapshot'])
    m = env.sim.model
    dev = env.standing.device
    assert dev.type=='cuda' and env.dt==.02 and abs(m.opt.timestep-.005)<1e-10 and env.interval==5
    scale=torch.tensor(contract['q_PD_scale'],device=dev)
    offset=torch.tensor(contract['q_PD_offset'],device=dev)
    qtarget=lambda a: offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
    original_act=model.act
    record=False
    chosen=env.standing
    control=[]
    last_q=None
    label='standing'
    def act(obs,z,mean=True):
        nonlocal last_q
        relative=float(env.tether.time)-3.
        alpha=coefficient(relative) if record and label!='standing' else 0.
        actual_z=spherical(env.standing,chosen,alpha)
        action=original_act(obs,actual_z,mean=True)
        if record:
            q=qtarget(action)
            row=dict(time=float(env.tether.time),relative_time=relative,alpha=alpha,
                executed_z=actual_z.tolist(),z_angle_deg=float(torch.rad2deg(torch.acos(((actual_z*env.standing).sum()/256.).clamp(-1,1)))),
                action=action[0].tolist(),q_PD=q[0].tolist(),bfm_obs={k:v.tolist() for k,v in obs.items()},
                pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),
                action_saturation_fraction=float(((action*5).abs()>=contract['clip']).float().mean()),
                q_PD_step_max_rad=0. if last_q is None else float((q-last_q).abs().max()))
            control.append(row);last_q=q.clone()
        return action
    model.act=act
    result=dict(status='running',episodes=[],first_failure=None)
    save('protocol.json',protocol)
    try:
        for label in ['standing']+protocol['tasks']:
            env.qualification_active=False;record=False;chosen=env.standing
            env.cfg.enabled=False;env.cfg.activation_time=3.;env.cfg.ramp_duration=4.;env.cfg.release_time=8.
            env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
            for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
                value=getattr(env.base,name)
                value.copy_(torch.as_tensor(initial0['base_tensors'][name],device=value.device,dtype=value.dtype))
            env.base.common_step_counter=initial0['common_step_counter']
            mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(m,env.sim.data)
            mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
            for _ in range(30):
                assert time.monotonic()<float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
                _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True)
                assert not done
            diffs={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max())
                   for k in ['qpos','qvel','wrist','root','force']}
            save(label+'-prefix-parity.json',diffs)
            assert max(diffs.values())==0, 'Original zero-force prefix parity failed'
            env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype))
            state=snapshot(env)
            if state!=common:
                save(label+'-snapshot-mismatch.json',dict(expected=common,actual=state))
                raise AssertionError('Full common standing snapshot mismatch')
            save(label+'-initial-state.json',state)
            start=len(env.trace);initial=copy.deepcopy(env.trace[-1]);boundary=None
            control=[];last_q=qtarget(env.previous_action).clone();record=True
            chosen=env.standing if label=='standing' else candidates[label]
            env.qualification_active=True
            # Same-observation interface precheck, no state updates or z fitting.
            check=[]
            for a in [0.,.25,.5,.75,1.]:
                z=spherical(env.standing,chosen,a);action=original_act(env.obs,z,mean=True)
                assert torch.isfinite(action).all()
                check.append(dict(alpha=a,z_norm=float(z.norm()),action=action.tolist(),q_PD=qtarget(action).tolist()))
            save(label+'-command-precheck.json',check)
            for high in range(60):
                assert time.monotonic()<float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
                try:
                    _,_,done,info=env.step(torch.zeros_like(env.standing),baseline=True)
                except UnsafePhysics as exc:
                    boundary=dict(kind='pre_registered_physical_boundary',physics=exc.row);break
                except FloatingPointError as exc:
                    boundary=dict(kind='nonfinite',message=str(exc),time=float(env.tether.time));break
                if info['failed'] or (done and high!=59):
                    boundary=dict(kind='native_failure_or_early_termination',time=float(env.tether.time),failed=info['failed']);break
            env.qualification_active=False;record=False
            rows=copy.deepcopy(env.trace[start:])
            episode=dict(label=label,initial=initial,physics=rows,control=control,boundary=boundary,
                         common_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
            save(label+'.json',episode)
            summary=dict(label=label,physics_samples=len(rows),simulated_seconds=len(rows)/200.,
                         complete=len(rows)==1200 and boundary is None and not env.failed,boundary=boundary)
            result['episodes'].append(summary)
            print('NATIVE_SKILL_BRANCH',label,summary['simulated_seconds'],summary['complete'],flush=True)
            if boundary is not None or not summary['complete']:
                result.update(status='physical-stopped',first_failure=summary);save('physical-result.json',result);break
            save('physical-result.json',result)
        else:
            result['status']='completed'
        save('physical-result.json',result)
        return result
    finally:
        model.act=original_act
        env.qualification_active=False
