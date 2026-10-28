"""Owner-authorized minimal right-wrist pair; original frozen BFM/Tether only.

CPU, seed 0, two 6s episodes, no training/rendering. Run under the bounded
right_wrist_supervisor.py. --analyze DIR recomputes metrics from recorded traces.
The imported run.evaluate uses baseline=True; its high-level input is ignored.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MUJOCO_GL'] = 'disable'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
# Reuse the successful launcher's asset validation and CPFS IPC repair.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_action_contract_check as recovered
OUTPUT = recovered.OUTPUT
OVERRIDES = dict(direction=[1., 0., 0.], displacement_min=.03,
                 displacement_max=.03, max_force=5., activation_time=1.,
                 anchor_delay=0., release_time=4.5)
PHASES = {'pre_contact': (0., 1.), 'ramp': (1., 3.),
          'hold_late': (3.5, 4.5), 'released': (4.5, 6.000001)}


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def norm(x):
    return math.sqrt(sum(v*v for v in x))


def average(rows, key):
    values = [r[key] for r in rows]
    if not values:
        return None
    if isinstance(values[0], list):
        return [sum(v[i] for v in values)/len(values) for i in range(len(values[0]))]
    return sum(values)/len(values)


def summarize(episode):
    p, a = episode['physics'], episode['actor']
    phases = {}
    for name, (start, end) in PHASES.items():
        subset = [r for r in p if start <= r['force_time'] < end]
        phases[name] = dict(samples=len(subset), force_mean_vector_n=average(subset, 'force'),
            force_rms_n=math.sqrt(sum(norm(r['force'])**2 for r in subset)/len(subset)) if subset else None,
            wrist_mean_world_m=average(subset, 'wrist'), wrist_mean_relative_root_m=average(subset, 'wrist_relative_root'),
            wrist_mean_pelvis_frame_m=average(subset, 'wrist_pelvis_frame'), root_mean_m=average(subset, 'root'),
            anchor_error_mean_m=average(subset, 'anchor_error'))
    return dict(simulated_seconds=len(p)*episode['physics_dt'], failed=episode['metrics']['fall_rate'] != 0,
        original_metrics=episode['metrics'], actor_calls=len(a), physics_samples=len(p),
        force_rms_n=math.sqrt(sum(norm(r['force'])**2 for r in p)/len(p)),
        force_peak_n=max(norm(r['force']) for r in p),
        cap_fraction=sum(norm(r['force']) >= 5.-1e-8 for r in p)/len(p),
        root_height_min_m=min(r['root'][2] for r in p),
        tilt_max_deg=max(r['tilt_deg'] for r in p),
        root_rpy_abs_max_deg=[max(abs(r['root_rpy_deg'][i]) for r in p) for i in range(3)],
        any_foot_contact_fraction=sum(any(r['support']) for r in p)/len(p),
        double_foot_contact_fraction=sum(all(r['support']) for r in p)/len(p),
        no_foot_contact_times_s=[r['time'] for r in p if not any(r['support'])],
        foot_xy_displacement_max_m=max(r['foot_displacement'] for r in p),
        contact_normal_force_mean_n=average(p, 'foot_normal_forces'),
        action_abs_max=max(abs(x) for r in a for x in r['action']),
        action_saturation_fraction=sum(abs(x)>=1 for r in a for x in r['action'])/(29*len(a)),
        torque_saturation_fraction=average(p, 'torque_saturation'),
        wrist_final_world_m=p[-1]['wrist'], wrist_final_relative_root_m=p[-1]['wrist_relative_root'],
        root_final_m=p[-1]['root'], phases=phases)


def analyze(baseline, tether):
    bp, tp = baseline['physics'], tether['physics']
    ba, ta = baseline['actor'], tether['actor']
    initial_equal = baseline['initial'] == tether['initial']
    same_grid = len(bp)==len(tp) and all(abs(b['time']-t['time']) < 1e-10 for b,t in zip(bp,tp))
    pairs = list(zip(bp,tp))
    pre = [(b,t) for b,t in pairs if t['force_time'] < 1.]
    pre_delta = max((abs(x-y) for b,t in pre for x,y in zip(b['qpos'],t['qpos'])), default=0.)
    action_pre_delta=max((abs(x-y) for b,t in zip(ba,ta) if t['time']<1.-1e-10
                          for x,y in zip(b['action'],t['action'])),default=0.)
    summary = {'no_force': summarize(baseline), 'tether': summarize(tether)}
    comparisons = {}
    for phase,(start,end) in PHASES.items():
        ps=[(b,t) for b,t in pairs if start<=t['force_time']<end]
        actors=[(b,t) for b,t in zip(ba,ta) if start<=t['time']<end]
        entry={}
        for key in ['wrist', 'wrist_relative_root', 'wrist_pelvis_frame', 'root']:
            entry[key+'_paired_mean_delta_m']=[sum(t[key][i]-b[key][i] for b,t in ps)/len(ps) for i in range(3)] if ps else None
        entry['action_rms_delta']=math.sqrt(sum((x-y)**2 for b,t in actors for x,y in zip(b['action'],t['action']))/(29*len(actors))) if actors else None
        entry['action_max_abs_delta']=max((abs(x-y) for b,t in actors for x,y in zip(b['action'],t['action'])),default=None)
        comparisons[phase]=entry
    final={key:[tp[-1][key][i]-bp[-1][key][i] for i in range(3)] for key in ['wrist','wrist_relative_root','root']}
    valid = initial_equal and same_grid and pre_delta<=1e-10 and action_pre_delta<=1e-10
    # A completed or negative scientific result is not a reason to tune/retry.
    return dict(pair_valid=valid, initial_state_and_actor_observation_equal=initial_equal,
        same_physics_grid=same_grid, pre_contact_qpos_max_delta=pre_delta,
        pre_contact_actor_max_delta=action_pre_delta, episodes=summary,
        paired_phase_deltas=comparisons, paired_final_delta_m=final,
        limitation='Action changes demonstrate closed-loop feedback, not a causal separation of active assistance from passive deflection or PD impedance.')


def runtime(result):
    sys.path.insert(0, str(ROOT))
    import multiprocessing as mp
    mp.set_start_method('forkserver', force=True)
    assert OUTPUT is not None and Path.cwd().resolve()==OUTPUT.resolve()
    (OUTPUT/'ipc').mkdir()
    mp.current_process()._config['tempdir']='ipc'
    with mp.Manager() as manager:
        q=manager.Queue();q.put('ready');assert q.get()=='ready'
    import numpy as np
    import torch
    import mujoco
    from humanoidverse.force_compliance.config import ComplianceConfig
    from humanoidverse.force_compliance.environment import ComplianceEnv
    from humanoidverse.force_compliance.run import evaluate
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    for f,h in recovered.PINS.items():
        assert recovered.digest(ROOT/f)==h, f
    default=json.loads((ROOT/'humanoidverse/force_compliance/default.json').read_text())
    cfg=ComplianceConfig(**(default|OVERRIDES))
    assert cfg.stiffness==80 and cfg.damping==8 and cfg.episode_duration==6 and cfg.seed==0
    cfg.enabled=False
    result['stage']='original_ComplianceEnv_constructor'
    env=ComplianceEnv(recovered.MODEL,cfg,device='cpu')
    result.update(model_loaded=True,original_wrapper_used=True,model_class=type(env.model).__name__,
        wrapper_class=type(env.wrapped).__name__, standing_source=env.standing_source,
        latent=env.standing.tolist(),latent_norm=float(env.standing.norm()),
        actor_keys=env.keys, torch_version=torch.__version__,mujoco_version=mujoco.__version__)
    write(OUTPUT/'standing_latent.json',result['latent'])
    torch.save(env.standing.cpu(),OUTPUT/'standing_latent.pt')
    # evaluate requires a .distribution().mean even though baseline=True ignores it.
    # This is only a zero placeholder, never a controller/action/PD replacement.
    policy=SimpleNamespace(distribution=lambda observation:SimpleNamespace(mean=torch.zeros_like(env.standing)))
    original_reset, original_act = env.reset, env.model.act
    original_physics = env._physics_step
    episode=None
    try:
        m,d=env.sim.model,env.sim.data
        pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
        feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link') for side in ('left','right')]
        effort=env.base.torque_limits.cpu().reshape(-1).numpy()
        def point():
            return d.xpos[env.tether.body]+d.xmat[env.tether.body].reshape(3,3)@np.asarray(env.tether.cfg.local_point)
        def observed_physics():
            before_time=float(env.tether.time)
            before_point=point().copy()
            original_physics()
            wrist=point();root=d.qpos[:3].copy();rot=d.xmat[pelvis].reshape(3,3)
            normal=[0.,0.]
            for con_index,contact in enumerate(d.contact[:d.ncon]):
                bodies=[int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2])]
                if 0 not in bodies:continue
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,max(bodies)) or ''
                if 'ankle' in name or 'foot' in name:
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,con_index,wrench)
                    normal[0 if 'left' in name else 1]+=float(wrench[0])
            rpy=[math.atan2(rot[2,1],rot[2,2]),math.asin(np.clip(-rot[2,0],-1,1)),math.atan2(rot[1,0],rot[0,0])]
            episode['physics'].append(dict(time=float(env.tether.time),force_time=before_time,
                force=env.tether.force.tolist(),anchor=env.tether.anchor.tolist(),active=bool(env.tether.active),
                point_before=before_point.tolist(),wrist=wrist.tolist(),root=root.tolist(),
                wrist_relative_root=(wrist-root).tolist(),wrist_pelvis_frame=(rot.T@(wrist-root)).tolist(),
                anchor_error=float(np.linalg.norm(env.tether.anchor-wrist)),
                qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),root_rpy_deg=[math.degrees(x) for x in rpy],
                tilt_deg=math.degrees(math.acos(float(np.clip(rot[2,2],-1,1)))),
                support=list(env.last_support),foot_positions=d.xpos[feet].tolist(),foot_normal_forces=normal,
                foot_displacement=float(np.linalg.norm(d.xpos[feet,:2]-episode['initial']['foot_xy'],axis=1).max()),
                torque=d.ctrl[6:].tolist(),torque_saturation=float(np.mean(abs(d.ctrl[6:])>=effort-1e-6)),
                failed=bool(env.failed)))
        def observed_reset(*args,**kwargs):
            value=original_reset(*args,**kwargs)
            episode['initial']={'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),
                'actor_obs':{k:env.obs[k].tolist() for k in env.keys},
                'foot_xy':d.xpos[feet,:2].tolist(),'standing':env.standing.tolist()}
            env.sim.simulate_at_each_physics_step=observed_physics
            return value
        def observed_act(obs,z,*args,**kwargs):
            action=original_act(obs,z,*args,**kwargs)
            episode['actor'].append(dict(time=env.steps*env.dt,action=action.reshape(-1).tolist(),
                obs={k:obs[k].tolist() for k in env.keys},z=z.tolist()))
            return action
        env.reset=observed_reset;env.model.act=observed_act
        trajectory=None;episodes={}
        for name,enabled in [('no_force',False),('tether',True)]:
            cfg.enabled=enabled
            episode={'name':name,'enabled':enabled,'physics_dt':float(m.opt.timestep),'physics':[],'actor':[]}
            result['stage']=name+'_rollout'
            print('MILESTONE '+result['stage'],flush=True)
            trajectory,rows,metrics=evaluate(env,policy,True,cfg.seed,trajectory=trajectory)
            episode.update(trajectory=[v.tolist() for v in trajectory],control_rows=rows,metrics=metrics)
            write(OUTPUT/(name+'.json'),episode);episodes[name]=episode
            print('EPISODE '+json.dumps(metrics),flush=True)
            if metrics['fall_rate'] and name=='no_force':
                result.update(runtime_status='scientific_negative',first_boundary='No-force pair member fell; no tether execution')
                return
        result['analysis']=analyze(episodes['no_force'],episodes['tether'])
        result.update(runtime_status='executed',stage='finished',cuda_initialized=torch.cuda.is_initialized(),
            all_parameters_cpu=all(p.device.type=='cpu' for p in env.model.parameters()),
            all_parameters_frozen=not any(p.requires_grad for p in env.model.parameters()),
            no_parameter_gradients=all(p.grad is None for p in env.model.parameters()),
            model_eval=not env.model.training,scientific_outcome='paired_measurement_complete')
        if not result['analysis']['pair_valid']:
            result['first_boundary']='Pair initial state, pre-contact trace, or duration mismatch; causal paired comparison invalid'
            result['scientific_outcome']='pair_invalid'
        elif episodes['tether']['metrics']['fall_rate']:
            result['first_boundary']='Original failure/termination criterion reached under tether'
            result['scientific_outcome']='tether_negative'
        else:
            result['first_boundary']=None
    finally:
        env.reset=original_reset;env.model.act=original_act
        env.sim.simulate_at_each_physics_step=original_physics;env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze',type=Path)
    args=parser.parse_args()
    if args.analyze:
        print(json.dumps(analyze(*[json.loads((args.analyze/(n+'.json')).read_text()) for n in ('no_force','tether')]),indent=2));return 0
    result=recovered.preflight()
    result.update(protocol_overrides=OVERRIDES,script_sha256=recovered.digest(Path(__file__)),first_boundary=None)
    if result['runtime_status']!='ready':write(OUTPUT/'result.json',result);return 2
    try:
        runtime(result)
    except Exception as exc:
        result.update(runtime_status='technical_error',first_boundary=type(exc).__name__+': '+str(exc))
        raise
    finally:
        write(OUTPUT/'result.json',result)
        print('RESULT '+json.dumps(result),flush=True)
    return 0 if result.get('scientific_outcome')=='paired_measurement_complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
