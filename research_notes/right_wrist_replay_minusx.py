"""Issue #19: full no-force replay validation, then matched minus-X control.

Three CPU episodes: no-force BFM, no-force replay, then gated -X BFM+tether.
Original wrapper, Tether and PD stay; previously saved +X is never rerun.
No model training, shared source edit, WBC replacement, GPU or Bridge queue use.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MUJOCO_GL']='disable'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse,hashlib,json,math,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import right_wrist_pair as previous
from right_wrist_pair import recovered,write,norm,average,OVERRIDES,PHASES
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=recovered.OUTPUT
BASELINE_PATH=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-wrist-20261028T054245Z/attempt-01/outputs/no_force.json')
BASELINE_SHA='ef6e2dfab4e314b59636870f1abc719c304192758420be190af307991bcf4898'


from right_wrist_anchor import json_scalar, write, select_anchor, replay_row, condition_summary, comparison

PLUS_PATH=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-wrist-anchor-20261028T062503Z/attempt-02/outputs/closed_loop.json')
PLUS_SHA='e03f6b23bb426c884a79a33c232e5c45aa0b06036590a2d416468577496fbd36'
CONDITIONS=('no_force','no_force_replay','minus_x')


def audit_anchor_velocity(time,delta):
    # Observation only: actual control still comes from unmodified Tether.
    if not 1.<time<3.:return [0.,0.,0.]
    u=(time-1.)/2.
    return [v*6*u*(1-u)/2. for v in delta]


def first_difference(a,b,path=''):
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return path+': keys differ'
        for k in a:
            found=first_difference(a[k],b[k],path+'/'+str(k))
            if found:return found
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):return path+': lengths differ'
        for i,(x,y) in enumerate(zip(a,b)):
            found=first_difference(x,y,path+'/'+str(i))
            if found:return found
    elif a!=b:return path+': '+repr(a)+' != '+repr(b)
    return None


def trace_match(reference,condition,full_physics=True):
    # This is a strict diagnostic equality gate, not a new survival threshold.
    physics_fields=['time','force_time','qpos','qvel','torque','wrist','root','support','foot_normal_forces','force','failed']
    actor_fields=['time','action','obs','z']
    def extract(e):
        return dict(initial=e['initial'],
            physics=e['physics'] if full_physics else [{k:r[k] for k in physics_fields} for r in e['physics']],
            actor=[{k:r[k] for k in actor_fields} for r in e['actor']])
    lengths={n:[len(reference[n]),len(condition[n])] for n in ['physics','actor']}
    full=(lengths=={'physics':[1200,1200],'actor':[300,300]} and
        reference['physics_dt']==condition['physics_dt']==.005 and
        abs(reference['physics'][-1]['time']-6.)<1e-10 and
        abs(condition['physics'][-1]['time']-6.)<1e-10)
    first=first_difference(extract(reference),extract(condition))
    deltas={k:max((abs(x-y) for a,b in zip(reference['physics'],condition['physics'])
        for x,y in zip(a[k],b[k])),default=0.) for k in ['qpos','qvel','torque']}
    deltas['action']=max((abs(x-y) for a,b in zip(reference['actor'],condition['actor'])
        for x,y in zip(a['action'],b['action'])),default=0.)
    no_force=all(not any(r['force']) and not r['active'] for e in [reference,condition] for r in e['physics'])
    return dict(passed=full and first is None and no_force,full_horizon=full,counts=lengths,
        exact_trace=first is None,first_difference=first,all_forces_zero_and_inactive=no_force,max_abs_delta=deltas,
        physics_time_grid_exact=[r['time'] for r in reference['physics']]==[r['time'] for r in condition['physics']],
        action_time_grid_exact=[r['time'] for r in reference['actor']]==[r['time'] for r in condition['actor']])


def require_replay_gate(reference,replay):
    gate=trace_match(reference,replay)
    live=sum(r['bfm_evaluated'] for r in replay['actor'])
    indices=[r['source_index'] for r in replay['actor']]
    gate.update(replay_live_bfm_calls=live,replay_indices_exact=indices==list(range(300)))
    gate['passed']=gate['passed'] and live==0 and gate['replay_indices_exact']
    if not gate['passed']:raise ValueError('No-force replay implementation/runtime diagnostic failed: '+json.dumps(gate))
    return gate


def analyze_all(episodes,plus=None):
    result={'conditions':{k:condition_summary(e) for k,e in episodes.items()},'comparisons':{}}
    if 'no_force_replay' in episodes:
        result['replay_gate']=trace_match(episodes['no_force'],episodes['no_force_replay'])
    if 'minus_x' in episodes:
        result['comparisons']['minus_x_minus_no_force']=comparison(episodes['no_force'],episodes['minus_x'])
    if plus is not None:
        result['conditions']['plus_x_archived']=condition_summary(plus)
        result['comparisons']['plus_x_minus_no_force']=comparison(episodes['no_force'],plus)
        if 'minus_x' in episodes:result['comparisons']['minus_x_minus_plus_x']=comparison(plus,episodes['minus_x'])
    # Always retain signed force; direction is measured against both fixed world axes.
    for name,e in list(episodes.items())+([('plus_x_archived',plus)] if plus is not None else []):
        s=result['conditions'][name];active=[r for r in e['physics'] if r['active']]
        s['active_force_mean_xyz_n']=average(active,'force')
        s['active_positive_fx_fraction']=sum(r['force'][0]>0 for r in active)/len(active) if active else None
        angles=sorted(math.degrees(math.acos(max(-1.,min(1.,-r['force'][0]/norm(r['force'])))))
            for r in active if norm(r['force'])>=.05)
        s['force_direction_angle_to_minus_x_deg']={'min':min(angles),'median':angles[len(angles)//2],'max':max(angles)} if angles else None
    return result


def runtime(result):
    archived=json.loads(BASELINE_PATH.read_text())
    assert recovered.digest(BASELINE_PATH)==BASELINE_SHA
    assert recovered.digest(PLUS_PATH)==PLUS_SHA
    plus=json.loads(PLUS_PATH.read_text())
    result['plus_x_reference']={'path':str(PLUS_PATH),'sha256':PLUS_SHA}
    write(OUTPUT/'plus_x_reference.json',plus)
    origin=select_anchor(archived)
    result['anchor_source']={'path':str(BASELINE_PATH),'sha256':BASELINE_SHA,'time_s':1.,'origin':origin}
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
        point_jac=np.zeros((3,m.nv))
        def point():
            return d.xpos[env.tether.body]+d.xmat[env.tether.body].reshape(3,3)@np.asarray(env.tether.cfg.local_point)
        def observed_physics():
            before_time=float(env.tether.time)
            before_point=point().copy()
            mujoco.mj_jac(m,d,point_jac,None,before_point,env.tether.body)
            before_velocity=(point_jac@d.qvel).copy()
            original_physics()
            error_before=env.tether.anchor-before_point
            spring=cfg.stiffness*error_before
            damping=cfg.damping*(np.asarray(audit_anchor_velocity(before_time,trajectory[1]))-before_velocity)
            raw_force=spring+damping
            audited=raw_force*min(1.,cfg.max_force/max(float(np.linalg.norm(raw_force)),1e-12)) if env.tether.active else np.zeros(3)
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
                point_before=before_point.tolist(),point_velocity_before=before_velocity.tolist(),
                position_error_before=error_before.tolist(),position_error_after=(env.tether.anchor-wrist).tolist(),
                wrist_from_anchor_origin=(wrist-np.asarray(origin)).tolist(),
                spring_force=spring.tolist(),damping_force=damping.tolist(),audit_force=audited.tolist(),
                wrist=wrist.tolist(),root=root.tolist(),
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
            is_replay=episode['name']=='no_force_replay'
            if is_replay:
                values=replay_row(episodes['no_force']['actor'],env.steps,env.steps*env.dt)
                action=torch.tensor(values,dtype=torch.float32,device='cpu').reshape(1,-1)
            else:
                action=original_act(obs,z,*args,**kwargs)
            episode['actor'].append(dict(time=env.steps*env.dt,action=action.reshape(-1).tolist(),
                obs={k:obs[k].tolist() for k in env.keys},z=z.tolist(),bfm_evaluated=not is_replay,
                source_index=env.steps,action_source='matched_no_force_replay' if is_replay else 'live_frozen_bfm'))
            return action
        env.reset=observed_reset;env.model.act=observed_act
        trajectory=(np.asarray(origin),np.asarray([.03,0.,0.]));episodes={}
        for name,enabled in [('no_force',False),('no_force_replay',False),('minus_x',True)]:
            if name=='minus_x':
                result['replay_gate']=require_replay_gate(episodes['no_force'],episodes['no_force_replay'])
                trajectory=(np.asarray(origin),np.asarray([-.03,0.,0.]))
                cfg.direction=[-1.,0.,0.]
                print('GATE full no-force action/state/PD replay exact; -X enabled',flush=True)
            cfg.enabled=enabled
            episode={'name':name,'enabled':enabled,'physics_dt':float(m.opt.timestep),'physics':[],'actor':[]}
            result['stage']=name+'_rollout'
            print('MILESTONE '+result['stage'],flush=True)
            trajectory,rows,metrics=evaluate(env,policy,True,cfg.seed,trajectory=trajectory)
            episode.update(trajectory=[v.tolist() for v in trajectory],control_rows=rows,metrics=metrics)
            print('EPISODE_BEFORE_SAVE '+json.dumps(metrics,default=json_scalar),flush=True)
            write(OUTPUT/(name+'.json'),episode);episodes[name]=episode
            print('EPISODE '+json.dumps(metrics),flush=True)
            result['analysis']=analyze_all(episodes,plus)
            write(OUTPUT/'result.json',result)
            if name=='no_force':
                result['archived_no_force_gate']=trace_match(archived,episode,full_physics=False)
                if not result['archived_no_force_gate']['passed']:
                    raise ValueError('No-force reference differs from archived baseline; no comparison allowed')
            elif name=='no_force_replay':
                result['replay_gate']=require_replay_gate(episodes['no_force'],episode)
                write(OUTPUT/'replay-gate.json',result['replay_gate'])
            if metrics['fall_rate'] or len(episode['physics'])!=1200:
                if name!='minus_x':raise ValueError(name+': incomplete no-force diagnostic; debug runtime before interpretation')
                result.update(runtime_status='executed',scientific_outcome='minus_x_negative',
                    first_boundary='minus_x: original failure/termination or incomplete 6s horizon')
                break
            if name=='minus_x':
                pair=result['analysis']['comparisons']['minus_x_minus_no_force']
                if not pair['full_horizon_pair_valid']:raise ValueError('Minus-X initial/prefix mismatch')
                first=next(row for row in episode['physics'] if row['active'])
                if norm(first['position_error_before'])>1e-10:raise ValueError('Initial position preload')
                if result['analysis']['conditions']['minus_x']['force_law_reconstruction_max_error_n']>1e-9:
                    raise ValueError('Force instrumentation differs from original Tether')
        result['analysis']=analyze_all(episodes,plus)
        result.update(runtime_status='executed',stage='finished',cuda_initialized=torch.cuda.is_initialized(),
            all_parameters_cpu=all(p.device.type=='cpu' for p in env.model.parameters()),
            all_parameters_frozen=not any(p.requires_grad for p in env.model.parameters()),
            no_parameter_gradients=all(p.grad is None for p in env.model.parameters()),model_eval=not env.model.training)
        if not result.get('scientific_outcome'):
            result['scientific_outcome']='no_force_replay_and_minus_x_completed'
            result['first_boundary']=None
        result['completed_conditions']=list(episodes)
    finally:
        env.reset=original_reset;env.model.act=original_act
        env.sim.simulate_at_each_physics_step=original_physics;env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analyze',type=Path)
    args=parser.parse_args()
    if args.analyze:
        episodes={n:json.loads((args.analyze/(n+'.json')).read_text()) for n in CONDITIONS if (args.analyze/(n+'.json')).exists()}
        plus=json.loads((args.analyze/'plus_x_reference.json').read_text())
        print(json.dumps(analyze_all(episodes,plus),indent=2));return 0
    result=recovered.preflight()
    result.update(protocol_overrides=OVERRIDES,condition_directions={'no_force':[1,0,0],'no_force_replay':[1,0,0],'minus_x':[-1,0,0]},script_sha256=recovered.digest(Path(__file__)),
        first_boundary=None,scientific_outcome=None,
        protocol='Exact full 6s no-force action replay gate, then 3cm world -X; archived matched +X; original PD unchanged')
    if result['runtime_status']!='ready':write(OUTPUT/'result.json',result);return 2
    try:runtime(result)
    except Exception as exc:
        result.update(runtime_status='technical_error',first_boundary=type(exc).__name__+': '+str(exc))
        raise
    finally:
        write(OUTPUT/'result.json',result);print('RESULT '+json.dumps(result),flush=True)
    return 0 if result.get('scientific_outcome')=='no_force_replay_and_minus_x_completed' else 2

if __name__=='__main__':raise SystemExit(main())
