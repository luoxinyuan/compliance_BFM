"""Issue #19: activation-aligned anchor, then original-action replay ablation.

Three CPU episodes: matched no-force, live frozen BFM+tether, replay+tether.
Only the final arm replaces online BFM actor outputs; original PD/wrapper stays.
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


def json_scalar(value):
    # Original ComplianceEnv may return numpy.bool_ on a true failure branch.
    # Preserve that failure as a JSON boolean instead of losing the negative run.
    if type(value).__module__=='numpy' and hasattr(value,'item'):
        return value.item()
    raise TypeError('Unsupported result value: '+type(value).__name__)


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=json_scalar))


def select_anchor(episode):
    """Post-step wrist at t=1s equals the next pre-force state, not the 1.005s state."""
    rows=[r for r in episode['physics'] if abs(r['time']-1.)<=1e-9]
    if len(rows)!=1:raise ValueError('Exactly one no-force t=1s physics sample required')
    row=rows[0]
    if any(row['force']):raise ValueError('Anchor reference must be no force')
    if len(row['wrist'])!=3 or not all(math.isfinite(v) for v in row['wrist']):raise ValueError('Invalid anchor')
    return list(row['wrist'])


def replay_row(records,index,time):
    if index<0 or index>=len(records):raise IndexError('Replay exhausted; never repeat final action')
    row=records[index]
    if abs(row['time']-time)>1e-10:raise ValueError('Replay timing mismatch')
    action=row['action']
    if len(action)!=29 or not all(math.isfinite(v) for v in action):raise ValueError('Invalid replay action')
    return list(action)


def audit_anchor_velocity(time):
    """For logging/force-law verification only, never used to drive the simulation."""
    if not 1.<time<3.:return [0.,0.,0.]
    u=(time-1.)/2.
    return [.03*6*u*(1-u)/2.,0.,0.]


def condition_summary(episode):
    s=previous.summarize(episode)
    s['control_decisions']=s.pop('actor_calls')
    s['live_bfm_calls']=sum(r['bfm_evaluated'] for r in episode['actor'])
    p=episode['physics'];active=[r for r in p if r['active']]
    s['force_rms_xyz_n']=[math.sqrt(sum(r['force'][i]**2 for r in p)/len(p)) for i in range(3)]
    s['force_abs_peak_xyz_n']=[max(abs(r['force'][i]) for r in p) for i in range(3)]
    s['force_impulse_xyz_ns']=[sum(r['force'][i] for r in p)*episode['physics_dt'] for i in range(3)]
    s['first_failed_physics_time_s']=next((r['time'] for r in p if r['failed']),None)
    s['force_law_reconstruction_max_error_n']=max(norm([a-b for a,b in zip(r['force'],r['audit_force'])]) for r in p)
    s['max_support_loss_after_onset_samples']=sum(not any(r['support']) for r in p if r['force_time']>=1.)
    s['active_samples']=len(active)
    if active:
        s['active_force_rms_n']=math.sqrt(sum(norm(r['force'])**2 for r in active)/len(active))
        s['active_cap_fraction']=sum(norm(r['force'])>=5.-1e-8 for r in active)/len(active)
        s['first_applied_force']={k:active[0][k] for k in ['force_time','force','spring_force','damping_force','point_velocity_before','position_error_before']}
        significant=[r for r in active if norm(r['force'])>=.05]
        angles=sorted(math.degrees(math.acos(max(-1.,min(1.,r['force'][0]/norm(r['force']))))) for r in significant)
        s['force_direction_angle_to_plus_x_deg']={'min':min(angles),'median':angles[len(angles)//2],'max':max(angles)} if angles else None
        s['active_negative_fx_fraction']=sum(r['force'][0]<0 for r in active)/len(active)
    s['phase_details']={}
    for phase,(start,end) in PHASES.items():
        rows=[r for r in p if start<=r['force_time']<end]
        s['phase_details'][phase]={
            'force_rms_xyz_n':[math.sqrt(sum(r['force'][i]**2 for r in rows)/len(rows)) for i in range(3)] if rows else None,
            'position_error_mean_xyz_m':average(rows,'position_error_after'),
            'position_error_rms_m':math.sqrt(sum(norm(r['position_error_after'])**2 for r in rows)/len(rows)) if rows else None,
            'relative_to_anchor_origin_mean_xyz_m':average(rows,'wrist_from_anchor_origin'),
            'root_tilt_mean_deg':average(rows,'tilt_deg')}
    return s


def comparison(reference,condition):
    result=previous.analyze(reference,condition)
    # The previous analyzer's field names describe its original pair only.
    result['reference']=reference['name'];result['condition']=condition['name']
    result.pop('episodes');result.pop('limitation')
    valid_prefix=result['initial_state_and_actor_observation_equal'] and result['pre_contact_qpos_max_delta']<=1e-10 and result['pre_contact_actor_max_delta']<=1e-10
    result['matching_initial_and_prefix']=valid_prefix
    result['common_samples']=min(len(reference['physics']),len(condition['physics']))
    count=result['common_samples']
    rp,cp=reference['physics'][:count],condition['physics'][:count]
    result['common_end_time_s']=cp[-1]['time']
    result['full_horizon_pair_valid']=result.pop('pair_valid')
    result.pop('paired_final_delta_m')  # Different terminal times must never be subtracted.
    result['paired_common_horizon_delta_m']={k:[cp[-1][k][i]-rp[-1][k][i] for i in range(3)]
        for k in ['wrist','wrist_relative_root','root']}
    result['common_horizon_metrics']={name:{
        'force_rms_n':math.sqrt(sum(norm(r['force'])**2 for r in rows)/count),
        'force_peak_n':max(norm(r['force']) for r in rows),
        'root_height_min_m':min(r['root'][2] for r in rows),
        'tilt_max_deg':max(r['tilt_deg'] for r in rows)}
        for name,rows in [('reference',rp),('condition',cp)]}
    result['torque_rms_delta']=math.sqrt(sum((a-b)**2 for r,c in zip(reference['physics'],condition['physics']) for a,b in zip(r['torque'],c['torque']))/(29*result['common_samples']))
    return result


def analyze_all(episodes):
    result={'conditions':{k:condition_summary(e) for k,e in episodes.items()},'comparisons':{}}
    if 'closed_loop' in episodes:
        result['comparisons']['closed_loop_minus_no_force']=comparison(episodes['no_force'],episodes['closed_loop'])
    if 'replay' in episodes:
        result['comparisons']['replay_minus_no_force']=comparison(episodes['no_force'],episodes['replay'])
        result['comparisons']['closed_loop_minus_replay']=comparison(episodes['replay'],episodes['closed_loop'])
        base=episodes['no_force']['actor'];replay=episodes['replay']['actor']
        result['replay_actions_exactly_match_no_force']=all(a['action']==b['action'] and a['time']==b['time'] for a,b in zip(base,replay))
        result['same_absolute_tether_trajectory']=episodes['closed_loop']['trajectory']==episodes['replay']['trajectory']
    return result


def runtime(result):
    archived=json.loads(BASELINE_PATH.read_text())
    assert recovered.digest(BASELINE_PATH)==BASELINE_SHA
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
            damping=cfg.damping*(np.asarray(audit_anchor_velocity(before_time))-before_velocity)
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
            is_replay=episode['name']=='replay'
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
        for name,enabled in [('no_force',False),('closed_loop',True),('replay',True)]:
            cfg.enabled=enabled
            episode={'name':name,'enabled':enabled,'physics_dt':float(m.opt.timestep),'physics':[],'actor':[]}
            result['stage']=name+'_rollout'
            print('MILESTONE '+result['stage'],flush=True)
            trajectory,rows,metrics=evaluate(env,policy,True,cfg.seed,trajectory=trajectory)
            episode.update(trajectory=[v.tolist() for v in trajectory],control_rows=rows,metrics=metrics)
            print('EPISODE_BEFORE_SAVE '+json.dumps(metrics,default=json_scalar),flush=True)
            write(OUTPUT/(name+'.json'),episode);episodes[name]=episode
            print('EPISODE '+json.dumps(metrics),flush=True)
            result['analysis']=analyze_all(episodes)
            write(OUTPUT/'result.json',result)
            if name=='no_force':
                archived_match=(episode['initial']==archived['initial'] and
                    len(episode['physics'])==len(archived['physics']) and
                    all(a['qpos']==b['qpos'] and a['qvel']==b['qvel'] for a,b in zip(episode['physics'],archived['physics'])) and
                    all(a['action']==b['action'] for a,b in zip(episode['actor'],archived['actor'])))
                result['archived_no_force_state_and_action_reproduced']=archived_match
                if not archived_match:raise ValueError('No-force reference changed; anchor/replay pairing invalid')
            if metrics['fall_rate'] or len(episode['physics'])!=1200:
                result.update(runtime_status='executed',scientific_outcome=name+'_negative',
                    first_boundary=name+': original failure/termination or incomplete 6s horizon')
                break
            if name=='closed_loop':
                pair=result['analysis']['comparisons']['closed_loop_minus_no_force']
                if not pair['full_horizon_pair_valid']:raise ValueError('Closed-loop initial/prefix mismatch; replay not authorized')
                first=next(row for row in episode['physics'] if row['active'])
                if norm(first['position_error_before'])>1e-10:raise ValueError('Anchor did not remove initial position preload')
                if result['analysis']['conditions']['closed_loop']['force_law_reconstruction_max_error_n']>1e-9:
                    raise ValueError('Force instrumentation differs from original Tether')
                print('GATE closed_loop completed 6s and pairing/anchor/force-law verified; replay enabled',flush=True)
        result['analysis']=analyze_all(episodes)
        result.update(runtime_status='executed',stage='finished',cuda_initialized=torch.cuda.is_initialized(),
            all_parameters_cpu=all(p.device.type=='cpu' for p in env.model.parameters()),
            all_parameters_frozen=not any(p.requires_grad for p in env.model.parameters()),
            no_parameter_gradients=all(p.grad is None for p in env.model.parameters()),model_eval=not env.model.training)
        if not result.get('scientific_outcome'):
            result['scientific_outcome']='anchor_and_replay_completed'
            result['first_boundary']=None
        result['completed_conditions']=list(episodes)
    finally:
        env.reset=original_reset;env.model.act=original_act
        env.sim.simulate_at_each_physics_step=original_physics;env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analyze',type=Path)
    args=parser.parse_args()
    if args.analyze:
        episodes={n:json.loads((args.analyze/(n+'.json')).read_text()) for n in ['no_force','closed_loop','replay'] if (args.analyze/(n+'.json')).exists()}
        print(json.dumps(analyze_all(episodes),indent=2));return 0
    result=recovered.preflight()
    result.update(protocol_overrides=OVERRIDES,script_sha256=recovered.digest(Path(__file__)),
        first_boundary=None,scientific_outcome=None,
        protocol='No-force t=1s anchor; 3cm world +X; original live actor vs matched no-force action replay; PD unchanged')
    if result['runtime_status']!='ready':write(OUTPUT/'result.json',result);return 2
    try:runtime(result)
    except Exception as exc:
        result.update(runtime_status='technical_error',first_boundary=type(exc).__name__+': '+str(exc))
        raise
    finally:
        write(OUTPUT/'result.json',result);print('RESULT '+json.dumps(result),flush=True)
    return 0 if result.get('scientific_outcome')=='anchor_and_replay_completed' else 2

if __name__=='__main__':raise SystemExit(main())
