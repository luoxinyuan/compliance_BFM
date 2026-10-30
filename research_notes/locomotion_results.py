"""Summarize saved locomotion A/B records; kinematics/reward only, no mj_step."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
from reward_skill_summary import summarize


def enrich(ep,root=ROOT):
    import mujoco
    from humanoidverse.envs.g1_env_helper.rewards import LocomotionReward,get_center_of_mass_linvel,rot2eul,get_xmat
    m=mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml'))
    d=mujoco.MjData(m);reward=LocomotionReward(move_angle=0,move_speed=.3)
    cs=ep['control'];rs=ep['physics'];out=summarize(ep)
    v=[];rr=[]
    for r in rs:
        d.qpos[:]=r['qpos'];d.qvel[:]=r['qvel'];mujoco.mj_forward(m,d)
        vel=get_center_of_mass_linvel(m,d);yaw=rot2eul(get_xmat(m,d,'pelvis'))[-1]
        v.append([vel[0]*np.cos(yaw)+vel[1]*np.sin(yaw),-vel[0]*np.sin(yaw)+vel[1]*np.cos(yaw),np.linalg.norm(vel[:2])]);rr.append(reward.compute(m,d))
    v=np.array(v);rr=np.array(rr);t=np.array([r['force_time']-3 for r in rs])
    for phase,(lo,hi) in {'enter':(0,.75),'execute':(.75,3.25),'exit':(3.25,4),'settle':(4,6)}.items():
        mask=(t>=lo-1e-8)&(t<hi-1e-8)
        if not mask.any():continue
        x=out['phases'][phase];selected=[c for c in cs if lo-1e-8<=c['relative_time']<hi-1e-8]
        x.update(native_torso_velocity_forward_lateral_speed_mps=v[mask].mean(0).tolist(),
            native_forward_std_mps=float(v[mask,0].std()),native_reward_mean=float(rr[mask].mean()),
            speed_in_027_033_fraction=float(((v[mask,2]>=.27)&(v[mask,2]<=.33)).mean()),
            target_clip_joint_time_fraction=float(np.mean([c.get('target_clip_fraction',0) for c in selected])))
    actualq=np.asarray([c['pre_qpos'][7:] for c in cs]);vel=np.array([r['qvel'][6:] for r in rs]);tau=np.array([r['torque'] for r in rs])
    out['actual_joint_20ms_max_change_deg']=float(np.degrees(abs(np.diff(actualq,axis=0))).max())
    out['actual_joint_speed_max_rad_s']=float(abs(vel).max())
    out['torque_max_nm']=float(abs(tau).max());out['torque_rms_nm']=float(np.sqrt(np.square(tau).mean()))
    out['final_root_speed_mps']=float(np.linalg.norm(rs[-1]['qvel'][:3]))
    out['final_root_displacement_xyz_cm']=((np.array(rs[-1]['root'])-ep['initial']['root'])*100).tolist()
    out['last_1s_q_PD_step_max_deg']=float(np.degrees(max(c['q_PD_step_max_rad'] for c in cs if c['relative_time']>=5-1e-8))) if t.max()>=5 else None
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('outputs',type=Path);parser.add_argument('destination',type=Path);a=parser.parse_args()
    old=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-reward-stage-20261030A/physics-attempt-01/outputs')
    files=[old/'standing.json',old/'move-ego-0-0.3.json']+[a.outputs/(n+'.json') for n in ['native_50hz','deploy_target_clip']]
    exact=a.outputs.parents[1]/'physics-exact-attempt-01/outputs/deploy_target_clip_exact.json'
    if exact.exists():files.append(exact)
    result={}
    for f in files:
        if f.exists():result[f.stem]=enrich(json.loads(f.read_text()))
    a.destination.write_text(json.dumps(result,indent=2,allow_nan=False));print('SAVED_RECORD_SUMMARY_COMPLETED',list(result))
