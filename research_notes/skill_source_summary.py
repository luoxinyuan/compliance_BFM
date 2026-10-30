"""Recompute saved physical metrics only. No model loading or mj_step."""
import argparse,json
from pathlib import Path
import numpy as np

def summarize(ep):
    rows=ep['physics'];cs=ep['control'];initial=ep['initial']
    arr=lambda k:np.asarray([r[k] for r in rows])
    times=arr('force_time')-3;active=arr('active').astype(bool)
    noforce=ep['label']=='goal_free'
    phases={'enter':(0,.75),'execute':(.75,3.25),'exit':(3.25,4),'settle':(4,6)} if noforce else {'move':(0,4),'hold':(4,5),'release':(5,6)}
    T0=np.asarray(initial['torso_matrix']);Ts=arr('torso_matrix')
    torso=np.degrees(np.arccos(np.clip((np.einsum('nij,ij->n',Ts,T0)-1)/2,-1,1)))
    wrist=arr('wrist');shoulder=arr('shoulder_xyz');root=arr('root');w0=np.asarray(initial['wrist']);s0=np.asarray(initial['shoulder_xyz']);r0=np.asarray(initial['root'])
    q=arr('qpos');q0=np.asarray(initial['qpos']);force=arr('force');error=arr('anchor')-wrist
    result=dict(label=ep['label'],seconds=len(rows)/200,boundary=ep['boundary'],phases={})
    for name,(lo,hi) in phases.items():
        mask=(times>=lo-1e-8)&(times<hi-1e-8)
        # Release is not scored as active tracking. Actual Tether flag decides force presence.
        if not noforce:mask &= (~active if name=='release' else active)
        if not mask.any():continue
        ix=np.flatnonzero(mask);sel=[c for c in cs if lo-1e-8<=c['relative_time']<hi-1e-8]
        f=force[mask];loads=arr('foot_normal_force_n')[mask]>1
        entry=dict(samples=len(ix),force_rms_n=float(np.sqrt(np.mean(np.sum(f*f,axis=1)))),force_peak_n=float(np.linalg.norm(f,axis=1).max()),force_mean_xyz_n=f.mean(0).tolist(),force_cap_fraction=float((np.linalg.norm(f,axis=1)>=5-1e-6).mean()),
            tracking_rms_cm=None if noforce or name=='release' else float(np.sqrt(np.mean(np.sum(error[mask]**2,axis=1)))*100),
            tracking_xyz_rms_cm=None if noforce or name=='release' else (np.sqrt(np.mean(error[mask]**2,axis=0))*100).tolist(),
            wrist_end_xyz_cm=((wrist[ix[-1]]-w0)*100).tolist(),wrist_mean_xyz_cm=((wrist[mask]-w0).mean(0)*100).tolist(),
            wrist_relative_root_mean_xyz_cm=(((wrist-root)[mask]-(w0-r0)).mean(0)*100).tolist(),
            shoulder_end_xyz_cm=((shoulder[ix[-1]]-s0)*100).tolist(),shoulder_mean_xyz_cm=((shoulder[mask]-s0).mean(0)*100).tolist(),
            root_end_xyz_cm=((root[ix[-1]]-r0)*100).tolist(),root_min_height_cm=float(arr('height')[mask].min()*100),pelvis_tilt_max_deg=float(arr('tilt_deg')[mask].max()),
            torso_rotation_max_deg=float(torso[mask].max()),waist_change_max_abs_deg=np.degrees(abs(q[mask,19:22]-q0[19:22]).max(0)).tolist(),
            both_feet_load_fraction=float(loads.all(1).mean()),any_foot_load_fraction=float(loads.any(1).mean()),contact_tangent_speed_max_cm_s=float(arr('contact_tangent_speed_max_mps')[mask].max()*100),
            undesired_contacts_max=int(arr('undesired_ground_contacts')[mask].max()),actual_joint_limit_excess_max_deg=float(np.degrees(arr('joint_limit_excess_max_rad')[mask].max())),
            torque_saturation_mean_fraction=float(arr('torque_saturation')[mask].mean()),action_clip_joint_time=float(np.mean([c['action_saturation_fraction'] for c in sel])),
            qPD_step_max_deg=float(np.degrees(max(c['q_PD_step_max_rad'] for c in sel))),latent_angle_max_deg=max(c['z_angle_deg'] for c in sel))
        delta=np.asarray([c['delta_q_PD'] for c in sel])
        entry['same_obs_additional_qPD_rms_deg']={g:float(np.degrees(np.sqrt(np.mean(delta[:,sl]**2)))) for g,sl in {'legs':slice(0,12),'waist':slice(12,15),'left_arm':slice(15,22),'right_arm':slice(22,29)}.items()}
        result['phases'][name]=entry
    result['final_root_speed_mps']=float(np.linalg.norm(rows[-1]['qvel'][:3]))
    result['max_actual_joint_20ms_change_deg']=float(np.degrees(abs(np.diff(np.asarray([c['pre_qpos'][7:] for c in cs]),axis=0))).max())
    result['control_observation_count']=len(cs)
    result['actual_active_samples']=int(active.sum())
    if active.any():
        result['all_active_tracking_rms_cm']=float(np.sqrt(np.mean(np.sum(error[active]**2,axis=1)))*100)
        result['all_active_force_rms_n']=float(np.sqrt(np.mean(np.sum(force[active]**2,axis=1))))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    d={}
    for name in ['goal_free','standing_contact','goal_contact']:
        f=a.outputs/(name+'.json')
        if f.exists():d[name]=summarize(json.loads(f.read_text()))
    a.destination.write_text(json.dumps(d,indent=2,allow_nan=False))
    print('SAVED_TRACE_METRICS',list(d))
