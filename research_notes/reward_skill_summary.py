"""Recorded trajectory metrics only; no model or physics execution."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(ep):
    initial=ep['initial'];rows=ep['physics'];control=ep['control']
    if not rows:return {'empty':True,'boundary':ep['boundary']}
    a=lambda key:np.asarray([r[key] for r in rows],dtype=float)
    t=a('force_time')-3
    root=a('root');q=a('qpos');torso=a('torso_matrix')
    r0=np.asarray(initial['root']);q0=np.asarray(initial['qpos']);T0=np.asarray(initial['torso_matrix'])
    feet=a('feet_xyz');feet0=np.asarray(initial['feet_xyz'])
    com=a('com_xyz');com0=np.asarray(initial['com_xyz'])
    comvel=np.diff(np.concatenate([com0[None],com]),axis=0)/.005
    w,x,y,z=q[:,3:7].T
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    ego_forward=comvel[:,0]*np.cos(yaw)+comvel[:,1]*np.sin(yaw)
    ego_lateral=-comvel[:,0]*np.sin(yaw)+comvel[:,1]*np.cos(yaw)
    torso_delta=np.degrees(np.arccos(np.clip((np.einsum('nij,ij->n',torso,T0)-1)/2,-1,1)))
    phase_masks={'enter':(0,.75),'execute':(.75,3.25),'exit':(3.25,4),'settle':(4,6)}
    result={'label':ep['label'],'simulated_seconds':len(rows)/200,'boundary':ep['boundary'],
            'common_state_sha256':ep['common_state_sha256'],'phases':{}}
    for phase,(lo,hi) in phase_masks.items():
        mask=(t>=lo-1e-9)&(t<hi-1e-9)
        if not mask.any():continue
        idx=np.where(mask)[0];cs=[c for c in control if lo-1e-9<=c['relative_time']<hi-1e-9]
        positions={name:np.asarray([r['body_positions'][name] for r in rows])
                   for name in initial['body_positions']}
        wrist_info={}
        for side in ('left','right'):
            name=side+'_wrist_roll_link';wrist=positions[name];w0=np.asarray(initial['body_positions'][name])
            shoulder=positions[side+'_shoulder_pitch_link'];s0=np.asarray(initial['body_positions'][side+'_shoulder_pitch_link'])
            wrist_info[side]={'mean_world_height_cm':float(wrist[mask,2].mean()*100),
                'world_displacement_mean_xyz_cm':((wrist[mask]-w0).mean(axis=0)*100).tolist(),
                'relative_root_displacement_mean_xyz_cm':(((wrist-root)[mask]-(w0-r0)).mean(axis=0)*100).tolist(),
                'shoulder_displacement_mean_xyz_cm':((shoulder[mask]-s0).mean(axis=0)*100).tolist()}
        contact=a('support')[mask].astype(bool)
        native_arm_fraction=np.mean((positions['left_wrist_roll_link'][mask,2]>=.6)&
                                    (positions['left_wrist_roll_link'][mask,2]<=.8)&
                                    (positions['right_wrist_roll_link'][mask,2]>=1.))
        result['phases'][phase]={
            'duration_seconds':len(idx)/200,'wrist':wrist_info,
            'root_displacement_end_xyz_cm':((root[idx[-1]]-r0)*100).tolist(),
            'root_displacement_peak_norm_cm':float(np.linalg.norm(root[mask]-r0,axis=1).max()*100),
            'root_min_height_cm':float(a('height')[mask].min()*100),
            'pelvis_tilt_max_deg':float(a('tilt_deg')[mask].max()),
            'torso_rotation_from_initial_max_deg':float(torso_delta[mask].max()),
            'waist_change_max_abs_deg':np.degrees(abs(q[mask,19:22]-q0[19:22]).max(axis=0)).tolist(),
            'ego_com_forward_mean_mps':float(ego_forward[mask].mean()),
            'ego_com_forward_std_mps':float(ego_forward[mask].std()),
            'ego_com_lateral_mean_mps':float(ego_lateral[mask].mean()),
            'both_feet_support_fraction':float(contact.all(axis=1).mean()),
            'any_foot_support_fraction':float(contact.any(axis=1).mean()),
            'foot_max_lift_from_initial_cm':((feet[mask,:,2]-feet0[:,2]).max(axis=0)*100).tolist(),
            'foot_peak_xy_displacement_cm':(np.linalg.norm(feet[mask,:,:2]-feet0[:,:2],axis=2).max(axis=0)*100).tolist(),
            'arm_reward_height_bounds_fraction':float(native_arm_fraction),
            'actual_force_peak_n':float(np.linalg.norm(a('force')[mask],axis=1).max()),
            'joint_limit_excess_max_deg':float(np.degrees(a('joint_limit_excess_max_rad')[mask].max())),
            'torque_saturation_mean_fraction':float(a('torque_saturation')[mask].mean()),
            'undesired_contacts_max':int(a('undesired_ground_contacts')[mask].max()),
            'action_saturation_mean_fraction':float(np.mean([c['action_saturation_fraction'] for c in cs])) if cs else None,
            'q_PD_step_max_deg':float(np.degrees(max(c['q_PD_step_max_rad'] for c in cs))) if cs else None,
            'latent_angle_max_deg':max(c['z_angle_deg'] for c in cs) if cs else None,
        }
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    result={'metric_notes':'COM speed from 200Hz position differences; wrist heights are native wrist_roll bodies. No reference/force tracking metric in no-Tether qualification.', 'episodes':[]}
    for name in ['standing','raisearms-l-m','move-ego-0-0.3']:
        f=a.outputs/(name+'.json')
        if f.exists():result['episodes'].append(summarize(json.loads(f.read_text())))
    a.destination.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))
