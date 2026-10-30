"""Read-only audit of all fixed physical branches, no model load or training."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import mujoco

BASELINE=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-accumulated-20261030A/attempt-02/outputs')

def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def analyze(out,root):
 g=json.loads((out/'geometry.json').read_text());results={};episodes={}
 m=mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');q0=np.array(g['qpos'])
 for label in ['accumulated','standing_accumulated']:
  source=BASELINE if label=='accumulated' else out
  ep=json.loads((source/(label+'.json')).read_text());episodes[label]=ep;rows=ep['physics'];control=ep['control'];result={}
  for phase,lo,hi in [('active',3.,8.),('ramp',3.,7.),('hold',7.,8.),('release',8.,9.)]:
   # Native active flag controls the release boundary (floating time can be just below 8s).
   selected=[r for r in rows if (not r['active'] and r['force_time']>=8.)] if phase=='release' else [r for r in rows if r['active'] and lo<=r['force_time']<hi]
   cs=[r for r in control if lo-1e-8<=r['tether_time']<hi-1e-8]
   if not selected:result[phase]=None;continue
   e=np.array([np.array(r['anchor'])-r['wrist'] for r in selected]);force=np.array([r['force'] for r in selected]);w=np.array([r['wrist'] for r in selected]);shoulder=np.array([r['shoulder_xyz'] for r in selected]);rpos=np.array([r['root'] for r in selected]);q=np.array([r['qpos'] for r in selected]);dq=np.array([r['delta_q_PD'] for r in cs])
   margin=[];body_kin=[];torso=[]
   for r in selected:
    points=np.unique(np.array(r['foot_contact_xy']),axis=0)
    if len(points)>=3:
     try:
      hull=ConvexHull(points);margin.append(float(-np.max(hull.equations[:,:2]@np.array(r['com_xyz'])[:2]+hull.equations[:,2])))
     except Exception:pass
    d.qpos[:]=q0;d.qpos[29:]=np.array(r['qpos'])[29:];mujoco.mj_forward(m,d)
    body_kin.append(np.array(r['wrist'])-d.xpos[body])
    rot=np.array(r['torso_matrix'])@np.array(ep['initial']['torso_matrix']).T;torso.append(np.degrees(Rotation.from_matrix(rot).magnitude()))
   sphere_radius=np.linalg.norm(w-np.array(g['shoulder']),axis=1)
   wristdelta=w[-1]-np.array(ep['initial']['wrist']);rootdelta=rpos[-1]-np.array(ep['initial']['root']);shoulderdelta=shoulder[-1]-np.array(ep['initial']['shoulder_xyz'])
   result[phase]=dict(actual_wrist_radius_from_initial_shoulder_max_cm=100*float(sphere_radius.max()),outside_initial_arm_sphere_fraction=float(np.mean(sphere_radius>g['arm_radius_upper_bound_m'])),n=len(selected),duration_s=len(selected)/200,error_rms_cm=100*rms(np.linalg.norm(e,axis=1)),error_xyz_rms_cm=(100*np.sqrt(np.mean(e*e,axis=0))).tolist(),endpoint_error_cm=float(100*np.linalg.norm(e[-1])),force_rms_n=rms(np.linalg.norm(force,axis=1)),force_peak_n=float(np.linalg.norm(force,axis=1).max()),force_mean_xyz_n=force.mean(0).tolist(),force_cap_fraction=float(np.mean(np.linalg.norm(force,axis=1)>=4.9999)),wrist_delta_xyz_cm=(100*wristdelta).tolist(),root_delta_xyz_cm=(100*rootdelta).tolist(),shoulder_delta_xyz_cm=(100*shoulderdelta).tolist(),wrist_relative_root_delta_xyz_cm=(100*(wristdelta-rootdelta)).tolist(),body_contribution_to_endpoint_wrist_cm=(100*np.array(body_kin[-1])).tolist(),waist_joint_delta_endpoint_deg=np.degrees(q[-1,19:22]-q0[19:22]).tolist(),torso_rotation_max_deg=max(torso),root_height_min_cm=100*min(r['height'] for r in selected),root_tilt_max_deg=max(r['tilt_deg'] for r in selected),both_feet_loaded_fraction=float(np.mean([all(f>1e-6 for f in r['foot_normal_force_n']) for r in selected])),foot_slip_peak_cm_s=100*max(r['contact_tangent_speed_max_mps'] for r in selected),COM_support_margin_min_cm=None if not margin else 100*min(margin),undesired_ground_contacts=max(r['undesired_ground_contacts'] for r in selected),actual_joint_excess_max_deg=np.degrees(max(r['joint_limit_excess_max_rad'] for r in selected)),torque_saturation_fraction=float(np.mean([r['torque_saturation'] for r in selected])),action_saturation_fraction=float(np.mean([r['action_saturation'] for r in cs])),q_PD_limit_excess_max_deg=np.degrees(max(r['q_PD_limit_excess_max_rad'] for r in cs)),q_PD_step_max_deg=np.degrees(max(r['q_PD_step_max_rad'] for r in cs)),direct_q_PD_waist_rms_deg=np.degrees(rms(dq[:,12:15])),direct_q_PD_leg_rms_deg=np.degrees(rms(dq[:,:12])),direct_q_PD_arm_rms_deg=np.degrees(rms(dq[:,22:])),z_angle_max_deg=max(r['z_angle_deg'] for r in cs))
  result['complete']=len(rows)==1200 and ep['stop_boundary'] is None and not ep['failed'];result['stop_boundary']=ep['stop_boundary'];results[label]=result
 for label,result in results.items():
  for phase in ['active','ramp','hold']:
   if result[phase] and results['accumulated'][phase]:result[phase]['improvement_vs_C_pct']=100*(1-result[phase]['error_rms_cm']/results['accumulated'][phase]['error_rms_cm'])
 diagnostics={}
 for label,ep in episodes.items():
  source=BASELINE if label=='accumulated' else out
  state=json.loads((source/(label+'-initial-state.json')).read_text());contract=json.loads((source/'action-contract.json').read_text())
  previous=np.array(contract['q_PD_offset'])[0]+np.clip(np.array(state['previous_action'])[0]*5,-contract['clip'],contract['clip'])*np.array(contract['q_PD_scale'])[0]/5
  q=np.asarray([c['q_PD'] for c in ep['control']]);stand=np.asarray([c['nominal_q_PD'] for c in ep['control']]);times=np.asarray([c['tether_time'] for c in ep['control']])
  if 'correction_q_PD' in ep['control'][0]:
   corr=np.asarray([c['correction_q_PD'] for c in ep['control']]);goal=np.asarray([c['goal_q_PD'] for c in ep['control']])
  else:
   corr=[]
   release_start=next((d['time'] for d in ep['decisions'] if d['time']>7 and not d['active']),None)
   for i,c in enumerate(ep['control']):
    delta=np.array(ep['decisions'][i//5]['correction'])
    if release_start is not None and c['tether_time']>=release_start:
     d0=next(d for d in ep['decisions'] if d['time']==release_start)
     delta=np.array(d0['correction'])*max(0,1-(c['tether_time']-release_start)/.2)
    corr.append(delta)
   corr=np.asarray(corr);goal=q.copy();goal[:,22:]-=corr
  steps=np.diff(np.vstack([previous,q]),axis=0)
  result={}
  for phase,lo,hi in [('active',3,8),('ramp',3,7),('hold',7,8),('release',8,9)]:
   use=(times>=lo-1e-8)&(times<hi-1e-8)
   if not use.any():continue
   c=corr[use];g=(goal-stand)[use];total=(q-stand)[use]
   result[phase]=dict(correction_joint_RMS_deg=float(np.degrees(rms(c))),correction_max_joint_deg=float(np.degrees(abs(c).max())),correction_max_norm_deg=float(np.degrees(np.linalg.norm(c,axis=1).max())),
     correction_mean_per_joint_deg=np.degrees(c.mean(0)).tolist(),goal_arm_RMS_deg=float(np.degrees(rms(g[:,22:]))),goal_waist_RMS_deg=float(np.degrees(rms(g[:,12:15]))),goal_legs_RMS_deg=float(np.degrees(rms(g[:,:12]))),
     total_arm_RMS_deg=float(np.degrees(rms(total[:,22:]))),correction_to_goal_arm_RMS_ratio=float(rms(c)/max(rms(g[:,22:]),1e-12)),
     correction_to_total_arm_RMS_ratio=float(rms(c)/max(rms(total[:,22:]),1e-12)),actual_q_PD_step_max_deg=float(np.degrees(abs(steps[use]).max())),
     correction_at_point2_fraction=float(np.mean(np.max(abs(c),axis=1)>=.2-1e-8)))
  result['onset_q_PD_step_max_deg']=float(np.degrees(abs(steps[0]).max()))
  result['final_correction_max_rad']=float(abs(corr[-1]).max())
  if 'correction_q_PD' in ep['control'][0]:
   velocities=np.asarray([c['accumulation']['velocity'] for c in ep['control']]);applied_step=np.diff(np.vstack([np.zeros(7),corr]),axis=0)
   result['constraint_audit']=dict(max_abs_rad=float(abs(corr).max()),max_norm_rad=float(np.linalg.norm(corr,axis=1).max()),max_velocity_rad_s=float(abs(velocities).max()),max_norm_velocity_rad_s=float(np.linalg.norm(velocities,axis=1).max()),
     max_executed_correction_error_rad=float(abs((q-goal)[:,22:]-corr).max()),max_nonarm_direct_delta_rad=float(abs((q-goal)[:,:22]).max()),max_50Hz_increment_rad=float(abs(applied_step).max()),
     rejected_increment_fraction=float(np.mean([np.linalg.norm(d['proposal'].get('rejected',[0]*7))>1e-9 for d in ep['decisions'] if d['active']])),
     release_start=next((c['accumulation']['release_start'] for c in ep['control'] if c['accumulation']['release_start'] is not None),None))
   assert result['constraint_audit']['max_abs_rad']<=.2+1e-8
   assert result['constraint_audit']['max_norm_rad']<=.4+1e-8
   assert result['constraint_audit']['max_velocity_rad_s']<=.4+1e-8
   assert result['constraint_audit']['max_norm_velocity_rad_s']<=.8+1e-8
   assert result['constraint_audit']['max_executed_correction_error_rad']<1e-6
   assert result['constraint_audit']['max_nonarm_direct_delta_rad']==0
  diagnostics[label]=result
 assert episodes['accumulated']['same_initial_sha256']==episodes['standing_accumulated']['same_initial_sha256']
 assert all(c['executed_z']==episodes['standing_accumulated']['control'][0]['executed_z'] for c in episodes['standing_accumulated']['control'])
 return dict(results=results,command_attribution=diagnostics,same_initial=True,standing_z_constant=True,policy_updates=0,
             attribution_note='Same-state command decomposition, not causal motion fractions; C uses archived data, no baseline rerun.')
if __name__=='__main__':
 out=Path(sys.argv[1]);result=analyze(out,Path(__file__).resolve().parents[1]);(out/'causal-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
