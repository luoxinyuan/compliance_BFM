"""Read-only audit of all fixed physical branches, no model load or training."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import mujoco

def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def analyze(out,root):
 g=json.loads((out/'geometry.json').read_text());results={};episodes={}
 m=mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');q0=np.array(g['qpos'])
 for label in ['anchored','native']:
  ep=json.loads((out/(label+'.json')).read_text());episodes[label]=ep;rows=ep['physics'];control=ep['control'];result={}
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
   if result[phase] and results['anchored'][phase]:result[phase]['improvement_vs_anchored_pct']=100*(1-result[phase]['error_rms_cm']/results['anchored'][phase]['error_rms_cm'])
 diagnostics={}
 target=np.asarray(json.loads((out/'goal-path.json').read_text())['waypoints'][-1]['qpos'])
 for label,ep in episodes.items():
  hold=[r for r in ep['physics'] if r['active'] and 7<=r['force_time']<8]
  controls=[r for r in ep['control'] if 7<=r['tether_time']<8]
  if not hold or not controls:diagnostics[label]={'hold_not_reached':True};continue
  deltas={k:[] for k in ['replace_root','replace_waist','replace_arm','replace_shoulder_roll','replace_all']}
  for r in hold:
   actual=np.array(r['qpos'])
   for name,idx in [('replace_root',slice(0,7)),('replace_waist',slice(19,22)),('replace_arm',slice(29,36)),('replace_shoulder_roll',slice(30,31)),('replace_all',slice(0,36))]:
    d.qpos[:]=actual;d.qpos[idx]=target[idx];mujoco.mj_forward(m,d);deltas[name].append(d.xpos[body]-np.array(r['wrist']))
  q=np.asarray([r['qpos'][7:] for r in hold]);qp=np.asarray([r['q_PD'] for r in controls]);groups={'legs':slice(0,12),'waist':slice(12,15),'rightarm':slice(22,29)}
  state=json.loads((out/(label+'-initial-state.json')).read_text());contract=json.loads((out/'action-contract.json').read_text())
  prev=np.array(contract['q_PD_offset'])[0]+np.clip(np.array(state['previous_action'])[0]*5,-contract['clip'],contract['clip'])*np.array(contract['q_PD_scale'])[0]/5
  onset=np.array(ep['control'][0]['q_PD'])-prev
  native_only=[];corrections=[]
  for ci,c in enumerate(ep['control']):
   if 7<=c['tether_time']<8:
    correction=np.array(ep['decisions'][ci//5]['correction']);native_only.append(np.array(c['q_PD'])[22:]-correction);corrections.append(correction)
  diagnostics[label]={'actual_onset_step_max_deg':float(np.degrees(abs(onset).max())),'onset_max_joint':contract['dof_names'][int(np.argmax(abs(onset)))],
   'arm_native_BFM_q_PD_minus_goal_mean_deg':np.degrees(np.mean(np.array(native_only)-target[29:],axis=0)).tolist(),
   'arm_correction_mean_deg':np.degrees(np.mean(corrections,axis=0)).tolist(),
   'FK_substitution_mean_delta_wrist_cm':{k:(100*np.mean(v,axis=0)).tolist() for k,v in deltas.items()},
   'actual_joint_minus_IK_goal_mean_deg':np.degrees(np.mean(q-target[7:],axis=0)).tolist(),
   'q_PD_minus_IK_goal_mean_deg':np.degrees(np.mean(qp-target[7:],axis=0)).tolist(),
   'joint_RMS_error_deg':{k:float(np.degrees(rms(q[:,idx]-target[7:][idx]))) for k,idx in groups.items()},
   'arm_correction_bound_fraction':float(np.mean([max(abs(np.asarray(r['correction'])))>=.04-1e-8 or np.linalg.norm(r['correction'])>=.08-1e-8 for r in ep['decisions'] if 7<=r['time']<8]))}
 hashes={e['same_initial_sha256'] for e in episodes.values()};assert len(hashes)==1
 return dict(results=results,hold_diagnostics=diagnostics,diagnostic_note='FK substitution is kinematic sensitivity only; not a physical intervention or additive causal decomposition.',geometry=dict(delta_x_cm=100*g['delta_x_m'],chain_radius_upper_bound_cm=100*g['arm_radius_upper_bound_m'],fixed_shoulder_error_lower_bound_cm=100*g['sphere_lower_bound_error_m']),same_full_initial_state=True,kinematic_decomposition_note='Body contribution: current FK wrist minus FK wrist with original nonarm/root and current right-arm q. This is kinematic attribution, not a new physical counterfactual.',policy_updates=0)
if __name__=='__main__':
 out=Path(sys.argv[1]);result=analyze(out,Path(__file__).resolve().parents[1]);(out/'goal-calibration-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
