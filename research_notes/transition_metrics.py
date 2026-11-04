"""Same physical cost, plus source-aware phase report. No latent success criterion."""
import math,numpy as np
from m39a_metrics import cost
from m39a_metrics import extra as old_extra
from transition_env import phase
from geometry_env import wrap

def extra(task):
 result=old_extra(task);groups={}
 for r in task.physics:groups.setdefault(phase(task.spec,r['force_time']),[]).append(r)
 result['phase_metrics']={}
 for k,rr in groups.items():
  root=np.array([r['root'] for r in rr]);yaw=np.array([math.atan2(r['torso_matrix'][1][0],r['torso_matrix'][0][0]) for r in rr]);pelvis=np.array([math.atan2(r['pelvis_matrix'][1][0],r['pelvis_matrix'][0][0]) for r in rr])
  v=dict(seconds=len(rr)*.005,root_displacement_cm=(100*(root[-1]-root[0])).tolist(),root_mean_velocity_mps=((root[-1]-root[0])/max(.005,(len(rr)-1)*.005)).tolist(),torso_yaw_change_deg=float(np.rad2deg(wrap(yaw[-1]-yaw[0]))),pelvis_yaw_change_deg=float(np.rad2deg(wrap(pelvis[-1]-pelvis[0]))),waist_relative_final_deg=float(np.rad2deg(wrap(yaw[-1]-pelvis[-1]))),support_loss_ticks=sum(not any(r['support']) for r in rr),yaw_moment_rms=float(np.sqrt(np.mean([r['net_yaw_moment_com']**2 for r in rr]))),external_work_j=sum(r['external_power']*.005 for r in rr),max_actual_q_step_deg=float(np.rad2deg(np.abs(np.diff(np.array([r['qpos'][7:] for r in rr]),axis=0))).max()) if len(rr)>1 else 0.)
  active=[r for r in rr if r['active']]
  if active:
   error=np.array([[np.array(c['anchor'])-c['point'] for c in r['contacts']] for r in active]);force=np.array([[c['force'] for c in r['contacts']] for r in active])
   v.update(point_error_rms_cm=float(100*np.sqrt(np.mean(np.sum(error**2,-1)))),point_error_xyz_mean_cm=(100*error.mean(0)).tolist(),point_force_xyz_rms_n=np.sqrt((force**2).mean(0)).tolist(),sum_force_norm_rms_n=float(np.sqrt(np.mean([r['sum_force_norm']**2 for r in active]))),cap_fraction=float(np.mean([any(np.linalg.norm(c['force'])>=c['cap']-.001 for c in r['contacts']) for r in active])),yaw_error_rms_deg=float(np.rad2deg(np.sqrt(np.mean([wrap(r['target_yaw']-math.atan2(r['torso_matrix'][1][0],r['torso_matrix'][0][0]))**2 for r in active])))))
  result['phase_metrics'][k]=v
 result['handoff_sha256']=task.prefix_data['audit']['sha256']
 return result
