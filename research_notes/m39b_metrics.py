"""Stage-wise actual contact metrics; no released-anchor task scoring."""
import math,numpy as np
from m39a_metrics import cost
from m39a_metrics import extra as old_extra
from m39b_env import phase
from geometry_env import wrap

def extra(task):
 result=old_extra(task);groups={}
 for r in task.physics:groups.setdefault(phase(task.spec,r['force_time']),[]).append(r)
 result['phase_metrics']={}
 for k,rr in groups.items():
  root=np.array([r['root'] for r in rr]);yaw=np.array([math.atan2(r['torso_matrix'][1][0],r['torso_matrix'][0][0]) for r in rr])
  v=dict(seconds=len(rr)*.005,root_displacement_cm=(100*(root[-1]-root[0])).tolist(),torso_yaw_change_deg=float(np.rad2deg(wrap(yaw[-1]-yaw[0]))),support_loss_ticks=sum(not any(r['support']) for r in rr))
  # Detailed per-point fields are analyzed from saved raw physics; preserve active-only old summary.
  if k not in ('released_recovery','pre_contact','no_actual_contact'):
   active=[r for r in rr if any(c['active'] for c in r['contacts'])]
   if active:
    v['point_error_rms_cm']=float(100*np.sqrt(np.mean([np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in r['contacts']]) for r in active])))
    v['sum_force_norm_rms_n']=float(np.sqrt(np.mean([r['sum_force_norm']**2 for r in active])))
    v['yaw_error_rms_deg']=float(np.rad2deg(np.sqrt(np.mean([wrap(r['target_yaw']-math.atan2(r['torso_matrix'][1][0],r['torso_matrix'][0][0]))**2 for r in active]))))
  result['phase_metrics'][k]=v
 return result
