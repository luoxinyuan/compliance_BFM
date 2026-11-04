"""Saved physical phase analysis; no simulation, optimization or test reselection."""
import json,gzip,sys,math
from pathlib import Path
import numpy as np
from m39b_env import phase
from m38b_analyze import metrics,read,yaw,wrap

def summarize(path):
 ep=read(path);s=ep['condition'];rr=ep['physics'];initial=ep['initial'];groups={}
 for r in rr:groups.setdefault(phase(s,r['force_time']),[]).append(r)
 out={'trace':str(path),'condition':s,'physical_quality':metrics(path),'phases':{}}
 for name,rows in groups.items():
  root=np.array([r['root'] for r in rows]);ty=np.array([np.rad2deg(wrap(yaw(r,'torso_matrix')-yaw(initial,'torso_matrix'))) for r in rows]);py=np.array([np.rad2deg(wrap(yaw(r,'pelvis_matrix')-yaw(initial,'pelvis_matrix'))) for r in rows]);dy=np.array([np.rad2deg(wrap(r['target_yaw']-yaw(initial,'torso_matrix'))) for r in rows])
  v=dict(seconds=.005*len(rows),root_delta_cm=(100*(root[-1]-root[0])).tolist(),torso_deg_start_end=[float(ty[0]),float(ty[-1])],pelvis_deg_start_end=[float(py[0]),float(py[-1])],waist_deg_start_end=[float(ty[0]-py[0]),float(ty[-1]-py[-1])],target_deg_start_end=[float(dy[0]),float(dy[-1])],root_velocity_last=rows[-1]['qvel'][:3])
  active=[r for r in rows if any(c['active'] for c in r['contacts'])]
  if active:
   err=np.array([[np.array(c['anchor'])-c['point'] for c in r['contacts']] for r in active]);force=np.array([[c['force'] for c in r['contacts']] for r in active]);caps=np.array([[c['cap'] for c in r['contacts']] for r in active]);n=np.linalg.norm(force,axis=-1)
   v.update(point_error_rms_cm=float(100*np.sqrt(np.mean(np.sum(err**2,axis=-1)))),point_signed_xyz_cm=(100*err.mean(0)).tolist(),point_force_xyz_rms=np.sqrt((force**2).mean(0)).tolist(),point_force_xyz_mean=force.mean(0).tolist(),sum_force_norm_rms=float(np.sqrt((n.sum(-1)**2).mean())),point_cap_fraction=(n>=caps-.001).mean(0).tolist(),any_cap_seconds=float(.005*np.any(n>=caps-.001,axis=-1).sum()),signed_x_error_integral_cm_s=float(100*.005*err[:,:,0].mean(-1).sum()),positive_lag_integral_cm_s=float(100*.005*np.maximum(err[:,:,0].mean(-1),0).sum()),negative_overshoot_integral_cm_s=float(100*.005*np.maximum(-err[:,:,0].mean(-1),0).sum()),yaw_error_rms_deg=float(np.sqrt(np.mean([np.rad2deg(wrap(r['target_yaw']-yaw(r,'torso_matrix')))**2 for r in active]))),yaw_moment_rms=float(np.sqrt(np.mean([r['net_yaw_moment_com']**2 for r in active]))))
  # No old-anchor error in released recovery or no-contact metrics.
  out['phases'][name]=v
 active=[r for r in rr if any(c['active'] for c in r['contacts'])]
 if active and s['distance']>0:
  # Descriptive threshold, not a pass criterion: first sustained 2cm root advance.
  x0=active[0]['root'][0];moving=[i for i in range(len(active)-20) if all(r['root'][0]-x0>.02 for r in active[i:i+20])]
  out['root_2cm_sustained_start_delay_s']=float(active[moving[0]]['force_time']-s['onset']) if moving else None
 out['quarter_second_curve']=[]
 for t in np.arange(rr[0]['force_time'],s['end']+.001,.25):
  r=min(rr,key=lambda r:abs(r['force_time']-t))
  out['quarter_second_curve'].append(dict(time=r['force_time'],phase=phase(s,r['force_time']),torso_deg=float(np.rad2deg(wrap(yaw(r,'torso_matrix')-yaw(initial,'torso_matrix')))),pelvis_deg=float(np.rad2deg(wrap(yaw(r,'pelvis_matrix')-yaw(initial,'pelvis_matrix')))),target_deg=float(np.rad2deg(wrap(r['target_yaw']-yaw(initial,'torso_matrix')))),root_xyz=r['root'],point_xyz=[c['point'] for c in r['contacts']],point_force_xyz=[c['force'] for c in r['contacts']],yaw_moment=r['net_yaw_moment_com']))
 return out
if __name__=='__main__':
 result=[summarize(Path(p)) for p in sys.argv[2:]];Path(sys.argv[1]).write_text(json.dumps(result,indent=2,allow_nan=False))
