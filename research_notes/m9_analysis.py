"""Saved trace analysis only. No model or MuJoCo execution."""
import gzip,json,argparse
from pathlib import Path
import numpy as np

def summarize(ep):
 rows=ep['physics'];cs=ep['control'];out={'label':ep['label'],'complete':ep['boundary'] is None,'boundary':ep['boundary'],'physics_seconds':len(rows)/200,'phases':{}}
 if not rows:return out
 t=np.array([r['force_time'] for r in rows]);q=np.array([r['qpos'] for r in rows]);v=np.array([r['qvel'] for r in rows]);R=np.array([r['pelvis_matrix'] for r in rows]);yaw=np.unwrap(np.arctan2(R[:,1,0],R[:,0,0]));feet=np.array([r['feet_xyz'] for r in rows]);normal=np.array([r['foot_normal_force_n'] for r in rows]);loaded=normal>10
 body=np.einsum('nji,nj->ni',R,v[:,:3]);ct=np.array([c['time'] for c in cs]);cq=np.array([c['pre_qpos'][7:] for c in cs]);initial=ep['initial'];initialroot=np.asarray(initial['root'])
 for name,(lo,hi) in ep['phases'].items():
  mask=(t>=lo-1e-7)&(t<hi-1e-7);ids=np.flatnonzero(mask);cms=(ct>=lo-1e-7)&(ct<hi-1e-7)
  if not len(ids):continue
  rr=[rows[i] for i in ids];cc=[c for c,k in zip(cs,cms) if k];a,b=ids[0],ids[-1];support=loaded[mask];phasefeet=feet[mask];force=np.array([r['force'] for r in rr]);delta=q[b,:3]-q[a,:3]
  actualsteps=np.abs(np.diff(cq[cms],axis=0))
  out['phases'][name]=dict(duration=len(ids)/200,root_delta_xyz_m=delta.tolist(),root_ego_velocity_mean_mps=body[mask].mean(0).tolist(),root_world_velocity_mean_mps=v[mask,:3].mean(0).tolist(),root_speed_mean_mps=float(np.linalg.norm(v[mask,:2],axis=1).mean()),forward_std_mps=float(body[mask,0].std()),heading_change_deg=float(np.degrees(yaw[b]-yaw[a])),heading_span_deg=float(np.degrees(np.ptp(yaw[mask]))),path_heading_deg=float(np.degrees(np.arctan2(delta[1],delta[0]))),root_height_min_m=float(q[mask,2].min()),pelvis_tilt_max_deg=max(r['tilt_deg'] for r in rr),foot_loaded_fraction=support.mean(0).tolist(),single_support_fraction=float(np.mean(support.sum(1)==1)),flight_fraction=float(np.mean(support.sum(1)==0)),foot_unloading_counts=np.sum(np.diff(support.astype(int),axis=0)==-1,axis=0).tolist(),foot_xyz_range_m=np.ptp(phasefeet,axis=0).tolist(),contact_tangent_speed_p95_mps=float(np.percentile([r['contact_tangent_speed_max_mps'] for r in rr],95)),q_PD_step_peak_deg=float(np.degrees(max([c['q_PD_step_max_rad'] for c in cc],default=0))),q_PD_step_p95_deg=float(np.degrees(np.percentile([c['q_PD_step_max_rad'] for c in cc],95))) if cc else 0,actual_joint_20ms_peak_deg=float(np.degrees(actualsteps.max())) if actualsteps.size else 0,torque_saturation_fraction=float(np.mean([r['torque_saturation'] for r in rr])),force_rms_N=float(np.sqrt(np.mean(np.sum(force**2,axis=1)))),force_xyz_mean_N=force.mean(0).tolist(),force_cap_fraction=float(np.mean(np.linalg.norm(force,axis=1)>=4.999)),wrist_delta_xyz_m=(np.asarray(rr[-1]['wrist'])-rr[0]['wrist']).tolist(),shoulder_delta_xyz_m=(np.asarray(rr[-1]['shoulder_xyz'])-rr[0]['shoulder_xyz']).tolist())
  x=out['phases'][name];active=[r for r in rr if r['active']]
  if active:
   errors=np.array([np.asarray(r['anchor'])-r['wrist'] for r in active]);x.update(contact_tracking_rms_m=float(np.sqrt(np.mean(np.sum(errors**2,axis=1)))),contact_error_xyz_mean_m=errors.mean(0).tolist(),contact_error_xyz_rms_m=np.sqrt(np.mean(errors**2,axis=0)).tolist())
  else:x['contact_tracking_rms_m']=None
  x['wrist_relative_root_delta_m']=((np.asarray(rr[-1]['wrist'])-rr[-1]['root'])-(np.asarray(rr[0]['wrist'])-rr[0]['root'])).tolist()
  x['root_speed_peak_mps']=float(np.linalg.norm(v[mask,:3],axis=1).max())
  x['root_path_length_xy_m']=float(np.linalg.norm(np.diff(q[mask,:2],axis=0),axis=1).sum())
  x['root_speed_last_second_mean_mps']=float(np.linalg.norm(v[ids[-200:],:3],axis=1).mean())
  x['heading_from_ready_end_deg']=float(np.degrees(yaw[b]-yaw[0]))
  x['force_xyz_rms_N']=np.sqrt(np.mean(force**2,axis=0)).tolist()
  x['force_peak_N']=float(np.linalg.norm(force,axis=1).max())
  x['joint_range_deg']=np.degrees(np.ptp(q[mask,7:],axis=0)).tolist()
  if cc and 'alpha' in cc[0] and not ep['label'].endswith(('__standing','__student','__m8')):x.update(alpha_range=[min(c['alpha'] for c in cc),max(c['alpha'] for c in cc)],phase_range=[min(c['phase'] for c in cc),max(c['phase'] for c in cc)],reference_exhausted_fraction=float(np.mean([c['reference_exhausted'] for c in cc])))
 out.update(final_root_displacement_xyz_m=(q[-1,:3]-initialroot).tolist(),final_root_speed_mps=float(np.linalg.norm(v[-1,:3])),final_heading_change_deg=float(np.degrees(yaw[-1]-yaw[0])),all_joint_limit_excess_rad=max(r['joint_limit_excess_max_rad'] for r in rows))
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);a=p.parse_args();results={}
 for f in sorted(a.outputs.glob('*.json.gz')):
  with gzip.open(f,'rt') as h:e=json.load(h)
  if 'physics' in e and 'control' in e and e.get('phases'):results[e['label']]=summarize(e)
 (a.outputs/'capability-metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False));print(json.dumps(results,indent=2))
