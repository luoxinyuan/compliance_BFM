"""Scalar summaries of actual physics. Never scores released anchor geometry."""
import json,gzip,math,sys
from pathlib import Path
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m38b-rotation-20261103')
def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)
def wrap(v):return np.arctan2(np.sin(v),np.cos(v))
def yaw(r,key):m=np.array(r[key]);return np.arctan2(m[1,0],m[0,0])
def metrics(path):
 d=read(path);rs=d['physics'];cs=d['control'];init=d['initial'];s=d['condition'];phase={};t=np.array([r['force_time'] for r in rs]);active=np.array([r['active'] for r in rs]);torso=np.array([yaw(r,'torso_matrix') for r in rs]);pelvis=np.array([yaw(r,'pelvis_matrix') for r in rs])
 for name,(lo,hi) in d['phases'].items():
  ids=np.flatnonzero((t>=lo-1e-8)&(t<hi-1e-8));v=[rs[i] for i in ids]
  if not v:continue
  av=[r for r in v if r['active']];entry=dict(duration_s=len(v)*.005,root_delta_cm=(100*(np.array(v[-1]['root'])-v[0]['root'])).tolist(),torso_yaw_change_deg=float(np.rad2deg(wrap(yaw(v[-1],'torso_matrix')-yaw(v[0],'torso_matrix')))),pelvis_yaw_change_deg=float(np.rad2deg(wrap(yaw(v[-1],'pelvis_matrix')-yaw(v[0],'pelvis_matrix')))),root_speed_end=float(np.linalg.norm(v[-1]['qvel'][:3])),support_loss_fraction=float(np.mean([not any(r['support']) for r in v])),tilt_max_deg=max(r['tilt_deg'] for r in v))
  if av:
   err=np.array([[np.array(c['anchor'])-c['point'] for c in r['contacts']] for r in av]);force=np.array([[c['force'] for c in r['contacts']] for r in av]);entry.update(error_cm=float(100*np.sqrt(np.mean(np.sum(err**2,-1)))),signed_xyz_cm=(100*err.mean((0,1))).tolist(),per_point_force_xyz_mean=force.mean(0).tolist(),per_point_force_xyz_rms=np.sqrt(np.mean(force**2,0)).tolist(),sum_force_rms=float(np.sqrt(np.mean([r['sum_force_norm']**2 for r in av]))),cap_fraction=float(np.mean([any(np.linalg.norm(c['force'])>=c['cap']-.001 for c in r['contacts']) for r in av])),moment_mean_nm=float(np.mean([r['net_yaw_moment_com'] for r in av])),external_work_j=sum(r['external_power']*.005 for r in av))
  phase[name]=entry
 moving=np.flatnonzero(active&(t<s['onset']+s['ramp']));last=int(moving[-1]) if len(moving) else len(rs)-1
 q=np.array([r['qpos'][7:] for r in rs]);dq=np.array([r['qvel'][6:] for r in rs]);qd=np.array([c['q_PD'] for c in cs]);diff=np.abs(np.diff(qd,axis=0));mi=np.unravel_index(diff.argmax(),diff.shape)
 return dict(path=str(path),condition=s,phases=phase,boundary=d['boundary'],torso_turn_deg=float(np.rad2deg(wrap(torso[last]-yaw(init,'torso_matrix')))),pelvis_turn_deg=float(np.rad2deg(wrap(pelvis[last]-yaw(init,'pelvis_matrix')))),waist_relative_turn_deg=float(np.rad2deg(wrap((torso[last]-pelvis[last])-(yaw(init,'torso_matrix')-yaw(init,'pelvis_matrix'))))),root_total_delta_cm=(100*(np.array(rs[-1]['root'])-init['root'])).tolist(),torso_no_contact_drift_deg=float(np.rad2deg(wrap(torso[-1]-torso[0]))) if not active.any() else None,qpd_max_step_deg=float(np.rad2deg(diff.max())),qpd_p99_step_deg=float(np.rad2deg(np.percentile(diff,99))),qpd_peak_joint=int(mi[1]),qpd_peak_time=cs[mi[0]+1]['time'],q_actual_max_step_deg=float(np.rad2deg(np.abs(np.diff(q,axis=0)).max())),dq_max=float(np.abs(dq).max()),torque_saturation_max=max(r['torque_saturation'] for r in rs),support_loss_fraction=float(np.mean([not any(r['support']) for r in rs])),slip_p95_mps=float(np.percentile([r['contact_tangent_speed_max_mps'] for r in rs],95)),joint_limit_max_rad=max(r['joint_limit_excess_max_rad'] for r in rs))
def main():
 output=Path(sys.argv[1]);output.mkdir(exist_ok=True);runs=[];details={};models=[]
 for p in sorted(R.glob('*/job/summaries.json')):
  for row in read(p):
   tr=p.parent/(row['label']+'.json.gz');runs.append(dict(**row,trace=str(tr)));details[str(tr)]=metrics(tr)
 for p in R.glob('*/job/model.json'):models.append(read(p))
 led=read(R/'compute-ledger.json');total=dict(job_wall_seconds=sum(x.get('wall_seconds',x.get('elapsed_seconds',0)) for x in led['attempts']),expert_updates=sum(x.get('actual_updates',0) for x in led['attempts']),episodes=len(runs),output_bytes=sum(x.stat().st_size for x in R.rglob('*') if x.is_file()))
 for name,obj in [('results',runs),('physics',details),('models',models),('resources',led),('totals',total)]: (output/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False))
 print(json.dumps(total));print('\n'.join(f"{r['label']}: {r['error_cm']:.3f}cm {r['sum_force_norm_rms']:.3f}N torso{r['actual_torso_turn_deg']:.2f} pelvis{r['actual_pelvis_turn_deg']:.2f}" for r in runs))
if __name__=='__main__':main()
