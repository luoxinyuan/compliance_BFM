"""Saved-trace-only M9 index and comparable contact/physical metrics."""
import json,gzip,hashlib
from pathlib import Path
import numpy as np
from m9_analysis import summarize
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m9-contact-locomotion-20261101A')
def main():
 ledger=json.loads((RUN/'compute-ledger.json').read_text());allrows=[];attempts=[];training=[]
 for a in ledger['attempts']:
  out=Path(a['path'])/'outputs';metrics={};short=[]
  for f in sorted(out.glob('*.json.gz')):
   with gzip.open(f,'rt') as h:e=json.load(h)
   if not e.get('physics') or not e.get('control') or not e.get('phases'):continue
   m=summarize(e);metrics[e['label']]=m;rr=e['physics'];active=[r for r in rr if r['active']];cs=e['control'];p=m['phases'];move=p.get('move',p.get('execute',{}));recover=p.get('recover',{})
   row=dict(stage=a['stage'],attempt=Path(a['path']).name,label=e['label'],complete=m['complete'],boundary=m['boundary'],condition=e['condition'],path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),physics_seconds=len(rr)/200,root_delta=m['final_root_displacement_xyz_m'],root_speed_final=m['final_root_speed_mps'],heading_total_deg=m['final_heading_change_deg'],move_heading_from_ready_deg=move.get('heading_from_ready_end_deg'),move_root_delta=move.get('root_delta_xyz_m'),move_single_support=move.get('single_support_fraction'),move_foot_unloads=move.get('foot_unloading_counts'),move_slip_p95=move.get('contact_tangent_speed_p95_mps'),qpd_peak_deg=max(x['q_PD_step_peak_deg'] for x in p.values()),qpd_actual_20ms_peak_deg=max(x['actual_joint_20ms_peak_deg'] for x in p.values()),root_height_min=min(x['root_height_min_m'] for x in p.values()),tilt_max=max(x['pelvis_tilt_max_deg'] for x in p.values()),joint_limit_excess=m['all_joint_limit_excess_rad'],torque_saturation_max=max(x['torque_saturation_fraction'] for x in p.values()),action_saturation_max=max(c.get('action_saturation_fraction',0) for c in cs),release_speed_peak=recover.get('root_speed_peak_mps'),release_last_second_speed=recover.get('root_speed_last_second_mean_mps'))
   if active:
    f3=np.array([r['force'] for r in active]);err=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);row.update(contact_seconds=len(active)/200,error_rms_m=float(np.sqrt(np.mean(np.sum(err**2,1)))),error_xyz_rms_m=np.sqrt(np.mean(err**2,0)).tolist(),force_rms_N=float(np.sqrt(np.mean(np.sum(f3**2,1)))),force_xyz_mean_N=f3.mean(0).tolist(),force_xyz_rms_N=np.sqrt(np.mean(f3**2,0)).tolist(),force_cap_fraction=float(np.mean(np.linalg.norm(f3,axis=1)>=4.999)))
   short.append(row);allrows.append(row)
  if metrics:(out/'capability-metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
  attempts.append(dict(stage=a['stage'],attempt=Path(a['path']).name,commit=a['commit'],status=a.get('result_status'),exit_code=a.get('exit_code'),wall_seconds=a.get('wall_seconds'),resource_boundary=a.get('resource_boundary'),episodes=len(short),complete=sum(x['complete'] for x in short)))
  if (out/'training.json').exists():
   t=json.loads((out/'training.json').read_text());training.append(dict(stage=a['stage'],checkpoint=t['checkpoint'],sha256=t['sha256'],updates=t['student_optimizer_updates'],manifest=t['manifest'],fit=t['training_metrics'],identity=t['native_projection_identity']))
 result=dict(attempts=attempts,episodes=allrows,training=training,total_physics_seconds=sum(r['physics_seconds'] for r in allrows),complete=sum(r['complete'] for r in allrows),partial=sum(not r['complete'] for r in allrows),total_compute_seconds=sum(a.get('wall_seconds',0) for a in ledger['attempts']),peak_rss_bytes=max(a.get('peak_rss_bytes',0) for a in ledger['attempts']),peak_gpu_mib=max(a.get('peak_gpu_mib',0) for a in ledger['attempts']))
 (RUN/'m9-results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','training']},indent=2))
if __name__=='__main__':main()
