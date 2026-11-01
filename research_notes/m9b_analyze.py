"""Read-only raw-trajectory analysis; only writes the continuation output index."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np
from m9_analysis import summarize
from m9b_stage import RUN

def main():
 ledger=json.loads((RUN/'compute-ledger.json').read_text());rows=[];full={}
 for a in ledger['attempts']:
  for p in sorted((Path(a['path'])/'outputs').glob('*.json.gz')):
   with gzip.open(p,'rt') as f:e=json.load(f)
   if not e.get('control'):continue
   m=summarize(e);full[a['stage']+'/'+e['label']]=m;rs=e['physics'];active=[r for r in rs if r['active']];cs=e['control'];ph=m['phases'];move=ph.get('move',{});rec=ph.get('recover',{})
   row=dict(stage=a['stage'],label=e['label'],path=str(p),complete=m['complete'],boundary=m['boundary'],root=m['final_root_displacement_xyz_m'],heading_move=move.get('heading_from_ready_end_deg'),heading_end=m['final_heading_change_deg'],single_support=move.get('single_support_fraction'),slip_p95=move.get('contact_tangent_speed_p95_mps'),qpd_peak=max(x['q_PD_step_peak_deg'] for x in ph.values()),qpd_p95=max(x['q_PD_step_p95_deg'] for x in ph.values()),joint20ms_peak=max(x['actual_joint_20ms_peak_deg'] for x in ph.values()),tilt_max=max(x['pelvis_tilt_max_deg'] for x in ph.values()),height_min=min(x['root_height_min_m'] for x in ph.values()),joint_limit=m['all_joint_limit_excess_rad'],torque=max(x['torque_saturation_fraction'] for x in ph.values()),action_clip=max(c.get('action_saturation_fraction',0) for c in cs),recover_speed=rec.get('root_speed_last_second_mean_mps'),move_ego_v=move.get('root_ego_velocity_mean_mps'),condition=e['condition'])
   if active:
    F=np.array([r['force'] for r in active]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);row.update(error_cm=float(np.sqrt(np.mean(np.sum(E**2,1)))*100),error_xyz_cm=(np.sqrt(np.mean(E**2,0))*100).tolist(),force=float(np.sqrt(np.mean(np.sum(F**2,1)))),force_xyz=np.sqrt(np.mean(F**2,0)).tolist(),cap=float(np.mean(np.linalg.norm(F,axis=1)>=4.999)))
   rows.append(row)
 result=dict(episodes=rows,compute_seconds=sum(a.get('wall_seconds',0) for a in ledger['attempts']),peak_rss=max(a.get('peak_rss_bytes',0) for a in ledger['attempts']),peak_vram=max(a.get('peak_gpu_mib',0) for a in ledger['attempts']),attempts=ledger['attempts'])
 (RUN/'metrics.json').write_text(json.dumps(result,indent=2));(RUN/'phase-metrics.json').write_text(json.dumps(full,indent=2))
 for r in rows:print(r['stage'],r['label'],r['complete'],'err',round(r.get('error_cm',0),2),'force',round(r.get('force',0),3),'heading',round(r['heading_move'],2),'qpd',round(r['qpd_peak'],2))
if __name__=='__main__':main()
