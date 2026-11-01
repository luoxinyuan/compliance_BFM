"""Read-only raw-trajectory analysis; only writes the continuation output index."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np
from m9_analysis import summarize
from m10_stage import RUN

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
    relative=np.array([np.array(r['wrist'])-r['root'] for r in active]);row['point_relative_root_range_xyz_m']=np.ptp(relative,axis=0).tolist()
    row['point_relative_velocity_rms_mps']=float(np.sqrt(np.mean(np.sum((np.diff(relative,axis=0)/.005)**2,axis=1))))
    row['recovery_speed_peak_mps']=rec.get('root_speed_peak_mps');row['recovery_heading_change_deg']=rec.get('heading_change_deg');row['recovery_qpd_peak_deg']=rec.get('q_PD_step_peak_deg')
   if cs and 'oracle_q_PD' in cs[0]:
    move_cs=[c for c in cs if e['condition']['onset']<=c['time']<e['condition']['release']]
    if move_cs:
     dq=np.array([np.array(c['q_PD'])-c['oracle_q_PD'] for c in move_cs]);z=np.array([c['executed_z'][0] for c in move_cs]);oz=np.array([c['oracle_z'][0] for c in move_cs]);angle=np.degrees(np.arccos(np.clip(np.sum(z*oz,axis=1)/256,-1,1)))
     row.update(visited_oracle_qpd_rms_deg=float(np.degrees(np.sqrt(np.mean(dq**2)))),visited_oracle_leg_waist_rms_deg=float(np.degrees(np.sqrt(np.mean(dq[:,:15]**2)))),visited_oracle_arm_rms_deg=float(np.degrees(np.sqrt(np.mean(dq[:,22:]**2)))),visited_oracle_z_angle_mean_deg=float(angle.mean()))
   if e['condition'].get('family')=='walking':
    rr=[r for r in rs if r['active'] and np.linalg.norm(r['anchor_velocity'][:2])>.02 and np.linalg.norm(r['qvel'][:2])>.05]
    if rr:
     R=np.array([r['pelvis_matrix'] for r in rr]);yaw=np.arctan2(R[:,1,0],R[:,0,0]);v=np.array([r['qvel'][:2] for r in rr]);av=np.array([r['anchor_velocity'][:2] for r in rr]);desired=np.arctan2(av[:,1],av[:,0]);actual=np.arctan2(v[:,1],v[:,0]);wrap=lambda a:np.arctan2(np.sin(a),np.cos(a))
     row.update(moving_heading_error_rms_deg=float(np.degrees(np.sqrt(np.mean(wrap(yaw-desired)**2)))),moving_direction_error_rms_deg=float(np.degrees(np.sqrt(np.mean(wrap(actual-desired)**2)))),moving_speed_mean=float(np.linalg.norm(v,axis=1).mean()),same_times_partner_speed_mean=float(np.linalg.norm(av,axis=1).mean()),moving_evaluation_seconds=len(rr)/200)
   rows.append(row)
 result=dict(episodes=rows,compute_seconds=sum(a.get('wall_seconds',0) for a in ledger['attempts']),peak_rss=max(a.get('peak_rss_bytes',0) for a in ledger['attempts']),peak_vram=max(a.get('peak_gpu_mib',0) for a in ledger['attempts']),attempts=ledger['attempts'])
 (RUN/'metrics.json').write_text(json.dumps(result,indent=2));(RUN/'phase-metrics.json').write_text(json.dumps(full,indent=2))
 for r in rows:print(r['stage'],r['label'],r['complete'],'err',round(r.get('error_cm',0),2),'force',round(r.get('force',0),3),'heading',round(r['heading_move'],2),'qpd',round(r['qpd_peak'],2))
if __name__=='__main__':main()
