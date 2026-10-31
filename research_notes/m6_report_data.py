"""Compact factual report inputs, derived only from immutable experiment outputs."""
import json
from pathlib import Path
from m6_analysis import RUN,read

def main():
 s=read(RUN/'analysis/physical-summary.json');m=read(RUN/'analysis/mechanism.json');ledger=read(RUN/'compute-ledger.json');out={'final':{},'training':{},'resource':{},'real_physics':{}}
 for k,e in s['episodes'].items():
  if not k.startswith('final/'):continue
  p=e['phases'].get('all_contact',{});w=e['phases']['whole'];x={q:p.get(q) for q in ['tracking_rms_cm','tracking_xyz_rms_cm','force_rms_n','force_peak_n','force_mean_xyz_n','force_cap_fraction','wrist_end_xyz_cm','shoulder_end_xyz_cm','root_end_xyz_cm','torso_rotation_max_deg','waist_change_max_abs_deg','same_obs_additional_qPD_rms_deg']}
  x.update(hold_rms_cm=e['phases'].get('hold',{}).get('tracking_rms_cm'),full_release=e['recovery'],release_speed_2_4_6_8=e['recovery_speed_2_4_6_8'],late_release=e['late_recovery'],release_progress=e['release_progress'],qPD_peak=e['qPD_peak'],actual20ms_joint_step=e['qpos_20ms_step_max_deg'],source=e['source'],free_difference=e.get('relative_to_native_free'),mechanism=m[k])
  x['safety']={q:w[q] for q in ['root_min_height_cm','pelvis_tilt_max_deg','both_feet_load_fraction','samples_without_either_foot_load','actual_joint_limit_excess_max_deg','torque_saturation_mean_fraction','failed_samples','undesired_contacts_max','action_clip_fraction']};out['final'][k.split('/',1)[1]]=x
 for a in ledger['attempts']:
  src=Path(a['path'])/'outputs';r=read(src/'result.json') if (src/'result.json').exists() else {}
  if a['stage'].startswith('train') and a.get('result_status')=='completed':
   t=read(src/'training.json');p=read(src/'new-data-projection.json');out['training'][a['stage']]=dict(samples=r['study']['samples'],new_samples=p['samples'],max_group_projection_arm_deg=max(g['arm_rms_deg'] for g in p['groups']),max_group_projection_nonarm_deg=max(g['nonarm_rms_deg'] for g in p['groups']),methods={k:{q:v[q] for q in ['before','after','updates','checkpoint','sha256']} for k,v in t.items()})
  pr=read(src/'physical-result.json') if (src/'physical-result.json').exists() else {}
  if 'episodes' in pr:out['real_physics'][Path(a['path']).name]=dict(episodes=len(pr['episodes']),complete=sum(e['complete'] for e in pr['episodes']),seconds=sum(e['seconds']+3 for e in pr['episodes']),boundaries=[e['boundary'] for e in pr['episodes'] if e['boundary']])
 out['resource']=dict(wall=sum(a.get('wall_seconds',0) for a in ledger['attempts']),peak_rss_gib=max(a.get('peak_rss_bytes',0) for a in ledger['attempts'])/1024**3,peak_gpu_mib=max(a.get('peak_gpu_mib',0) for a in ledger['attempts']),attempts=len(ledger['attempts']),noncompleted=[a['stage'] for a in ledger['attempts'] if a.get('result_status')!='completed'])
 out['precontact_parity']=s['precontact_pair_parity'];(RUN/'analysis/report-data.json').write_text(json.dumps(out,indent=2,allow_nan=False))
 for k,x in out['final'].items():print(k,x['tracking_rms_cm'],x['force_rms_n'],x['force_cap_fraction'])
 print('resources',out['resource']);print('physical',out['real_physics'])
if __name__=='__main__':main()
