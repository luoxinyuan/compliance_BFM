"""Read recorded CCL02A data only; no simulator/model/learning calls."""
import json,sys,importlib.metadata
from pathlib import Path
import numpy as np
root=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
old=root/'issue21-ccl02a-20261029T152000Z/attempt-04/outputs'
new=root/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs'
assert json.loads((new/'result.json').read_text())['status']=='completed'
summary=json.loads((new/'summary.json').read_text());results={}
for task in summary:
 b=json.loads((new/(task+'-baseline.json')).read_text());t=json.loads((new/(task+'-teacher.json')).read_text());o=json.loads((old/(task+'-teacher.json')).read_text())
 assert b['initial']==t['initial']
 assert b['decisions'][0]['bfm_obs']==t['decisions'][0]['bfm_obs']
 assert b['decisions'][0]['force_history']==t['decisions'][0]['force_history']
 # Reset history may retain -0.0; raw JSON hashes differ despite exact numeric equality.
 # The runner additionally enforces complete snapshot dictionary equality before each pair.
 cut=next(d['release_detected_at'] for d in t['decisions'] if d['release_detected_at'] is not None)
 n=sum(r['force_time']<cut for r in t['physics'])
 diffs={k:float(abs(np.array([r[k] for r in t['physics'][:n]])-np.array([r[k] for r in o['physics'][:n]])).max()) for k in ['qpos','qvel','force','wrist','root','torque']}
 assert max(diffs.values())==0
 q=np.array([c['q_PD'] for c in t['control']]);dq=np.diff(q,axis=0,prepend=q[:1]);res=np.array([c['teacher_delta_q'] for c in t['control']]);dr=np.diff(res,axis=0,prepend=res[:1])
 release=[i for i,c in enumerate(t['control']) if c['tether_time']>=4.5]
 afterzero=next(i for i in release if np.linalg.norm(res[i])<1e-9)
 assert np.max(abs(res[afterzero:]))==0
 pair=dict(initial_physics_and_actor_numeric_equal=True,raw_initial_json_hash_equal=b['same_initial_state_sha256']==t['same_initial_state_sha256'],release_detected_s=cut,teacher_zero_s=t['control'][afterzero]['time'],pre_release_diff=diffs,
  release_q_PD_single_joint_step_max_rad=float(abs(dq[release]).max()),release_teacher_single_joint_step_max_rad=float(abs(dr[release]).max()),
  teacher_action_residual_max_abs=float(abs(np.array([c['action'] for c in t['control']])-np.array([c['nominal'] for c in t['control']])).max()),
  pre_contact_teacher_max=float(abs(res[[i for i,c in enumerate(t['control']) if c['tether_time']<t['task']['onset_s']]]).max()),
  undesired_ground_contact_samples=sum(r['undesired_ground_contacts']>0 for r in t['physics']))
 pair['phase_common_baseline_error_axis']={}
 for phase,low,high in [('ramp',t['task']['onset_s'],t['task']['onset_s']+2),('hold',t['task']['onset_s']+2,4.5),('release',4.5,6.01)]:
  indices=[i for i,r in enumerate(b['physics']) if low<=r['force_time']<high];i=indices[0];before=b['initial'] if i==0 else b['physics'][i-1]
  e=np.array(b['physics'][i]['anchor'])-before['wrist'];norm=np.linalg.norm(e);u=None if norm<1e-9 else e/norm
  vals={}
  for label in ['baseline','teacher']:
   m=summary[task][label][phase];vals[label]=dict(wrist_mm=None if u is None else float(np.array(m['wrist_delta_xyz_mm'])@u),relative_mm=None if u is None else float(np.array(m['wrist_relative_root_delta_xyz_mm'])@u))
  pair['phase_common_baseline_error_axis'][phase]=dict(unit=None if u is None else u.tolist(),**vals)
 results[task]=pair
ledgers=[json.loads((root/r/'ledger.json').read_text()) for r in ['issue21-ccl02a-20261029T152000Z','issue21-ccl02a-20261029T153900Z']]
attempts=[a for l in ledgers for a in l['attempts']]
resource=dict(total_wall_seconds=sum(a['wall_seconds'] for a in attempts),peak_rss_gib=max(a['peak_rss_bytes'] for a in attempts)/1024**3,peak_gpu_mib=max(a['peak_gpu_mib'] for a in attempts),all_owned_groups_empty=all(not a['remaining_processes'] for a in attempts),all_shared_source_unchanged=all(a['shared_source_unchanged'] for a in attempts))
assert resource['total_wall_seconds']<1800
result=dict(task_audits=results,resources=resource,versions={name:importlib.metadata.version(name) for name in ['torch','numpy','mujoco']},python=sys.version)
(new/'paired-readback.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
