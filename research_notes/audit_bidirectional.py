"""Independent recorded-data audit. No model, fitting, rollout, or GPU."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUNS=P/'bridge-v2-owned/runs'
R=Path(sys.argv[1]);O=R/'outputs';read=lambda p:json.loads(p.read_text());arr=lambda rows,k:np.asarray([r[k] for r in rows])
result=read(O/'result.json');assert result['status']=='completed' and result['projector_updates']==2000 and result['ppo_updates']==0
summary=read(O/'summary.json');old=RUNS/'issue21-projector-20261030T004000Z/attempt-01/outputs';teacher=RUNS/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs'
standing=np.asarray(read(RUNS/'issue19-wrist-anchor-20261028T062503Z/attempt-02/outputs/standing_latent.json')).reshape(256)
episodes={}
for task,modes in summary.items():
 episodes[task]={}
 for mode in modes:
  path=O/f'{task}-{mode}.json'
  if not path.exists():path=(old if task=='dev-2' else teacher)/f'{task}-{mode}.json'
  episodes[task][mode]=read(path)
checks={};gains={};force_ablation={};windows={};direction_detail={}
for task,modes in episodes.items():
 b=modes['baseline'];s=modes['student'];t=modes['teacher'];onset=b['task']['onset_s'];checks[task]={};gains[task]={}
 for ph in ['ramp','hold']:
  e=[summary[task][m][ph]['error_rms_mm'] for m in ['baseline','teacher','student']]
  gains[task][ph]=dict(student_gain_fraction=1-e[2]/e[0],teacher_gap_recovered=(e[0]-e[2])/(e[0]-e[1]))
 direction_detail[task]={}
 for mode,ep in modes.items():
  assert ep['initial']==b['initial']
  assert ep['decisions'][0]['bfm_obs']==b['decisions'][0]['bfm_obs']
  assert np.array_equal(arr(ep['physics'],'anchor'),arr(b['physics'],'anchor'))
  assert len(ep['physics'])==1200 and len(ep['control'])==300 and len(ep['decisions'])==60
  c=ep['control'];rows=ep['physics'];q=arr(c,'q_PD');delta=q-arr(c,'nominal_q_PD');step=np.diff(q,axis=0,prepend=q[:1]);mask=arr(c,'tether_time')>=4.5
  first=lambda seq:next(iter(seq),None)
  check=dict(failed=ep['failed'],initial_state_equal=True,anchor_trajectory_equal=True,
    first_force_s=first(r['force_time'] for r in rows if np.linalg.norm(r['force'])>1e-6),
    first_action_residual_s=first(r['time'] for i,r in enumerate(c) if np.linalg.norm(delta[i])>1e-6),
    first_wrist_departure_s=first(r['time'] for i,r in enumerate(rows) if np.linalg.norm(np.asarray(r['wrist'])-b['physics'][i]['wrist'])>1e-6),
    release_joint_step_max_rad=float(abs(step[mask]).max()),release_delta_zero_s=first(r['time'] for i,r in enumerate(c) if r['time']>=4.6 and np.linalg.norm(delta[i])==0),
    unwanted_ground_frames=sum(r['undesired_ground_contacts']>0 for r in rows),
    postcontact=dict(root_min_m=min(r['height'] for r in rows if r['force_time']>=onset),tilt_max_deg=max(r['tilt_deg'] for r in rows if r['force_time']>=onset),
      both_feet_loaded_fraction=float(np.mean([all(v>1e-6 for v in r['foot_normal_force_n']) for r in rows if r['force_time']>=onset]))))
  if mode.startswith('student'):
   g=arr(c,'gate')
   z=arr(c,'executed_z').reshape(300,256)
   check.update(z_norm_error_max=float(abs(np.linalg.norm(z,axis=-1)-16).max()),angle_max_deg=float(np.rad2deg(np.arccos(np.clip((z*standing).sum(-1)/256,-1,1))).max()),
     zero_gate_standing_exact=bool(np.array_equal(z[g==0],np.broadcast_to(standing,z[g==0].shape))))
   d=[d for d in ep['decisions'] if onset+.1-1e-8<=d['time']<4.5]
   check['nearest_teacher_distance_mean']=float(np.mean([x['nearest_teacher_distance'] for x in d]))
  direction_detail[task][mode]={}
  for ph,lo,hi in [('ramp',onset,onset+2),('hold',onset+2,4.5)]:
   selected=[r for r in rows if lo<=r['force_time']<hi];vec=arr(selected,'anchor')-arr(selected,'wrist')
   direction_detail[task][mode][ph]=dict(error_xyz_rms_mm=(1000*np.sqrt(np.mean(vec**2,axis=0))).tolist(),error_mean_xyz_mm=(1000*vec.mean(0)).tolist())
  checks[task][mode]=check
 # Fixed relative windows describe persistence; no sample selection.
 edges=[onset,onset+.5,onset+1,onset+1.5,onset+2,4.5]
 windows[task]=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  metrics={}
  for m in ['baseline','teacher','student']:
   rows=[r for r in modes[m]['physics'] if lo<=r['force_time']<hi]
   metrics[m]=float(np.sqrt(np.mean(arr(rows,'error')**2))*1000)
  windows[task].append(dict(start=lo,end=hi,error_rms_mm=metrics))
 if 'student-force-zero' in modes:
  force_ablation[task]={ph:dict(normal_error_mm=summary[task]['student'][ph]['error_rms_mm'],zero_error_mm=summary[task]['student-force-zero'][ph]['error_rms_mm'],
   normal_force_n=summary[task]['student'][ph]['force_rms_n'],zero_force_n=summary[task]['student-force-zero'][ph]['force_rms_n']) for ph in ['ramp','hold']}
# Per-direction same absolute hold windows not pooled across different onset schedules.
old_plus=read(old/'summary.json')['dev-2']['student']
comparison=dict(old_plus_hold_mm=old_plus['hold']['error_rms_mm'],new_plus_hold_mm=summary['dev-2']['student']['hold']['error_rms_mm'],
 old_student_minus_hold_mm=summary['dev-3']['student-old']['hold']['error_rms_mm'],new_minus_hold_mm=summary['dev-3']['student']['hold']['error_rms_mm'])
sup=read(R/'supervisor-result.json');manifest=read(R/'manifest.json')
analysis=dict(direction_detail=direction_detail,gains=gains,checks=checks,force_ablation=force_ablation,windows=windows,old_student_comparison=comparison,
 runtime_commit=manifest['worktree_head'],script_sha256=manifest['launcher_sha256'],checkpoint_sha256=hashlib.sha256((O/'contact-projector-final.pt').read_bytes()).hexdigest(),
 resources={k:sup[k] for k in ['pid','pgid','wall_seconds','peak_rss_bytes','peak_gpu_mib','remaining_processes','shared_source_unchanged','resource_boundary']})
(O/'bidirectional-analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
print(json.dumps(dict(gains=gains,force_ablation=force_ablation,old_student_comparison=comparison,resources=analysis['resources']),indent=2))
