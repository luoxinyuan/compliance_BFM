"""Offline trace readback only; no simulator or BFM calls."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
R=P/'bridge-v2-owned/runs/issue21-projector-20261030T004000Z/attempt-01';O=R/'outputs'
old=P/'bridge-v2-owned/runs/issue20-ccl01-20261028T082201Z/attempt-04/outputs'
get=lambda x:json.loads((O/x).read_text())
eps={m:get('dev-2-'+m+'.json') for m in ['baseline','teacher','student','student-no-tether']}
summary=get('summary.json')['dev-2'];result=get('result.json');assert result['all_complete'] and result['frozen_bfm_unchanged']
zero=json.loads((old/'anchor-reference.json').read_text())
keys=['qpos','qvel','force','wrist','root']
diff=lambda a,b:{k:float(abs(np.asarray([r[k] for r in a])-np.asarray([r[k] for r in b])).max()) for k in keys}
no_tether=diff(eps['student-no-tether']['physics'],zero['physics'])
assert max(no_tether.values())<1e-7
checks={}
for mode,e in eps.items():
 c=e['control'];p=e['physics'];n=sum(r['force_time']<1.1 for r in p)
 pre=diff(p[:n],eps['baseline']['physics'][:n]);assert max(pre.values())<1e-7
 masks={name:[i for i,r in enumerate(c) if low<=r['tether_time']<high] for name,low,high in [('pre',0,1.1),('ramp',1.1,3.1),('hold',3.1,4.5),('release',4.5,6.01)]}
 a=np.array([r['action'] for r in c]);nom=np.array([r['nominal'] for r in c]);q=np.array([r['q_PD'] for r in c]);qn=np.array([r['nominal_q_PD'] for r in c]);dq=np.diff(q,axis=0,prepend=q[:1]);delta=q-qn
 phases={}
 for phase,ix in masks.items():
  phases[phase]=dict(q_PD_step_max_joint_rad=float(abs(dq[ix]).max()),q_PD_residual_arm_rms_rad=float(np.sqrt(np.mean(delta[ix,22:]**2))),q_PD_residual_nonarm_rms_rad=float(np.sqrt(np.mean(delta[ix,:22]**2))),q_PD_residual_nonarm_max_rad=float(abs(delta[ix,:22]).max()),action_residual_rms=float(np.sqrt(np.mean((a-nom)[ix]**2))))
 checks[mode]=dict(pre_contact_diff=pre,phase_action=phases,undesired_ground_contact_frames=sum(r['undesired_ground_contacts']>0 for r in p),actual_seconds=len(p)*.005)
 if mode.startswith('student'):
  zz=np.asarray([r['executed_z'][0] for r in c]);g=np.array([r['gate'] for r in c]);stand=np.asarray(json.loads((P/'bridge-v2-owned/runs/issue19-wrist-anchor-20261028T062503Z/attempt-02/outputs/standing_latent.json').read_text())).reshape(256)
  checks[mode].update(z_norm_error_max=float(abs(np.linalg.norm(zz,axis=-1)-16).max()),z_angle_max_deg=float(np.rad2deg(np.arccos(np.clip((zz*stand).sum(-1)/256,-1,1))).max()),zero_gate_z_diff=float(abs(zz[g==0]-stand).max()),gate_on_times=[r['time'] for i,r in enumerate(c) if g[i]>0 and (i==0 or g[i-1]==0)],gate_off_times=[r['time'] for i,r in enumerate(c) if g[i]==0 and i>0 and g[i-1]>0])
  active=[d for d in e['decisions'] if 1.2<=d['time']<4.6]
  checks[mode]['normalized_observation_shift']={k:dict(mean=float(np.mean([d[k] for d in active])),max=max(d[k] for d in active)) for k in ['nearest_teacher_distance','same_time_teacher_distance']}
weights=O/'contact-projector-final.pt';sup=json.loads((R/'supervisor-result.json').read_text());manifest=json.loads((R/'manifest.json').read_text())
checks['provenance']=dict(runtime_commit=manifest['worktree_head'],script_sha256=manifest['launcher_sha256'],checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),checkpoint_bytes=weights.stat().st_size,training_updates=result['projector_updates'],owned_pid=sup['pid'],wall_seconds=sup['wall_seconds'],peak_rss_gib=sup['peak_rss_bytes']/1024**3,peak_gpu_mib=sup['peak_gpu_mib'],remaining_processes=sup['remaining_processes'],shared_source_unchanged=sup['shared_source_unchanged'])
checks['no_tether_vs_saved_unforced']=no_tether
checks['hold_error_gap_recovered']=(summary['baseline']['hold']['error_rms_mm']-summary['student']['hold']['error_rms_mm'])/(summary['baseline']['hold']['error_rms_mm']-summary['teacher']['hold']['error_rms_mm'])
(O/'offline-audit.json').write_text(json.dumps(checks,indent=2,allow_nan=False))
print(json.dumps(checks,indent=2))
