"""Post-run saved-data diagnostics; no simulation or optimization."""
import json,gzip,time,hashlib,sys
from pathlib import Path
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105');start=time.monotonic();ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_notes'));from prior_metrics import summarize
rows=[]
for name in ['F4501','R4501','F4502','R4502','probe','final-F','final-R','final-reference']:
 for p in sorted((R/name).glob('*.json.gz')):
  if any(k in p.name for k in ['labels','prefix','queries']):continue
  m=summarize(p)
  if m:rows.append(m)
(R/'physical-metrics.json').write_text(json.dumps(rows,indent=2));comparisons=[]
for folder in ['final-F','final-R','final-reference','probe']:
 ss=json.loads((R/folder/'summaries.json').read_text());models=sorted(set(r['label'].split('__')[0] for r in ss if r['label'].endswith('__stop')))
 for name in models:
  a=json.load(gzip.open(R/folder/(name+'__stop.json.gz'),'rt'));b=json.load(gzip.open(R/folder/(name+'__continue.json.gz'),'rt'));branch=7.05
  dd=[]
  for u,v in zip(a['decisions'],b['decisions']):
   if abs(u['time']-v['time'])>1e-6:break
   dd.append(dict(time=u['time']-.1,obs_max=float(abs(np.array(u['obs'])-v['obs']).max()),raw_rms=float(np.mean((np.array(u['raw'])-v['raw'])**2)**.5)))
  first=next((d for d in dd if d['raw_rms']>1e-6),None);duration=[]
  for ep in [a,b]:
   rr=[r for r in ep['physics'] if branch<=r['force_time']<ep['condition']['release']];root=np.array([r['root'] for r in rr]);w=np.array([r['wrist'] for r in rr]);duration.append(dict(root_forward_cm=float(100*(root[-1,0]-root[0,0])),wrist_forward_cm=float(100*(w[-1,0]-w[0,0])),force_rms=float(np.sqrt(np.mean([sum(x*x for x in r['force']) for r in rr]))),error_cm=float(100*np.sqrt(np.mean([r['error']**2 for r in rr])))))
  comparisons.append(dict(model=name,folder=folder,first_command_difference=first,prefix_command_max=max(d['raw_rms'] for d in dd if d['time']<branch-1e-7),after_branch_stop_continue=duration,scope='matched deterministic real physical branches; differences do not identify force vs privileged geometry contribution'))
(R/'branch-response.json').write_text(json.dumps(comparisons,indent=2))
source=[]
for folder in ['final-F','final-R','final-reference']:
 for p in (R/folder).glob('*__moving65.json.gz'):
  ep=json.load(gzip.open(p,'rt'));r=min(ep['physics'],key=lambda r:abs(r['force_time']-6.5));source.append(dict(label=ep['label'],time=r['force_time'],root_velocity_xyz=r['qvel'][:3],root=r['root'],feet=r['feet_xyz'],wrist=r['wrist'],scope='actual I prefix, classify measured motion not intended source name'))
(R/'source-states.json').write_text(json.dumps(source,indent=2));cost=dict(seconds=time.monotonic()-start,physics_steps=0,updates=0);(R/'analysis-cost.json').write_text(json.dumps(cost));print(len(rows),cost)
