"""Recover exact takeover-boundary labels from saved decisions/controls, no rerun."""
import json,gzip,hashlib,time
from pathlib import Path
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/prior-recovery-20261105');d=json.loads((R/'recovery-decision.json').read_text());out=R/'corrected-labels';out.mkdir(exist_ok=False);checks=[];paths=[]
for branch in d['branches']:
 ep=json.load(gzip.open(R/'physical-03'/(branch['label']+'.json.gz'),'rt'));old=json.load(gzip.open(branch['path'],'rt'));rows=[];t=branch['switch'];allc=ep['control']
 for offset in range(10):
  tt=t+offset*.1;ix=min(range(len(ep['decisions'])),key=lambda i:abs(ep['decisions'][i]['time']-(tt+.1)));decision=ep['decisions'][ix];assert abs(decision['time']-(tt+.1))<1e-7
  cc=allc[ix*5:(ix+1)*5];assert len(cc)==5 and abs(cc[0]['time']-tt)<1e-7
  raw=np.asarray(decision['raw']);z=16*raw/np.linalg.norm(raw);prev=np.asarray(decision['obs'][565:821])*16
  row=dict(x=decision['obs'],raw=raw.tolist(),z=z.tolist(),previous_z=prev.tolist(),time=cc[0]['time'],episode=branch['label'],kind='walk',sequence=cc,valid_steps=5)
  if offset:
   prior=min(old,key=lambda r:abs(r['time']-tt));assert abs(prior['time']-tt)<1e-7
   delta={k:float(np.max(abs(np.asarray(row[k])-prior[k]))) for k in ['x','raw','z','previous_z']};assert max(delta.values())<3e-6,delta
  rows.append(row)
 # First label input matches actual baseline B state at identical decision boundary.
 b=json.load(gzip.open(R/'physical-03/B__walk.json.gz','rt'));q=min(b['decisions'],key=lambda r:abs(r['time']-(t+.1)));assert np.max(abs(np.asarray(q['obs'])-rows[0]['x']))==0
 p=out/(branch['label']+'-labels.json.gz')
 with gzip.open(p,'wt') as f:json.dump(rows,f)
 paths.append(str(p));checks.append(dict(branch=branch['label'],time=t,old_count=len(old),corrected_count=len(rows),exact_B_input=True,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
d.update(accepted_label_files=paths,correction='Strict float switch<=t omitted boundary by ~1e-13. Restored from exact aligned recorded decision obs/raw and five measured controls. No extra physics.',checks=checks,matched_continuation=dict(updates_per_arm=500,prior_updates_per_arm=1500,total_stage_updates=4000,optimizer='fresh Adam same in both, old checkpoint did not save optimizer',retention_coefficient=2.473907620604606,selection='fixed final2000; no final test used',reason='engineering boundary-label correction, not test-driven refitting'))
(R/'recovery-decision-corrected.json').write_text(json.dumps(d,indent=2));print(json.dumps(checks))
