"""Descriptive first persistent divergence; thresholds are analysis markers, not success gates."""
import json,gzip
from pathlib import Path
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/contact-geometry-20261103/C-goal-10/job')
def first(t,v,threshold,count=20):
 for i in range(len(v)-count):
  if np.all(v[i:i+count]>threshold):return dict(time=float(t[i]),value=float(v[i]))
 return None
out={}
for name in ['straight','pair_left','pair_right','pair_translate']:
 a=json.load(gzip.open(R/('source_'+name+'.json.gz' if name in ['pair_left','pair_right'] else 'retain_'+name+'.json.gz'),'rt'));b=json.load(gzip.open(R/('direct_'+name+'.json.gz'),'rt'))
 aa=a['physics'];bb=b['physics'];assert len(aa)==len(bb);t=np.array([r['time'] for r in aa]);q1=np.array([r['qpos'][7:] for r in aa]);q2=np.array([r['qpos'][7:] for r in bb]);root1=np.array([r['root'] for r in aa]);root2=np.array([r['root'] for r in bb]);qdiff=np.degrees(np.max(abs(q1-q2),axis=1));rdiff=np.linalg.norm(root1-root2,axis=1)*100
 ca=a['control'];cb=b['control'];ct=np.array([r['time'] for r in ca]);pd=np.degrees(np.max(abs(np.array([r['q_PD'] for r in ca])-np.array([r['q_PD'] for r in cb])),axis=1))
 out[name]=dict(actual_joint_difference_over1deg_100ms=first(t,qdiff,1),root_difference_over1cm_100ms=first(t,rdiff,1),qPD_difference_over1deg_100ms=first(ct,pd,1,5),caveat='Two closed-loop trajectories: after first divergence, not same-state action causality; source-label alignment is reported separately.')
(R/'first-divergence.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
