import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/direct-transfer-20261103');out=R/'analysis';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for arm,color in [('B1','tab:blue'),('B2','tab:orange')]:
 p=next(R.glob(arm+'-train-*'))/'job';its=[];err=[];force=[];worst=[];base={}
 for f in sorted(p.glob('dev-*/summary.json')):
  rows=[r for r in json.loads(f.read_text()) if r['label'].split('__')[0] in ['single','slower','faster']];it=int(f.parent.name.split('-')[1]);scores={r['label']:r['force_rms']+3*r['error_cm']/100+r['cap_fraction']+20*bool(r['boundary']) for r in rows}
  if not base:base=scores
  its.append(it);err.append(np.mean([r['error_cm'] for r in rows]));force.append(np.mean([r['force_rms'] for r in rows]));worst.append(max(scores[k]/base[k] for k in scores)+np.mean([scores[k]/base[k] for k in scores]))
 for ax,vals in zip(axes.flat,[err,force,worst]):ax.plot(its,vals,'o-',label=arm,color=color)
 rows=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()];axes.flat[3].plot([r['iteration'] for r in rows],[r['force_rms'] for r in rows],label=arm,color=color)
for ax,title,y in zip(axes.flat,['Deterministic: mean of 3 dev conditions','Deterministic: mean of 3 dev conditions','Selection: worst relative + average relative','Stochastic training batches (different states)'],['3D error RMS (cm)','Actual force RMS (N)','Score (initial = 2)','Actual force RMS (N)']):ax.set(title=title,xlabel='PPO batch (768 decisions)',ylabel=y);ax.grid(alpha=.25);ax.legend()
fig.suptitle('Narrow-domain Direct Teacher: frozen reward, network, exploration and physics')
fig.savefig(out/'teacher_learning.png',dpi=150)
