import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105')
d=json.loads((R/'learning-curves.json').read_text());fig,axs=plt.subplots(2,3,figsize=(14,8))
for seed,c in d.items():
 t=c['training'];x=[v['rl_decisions'] for v in t]
 vals=[(0,0,[v['optimization']['epochs'][-1]['joint_gaussian_KL'] for v in t],'Analytic joint Gaussian KL'),(0,1,[v['optimization']['updated_value_EV'] for v in t],'Updated critic EV'),(0,2,[v['optimization']['raw_mean_change_rms'] for v in t],'Batch raw-mean change RMS')]
 for a,b,y,title in vals:axs[a,b].plot(x,y,label=seed);axs[a,b].set_title(title)
 dev=c['development'];dx=[v['iteration']*768 for v in dev]
 scores=[1.5 if v['score'] is None else v['score']['score'] for v in dev]
 axs[1,0].plot(dx,scores,'o-',label=seed);axs[1,0].set_title('Deterministic dev score (lower better)')
 for ax,key,title in [(axs[1,1],'local9','9cm dev tracking RMS (cm)'),(axs[1,2],'long65','65cm dev tracking RMS (cm)')]:
  ax.plot(dx,[next(r['error_cm'] for r in v['rows'] if r['label']==key+'__mean') for v in dev],'o-',label=seed);ax.set_title(title)
for ax in axs.flat:ax.set_xlabel('Real high-level RL decisions');ax.grid(alpha=.25);ax.legend()
fig.suptitle('Two independent seeds; task-mix training reward is not a matched physical evaluation');fig.tight_layout();fig.savefig(R/'learning-curves.png',dpi=160);plt.close(fig)
