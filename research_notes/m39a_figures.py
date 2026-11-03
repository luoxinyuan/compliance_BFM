import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m39a-demand-20261104');out=Path(sys.argv[1]);out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
base={r['label']:r for r in read(R/'A-prepare-02/job/summaries.json')};coverage={r['label'].replace('coverage_',''):r for r in read(R/'A-train-05/job/summaries.json')}
fig,axs=plt.subplots(4,2,figsize=(11,12))
for i,key in enumerate(['zero','small_left','straight','sequence']):
 for j,metric in enumerate(['error_cm','sum_force_norm_rms']):
  ax=axs[i,j]
  for d,label,color in [('B-ppo-03','PPO seed3901','#2277bb'),('C-ppo-04','PPO seed3902','#d06623')]:
   vals=[]
   for p in sorted((R/d/'job').glob('dev-*/summary.json')):
    row=next(r for r in read(p) if r['label']==key+'__mean');vals.append((int(p.parent.name.split('-')[-1])*768,row[metric]))
   ax.plot(*zip(*vals),'-o',ms=3,color=color,label=label)
  for k,label,color,ls in [('D2_','Frozen D2','#555555',':'),('T30_','Frozen T30','#888888','--')]:ax.axhline(base[k+key][metric],color=color,ls=ls,label=label)
  ax.axhline(coverage[key][metric],color='#449944',ls='--',label='Coverage supervision')
  ax.set(title=key,ylabel='3D error (cm)' if j==0 else 'RMS sum point loads (N)',xlabel='Actual RL decisions');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=7);fig.suptitle('M39-A deterministic development; training aggregate return is not physical success');fig.tight_layout();fig.savefig(out/'development-curves.png',dpi=140);plt.close(fig)
