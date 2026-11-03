"""Publication-small figures from fixed checkpoints and recorded actual physics."""
import json,gzip,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m38c-history-20261103');out=Path(sys.argv[1]);out.mkdir(exist_ok=True)
fig,ax=plt.subplots(1,3,figsize=(13,3.5))
for p in sorted(R.glob('*-train-*/job/model.json')):
 m=json.loads(p.read_text());v=json.loads((p.parent/'learning-curve.json').read_text())
 for a,key in zip(ax,['qpd_loss','z_loss','gradient']):a.plot([r['step'] for r in v],[r[key] for r in v],label=m['arm']);a.set_yscale('log');a.set_title(key);a.set_xlabel('supervised update');a.grid(alpha=.2)
ax[0].legend();fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(12,7));colors={'C2':'gray','D1':'tab:orange','D2':'tab:blue'}
for col,side in enumerate(['left','right']):
 for model,c in colors.items():
  for no in [False,True]:
   name=model+'_'+side+('_noforce_new' if no else '_new');p=next(R.glob('*-final-*/job/'+name+'.json.gz'));e=json.load(gzip.open(p,'rt'));rr=e['physics'][::4];time=np.array([r['force_time'] for r in rr]);yaw=lambda m:np.arctan2(m[1][0],m[0][0]);y=np.unwrap([yaw(r['torso_matrix']) for r in rr])-yaw(e['initial']['torso_matrix'])
   ax[0,col].plot(time,np.degrees(y),'--' if no else '-',color=c,label=model+(' no force' if no else ' contact'))
   if not no:ax[1,col].plot(time,[r['sum_force_norm'] for r in rr],color=c,label=model)
 for a in ax[:,col]:a.axvline(4.45,c='black',lw=.6);a.axvline(14.5,c='black',lw=.6);a.grid(alpha=.2);a.set_xlabel('physical time (s)')
 ax[0,col].set_title(side+(' new: +12 deg' if side=='left' else ' new: -12 deg'));ax[0,col].set_ylabel('actual torso change (deg)');ax[1,col].set_ylabel('sum point force norms (N)');ax[0,col].legend(fontsize=8,ncol=2)
fig.tight_layout();fig.savefig(out/'sealed-contact-curves.png',dpi=150)
