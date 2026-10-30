"""Standalone saved-trace plots; no inference or simulation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-alpha-cross-20261031A')
P=json.loads(Path(__file__).with_name('alpha_cross_protocol.json').read_text())
conditions=[c for c in P['conditions'] if c['split']=='sealed_final']
fig,axes=plt.subplots(3,4,figsize=(16,10),sharex='col',layout='constrained')
colors={'standing':'#888888','contact':'#111111','A':'#2271b2','B':'#d55e00'}
for col,c in enumerate(conditions):
 for ctl in ['standing','contact','A','B']:
  e=json.loads((R/'final-attempt-01/outputs'/(c['name']+'__'+ctl+'.json')).read_text());cs=e['control'];ph=e['physics']
  t=np.array([v['time'] for v in cs]);pt=np.array([v['force_time'] for v in ph]);mask=np.array([v['active'] for v in ph]);error=np.array([v['error'] for v in ph])*100;error[~mask]=np.nan
  axes[0,col].plot(t,[v['alpha'] for v in cs],color=colors[ctl],label={'A':'A2 direct alpha','B':'B2 alignment','contact':'Teacher','standing':'Standing'}[ctl],lw=1.3)
  axes[1,col].plot(pt,np.linalg.norm([v['force'] for v in ph],axis=1),color=colors[ctl],lw=1.1)
  axes[2,col].plot(pt,error,color=colors[ctl],lw=1.1)
 for row in range(3):
  axes[row,col].axvspan(c['disconnect_start'],c['disconnect_end'],color='#777777',alpha=.15)
  axes[row,col].axvline(10.4,color='#555555',ls=':',lw=.8);axes[row,col].grid(alpha=.2);axes[row,col].set_xlim(5.3,12.4)
 axes[0,col].set_title(f"Gap start {c['disconnect_start']:.1f}s / duration {c['disconnect_end']-c['disconnect_start']:.2f}s")
 axes[2,col].set_xlabel('Simulation time (s)')
axes[0,0].set_ylabel('Skill progress alpha');axes[1,0].set_ylabel('Actual force norm (N)');axes[2,0].set_ylabel('Active-contact 3D error (cm)')
axes[0,0].legend(fontsize=8,loc='upper left')
fig.suptitle('Fixed final checkpoints: unseen interruption combinations\nShaded = real detach; dotted = final release. Geometry error omitted when disconnected.',fontsize=13)
fig.savefig(R/'analysis/sealed-cross-physics.png',dpi=160)
plt.close(fig)
