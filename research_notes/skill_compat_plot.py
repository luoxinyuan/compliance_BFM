"""Plot completed qualification records, no simulation or model execution."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);dest=Path(sys.argv[2])
fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True)
colors={'standing':'#666666','local':'#0072B2','native':'#D55E00','transition':'#009E73'}
for col,task in enumerate(['small','large']):
 for ctl in ['standing','local','native']:
  ep=json.loads((out/(task+'__'+ctl+'.json')).read_text());r=ep['physics'];c=ep['control'];t=np.array([x['force_time'] for x in r]);active=np.array([x['active'] for x in r]);a=np.array([x['anchor'] for x in r]);w=np.array([x['wrist'] for x in r]);initial=ep['initial']
  err=100*np.linalg.norm(a-w,axis=1);err[~active]=np.nan
  axes[0,col].plot(t,err,label=ctl,color=colors[ctl])
  axes[1,col].plot(t,np.linalg.norm([x['force'] for x in r],axis=1),color=colors[ctl])
  shoulder=100*(np.array([x['shoulder_xyz'] for x in r])[:,0]-initial['shoulder_xyz'][0]);axes[2,col].plot(t,shoulder,color=colors[ctl])
  axes[3,col].plot([x['time'] for x in c],np.degrees([x['q_PD_step_max_rad'] for x in c]),color=colors[ctl])
 for row in range(4):
  axes[row,col].axvline(3.8,color='k',ls=':',lw=.8);axes[row,col].axvline(7.8,color='k',ls=':',lw=.8);axes[row,col].axvline(8.8,color='k',ls='--',lw=.8);axes[row,col].grid(alpha=.2)
 axes[0,col].set_title('+X 3.5 cm' if col==0 else '+X 18.6819 cm');axes[0,col].legend();axes[3,col].set_xlabel('Simulation time (s)')
for ax,label in zip(axes[:,0],['Contact 3D error (cm)','Actual force (N)','Shoulder forward shift (cm)','Max q_PD step (degrees / 20ms)']):ax.set_ylabel(label)
fig.suptitle('Frozen S1 local vs A2 native-goal: paired qualification, seed 0\n4s partner ramp / K80 D8 / 5N cap; release tracking error excluded')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(dest,dpi=160);plt.close(fig)
