"""Plot saved physical evidence; no model/simulator invocation."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);dest=Path(sys.argv[2])
fig,axs=plt.subplots(3,3,figsize=(15,10),sharex=True)
for ctl,color in [('standing','#606060'),('A2','#d88c12'),('forward_clock','#b83c3c'),('raise_clock','#256cbb')]:
 p=out/('up30__'+ctl+'.json')
 if not p.exists():continue
 e=json.loads(p.read_text());r=e['physics'];c=e['control'];t=np.array([a['time'] for a in r]);tc=[a['time'] for a in c]
 wrist=np.array([a['wrist'] for a in r]);anchor=np.array([a['anchor'] for a in r]);force=np.array([a['force'] for a in r]);root=np.array([a['root'] for a in r]);initial=e['initial']
 for j in range(3):
  axs[0,j].plot(t,100*(wrist[:,j]-initial['wrist'][j]),label=ctl,color=color,lw=1.3)
  if ctl=='standing':axs[0,j].plot(t,100*(anchor[:,j]-initial['wrist'][j]),'k--',lw=1,label='partner geometry')
  axs[1,j].plot(t,force[:,j],color=color,lw=1.3)
 axs[2,0].plot(tc,[a['alpha'] for a in c],color=color,lw=1.3)
 axs[2,1].plot(t,100*np.linalg.norm(root-np.array(initial['root']),axis=1),color=color,lw=1.3)
 axs[2,2].plot(tc,np.degrees([a['q_PD_step_max_rad'] for a in c]),color=color,lw=1.3)
for j,axis in enumerate('XYZ'):
 axs[0,j].set_title(f'Wrist {axis} displacement (cm)');axs[1,j].set_title(f'Actual force {axis} (N)')
axs[2,0].set_title('Executed alpha (clock arms privileged)');axs[2,1].set_title('Root displacement magnitude (cm)');axs[2,2].set_title('Maximum q_PD step /20ms (degrees)')
for ax in axs.ravel():
 ax.axvline(3.8,c='k',ls=':',lw=.6);ax.axvline(7.8,c='k',ls=':',lw=.6);ax.axvline(8.8,c='k',ls=':',lw=.6);ax.grid(alpha=.2)
for ax in axs[2]:ax.set_xlabel('Simulation time (s)')
axs[0,0].legend(fontsize=8);fig.suptitle('Frozen BFM: upward wrist qualification, one fixed seed / no training\n3.8s onset | 4s ramp | 1s hold | 8.8s release | original 5N cap')
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(dest,dpi=160);plt.close(fig)
