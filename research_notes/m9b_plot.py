"""Physical outcome plots from already-saved final trajectories."""
import json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m9b_stage import RUN,completed

def main():
 out=completed('final');colors={'expert':'#36a657','legacy':'#888888','student':'#dc6b27','ablation':'#326fc6'}
 fig,axes=plt.subplots(4,3,figsize=(15,11),sharex='col')
 for j,cond in enumerate(['held_straight','held_heading','held_right']):
  for ctl,color in colors.items():
   p=out/(cond+'__'+ctl+'.json.gz')
   if not p.exists():continue
   with gzip.open(p,'rt') as f:e=json.load(f)
   rs=e['physics'];t=np.array([r['force_time'] for r in rs]);active=np.array([r['active'] for r in rs]);error=np.array([np.linalg.norm(np.array(r['anchor'])-r['wrist'])*100 if r['active'] else np.nan for r in rs]);F=np.array([np.linalg.norm(r['force']) for r in rs]);R=np.array([r['pelvis_matrix'] for r in rs]);yaw=np.degrees(np.unwrap(np.arctan2(R[:,1,0],R[:,0,0])));speed=np.array([np.linalg.norm(r['qvel'][:2]) for r in rs]);cs=e['control']
   for i,x in enumerate([error,F,yaw-yaw[0],speed]):axes[i,j].plot(t,x,color=color,lw=1,label=ctl)
   if e['boundary']:axes[0,j].text(.02,.05,ctl+' PARTIAL',transform=axes[0,j].transAxes,color=color)
  axes[0,j].set_title(cond)
  for i,ylabel in enumerate(['Active 3D error (cm)','Actual force norm (N)','Heading from ready (deg)','Root XY speed (m/s)']):axes[i,j].set_ylabel(ylabel);axes[i,j].grid(alpha=.2)
  axes[-1,j].set_xlabel('Episode time (s)')
 axes[0,0].legend();fig.suptitle('M9 continuation: fixed terminal checkpoints; no evaluation-based selection');fig.tight_layout();fig.savefig(RUN/'terminal-physical.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
