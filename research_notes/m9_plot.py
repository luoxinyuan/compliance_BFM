"""Plot complete saved physical trajectories, including failures and recovery."""
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m9-contact-locomotion-20261101A')
def main():
 out=R/'final-attempt-01/outputs'
 for name in ['new_straight','new_right','unsupported_left','new_heading']:
  fig,ax=plt.subplots(3,2,figsize=(13,11));fig.subplots_adjust(left=.08,right=.98,bottom=.07,top=.92,hspace=.42,wspace=.25)
  for ctl in ['standing','feedback','expert','m8','student']:
   with gzip.open(out/(name+'__'+ctl+'.json.gz'),'rt') as f:e=json.load(f)
   p=e['physics'];c=e['control'];t=np.array([r['force_time'] for r in p]);q=np.array([r['qpos'] for r in p]);w=np.array([r['wrist'] for r in p]);a=np.array([r['anchor'] for r in p]);force=np.array([r['force'] for r in p]);active=np.array([r['active'] for r in p]);rot=np.array([r['pelvis_matrix'] for r in p]);yaw=np.unwrap(np.arctan2(rot[:,1,0],rot[:,0,0]));normal=np.array([r['foot_normal_force_n'] for r in p]);ct=np.array([r['time'] for r in c]);label=ctl+(' PARTIAL' if e['boundary'] else '');color={'standing':'C0','feedback':'C1','expert':'C2','m8':'C3','student':'C4'}[ctl]
   ax[0,0].plot(q[:,0]-q[0,0],q[:,1]-q[0,1],label=label,color=color)
   if ctl=='standing':ax[0,0].plot(a[:,0]-a[0,0],a[:,1]-a[0,1],'k--',label='partner displacement')
   err=np.linalg.norm(a-w,axis=1);err[~active]=np.nan
   ax[0,1].plot(t,err,label=label,color=color);ax[1,0].plot(t,np.linalg.norm(force,axis=1),label=label,color=color);ax[1,1].plot(t,np.degrees(yaw-yaw[0]),label=label,color=color)
   ax[2,0].plot(ct,np.degrees([r['q_PD_step_max_rad'] for r in c]),label=label,color=color)
   if ctl in ['expert','student']:
    frac=(normal>10).sum(1)==1;bins=np.arange(0,len(t),40)
    ax[2,1].plot(t[bins],[frac[i:i+40].mean() for i in bins],label=label,color=color)
  for aa,title in zip(ax.flat,['Root XY / partner displacement (m)','Active 3D tracking error (m)','Actual force norm (N)','Heading relative to ready (deg)','q_PD 20 ms step maximum (deg)','Single support fraction / 0.2 s']):aa.set_title(title,fontsize=11);aa.grid(alpha=.25)
  ax[0,0].set_aspect('equal',adjustable='datalim');ax[0,0].set_xlabel('world X (m)');ax[0,0].set_ylabel('world Y (m)');ax[0,0].legend(fontsize=8)
  for aa in [ax[0,1],*ax[1],*ax[2]]:aa.set_xlabel('physics time (s)')
  ax[2,1].legend();fig.suptitle('M9 sealed condition: '+name+' | full phases, same initial state')
  fig.savefig(R/(name+'-physical.png'),dpi=140,bbox_inches='tight',pad_inches=.15);plt.close(fig)
if __name__=='__main__':main()
