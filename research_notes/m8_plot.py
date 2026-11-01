"""Render complete saved physical curves, never step BFM or simulation."""
import os,json,gzip
from pathlib import Path
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m8-contact-body-20261101A');(RUN/'analysis/mpl-cache').mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(RUN/'analysis/mpl-cache')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m8_analysis import read

def main():
 a=[a for a in read(RUN/'compute-ledger.json')['attempts'] if a['stage']=='final' and a.get('result_status')=='completed'][0];out=Path(a['path'])/'outputs'
 fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True);colors=dict(standing='gray',rule_bounded='orange',teacher='black',distill='tab:blue',alignment='tab:green')
 for ctl,color in colors.items():
  e=read(out/('down_test__'+ctl+'.json.gz'));r=e['physics'];cs=e['control'];t=np.array([v['time'] for v in r]);ct=np.array([v['time'] for v in cs]);s=e['condition'];active=np.array([v['active'] for v in r]);f=np.array([v['force'] for v in r]);p=np.array([v['wrist'] for v in r]);anchor=np.array([v['anchor'] for v in r]);q=np.array([v['qpos'] for v in r]);base=np.array(e['initial']['wrist']);err=np.linalg.norm(anchor-p,axis=1)*100;err[~active]=np.nan
  axes[0,0].plot(t,(p[:,2]-base[2])*100,color=color,label=ctl);axes[0,1].plot(t,err,color=color);axes[1,0].plot(t,np.linalg.norm(f,axis=1),color=color);axes[1,1].plot(t,q[:,2]*100,color=color);axes[2,0].plot(t,np.degrees(q[:,[10,16]].mean(1)),color=color);axes[2,1].plot(t,np.linalg.norm(np.array([v['qvel'][:3] for v in r]),axis=1),color=color)
  if ctl=='standing':
   z=(anchor[:,2]-base[2])*100;z[~active]=np.nan;axes[0,0].plot(t,z,'--',color='red',label='actual anchor Z')
 for ax,title in zip(axes.flat,['Upper-torso attachment Z displacement (cm)','Active 3D tracking error (cm)','Actual force magnitude (N)','Root height (cm)','Mean knee angle (deg)','Root speed (m/s), full release']):
  ax.set_title(title);ax.grid(alpha=.25);ax.axvline(s['onset'],color='k',ls=':',alpha=.4);ax.axvline(s['release'],color='k',ls=':',alpha=.4)
 axes[0,0].legend(fontsize=8);axes[2,0].set_xlabel('Actual simulation time (s)');axes[2,1].set_xlabel('Actual simulation time (s)');fig.suptitle('M8 sealed upper-torso downward maintained-contact guidance; frozen BFM/PD');fig.tight_layout();fig.savefig(RUN/'analysis/m8-body-response.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for ax,name in zip(axes.flat,['wrist_early','wrist_neg_early','wrist_test_pos_small','wrist_sealed_mid']):
  for ctl,c in [('m7','gray'),('distill','tab:blue'),('alignment','tab:green')]:
   e=read(out/(name+'__'+ctl+'.json.gz'));r=e['physics'];t=np.array([v['time'] for v in r]);p=np.array([v['wrist'] for v in r]);a=np.array([v['anchor'] for v in r]);mask=np.array([v['active'] for v in r]);err=np.linalg.norm(a-p,axis=1)*100;err[~mask]=np.nan;ax.plot(t,err,color=c,label=ctl)
  ax.set_title(name+' active 3D error (cm)');ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.25);ax.legend()
 fig.suptitle('M8 old-skill regressions; no re-selection from these results');fig.tight_layout();fig.savefig(RUN/'analysis/m8-wrist-retention.png',dpi=150)
if __name__=='__main__':main()
