"""Plots from saved physics only; no simulation/model execution."""
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs');R=P/'m3-continuous-body-20261031A'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 fig,ax=plt.subplots(3,3,figsize=(15,10),sharex=True)
 colors={'native_hold':'#777777','teacher':'#8053b8','local_teacher':'#0082b3'}
 for i,u in enumerate([.25,.5,.75]):
  for ctl in colors:
   p=R/'qualification-attempt-01/outputs'/f'u{round(u*1000):03d}__{ctl}.json.gz'
   if not p.exists():continue
   e=read(p);rows=e['physics'];t=np.array([r['force_time'] for r in rows]);mask=(t>=12.4-1e-8)&(t<17.4-1e-8);err=np.array([r['error'] for r in rows])*100;force=np.linalg.norm([r['force'] for r in rows],axis=1);speed=np.linalg.norm(np.array([r['qvel'][:3] for r in rows]),axis=1)
   for j,y in enumerate([np.where(mask,err,np.nan),force,speed]):ax[i,j].plot(t,y,label=ctl,color=colors[ctl],linewidth=1.3)
  for j in range(3):
   a=ax[i,j];a.axvline(12.4,color='k',ls=':',lw=.8);a.axvline(17.4,color='k',ls='--',lw=.8);a.set_xlim(10.4,25.4);a.grid(alpha=.15);a.set_title(f'u={u}: '+['active 3D error (cm)','real force norm (N)','root speed (m/s)'][j])
  ax[i,1].axhline(5,color='k',ls=':',lw=.8)
 for a in ax[-1]:a.set_xlabel('Physical time (s), contact12.4 / release17.4')
 ax[0,0].legend(fontsize=8);fig.suptitle('M3 fixed teacher candidates; original BFM/PD/limits, no tuning');fig.tight_layout();fig.savefig(R/'analysis/teacher-qualification.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(12,7),sharex=True)
 for i,ctl in enumerate(['distill','alignment']):
  for label,p,color in [('normal Student',P/'m2-unseen-body-20261031A/comparison-attempt-01/outputs'/f'midpoint__{ctl}.json.gz','#0072b2'),('no-contact z replay + real contact',R/'ablation-attempt-01/outputs'/f'u500__{ctl}_z_replay.json.gz','#d55e00')]:
   e=read(p);rs=e['physics'];t=np.array([r['force_time'] for r in rs]);mask=(t>=12.4-1e-8)&(t<17.4-1e-8)
   for j,y in enumerate([np.where(mask,np.array([r['error'] for r in rs])*100,np.nan),np.linalg.norm([r['force'] for r in rs],axis=1)]):ax[i,j].plot(t,y,label=label,color=color);ax[i,j].set_title(ctl+': '+['3D error (cm)','actual force (N)'][j]);ax[i,j].grid(alpha=.15);ax[i,j].set_xlim(12.4,23.4)
  ax[i,0].legend(fontsize=8)
 for a in ax[-1]:a.set_xlabel('Physical time (s); release17.4')
 fig.suptitle('Physical command ablation: online BFM proprioceptive feedback and PD retained');fig.tight_layout();fig.savefig(R/'analysis/contact-command-ablation.png',dpi=160)
if __name__=='__main__':main()
