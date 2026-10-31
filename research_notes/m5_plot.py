"""Plots saved final physical traces only, no model or simulator."""
import json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m5-contact-lpp-20261031A')
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 a=next(a for a in read(R/'compute-ledger.json')['attempts'] if a['stage']=='final' and a.get('result_status')=='completed');src=Path(a['path'])/'outputs';colors={'native_hold':'#888888','rule':'#208343','teacher':'#151515','alignment':'#1874d1','distill':'#cc641c'};fig,axes=plt.subplots(4,2,figsize=(13,11),sharex='col')
 for j,condition in enumerate(['test_timing_size','test_longer_larger']):
  for ctl,col in colors.items():
   e=read(src/(condition+'__'+ctl+'.json.gz'));p=e['physics'];c=e['control'];t=np.array([r['force_time'] for r in p]);active=np.array([r['active'] for r in p]);err=np.linalg.norm(np.array([r['anchor'] for r in p])-np.array([r['wrist'] for r in p]),axis=1)*100;err[~active]=np.nan;force=np.linalg.norm(np.array([r['force'] for r in p]),axis=1);w=(np.array([r['wrist'] for r in p])-e['initial']['wrist'])*100;root=np.linalg.norm(np.array([r['qvel'][:3] for r in p]),axis=1);alpha=[r['native_coords'][0] if ctl in ['alignment','distill'] else 0 if ctl=='native_hold' else r['alpha'] for r in c]
   axes[0,j].plot(t,err,color=col,label=ctl,lw=1.3);axes[1,j].plot(t,force,color=col,lw=1.2);axes[2,j].plot([r['time'] for r in c],alpha,color=col,lw=1.2);axes[3,j].plot(t,root,color=col,lw=1.2)
  for ax in axes[:,j]:
   ax.axvline(e['condition']['onset'],color='#777',ls=':',lw=.8);ax.axvline(e['condition']['release'],color='#777',ls=':',lw=.8);ax.grid(alpha=.2)
  axes[0,j].set_title(condition);axes[3,j].set_xlabel('Physical time (s)')
 for i,label in enumerate(['Contact 3D error (cm)','Actual force norm (N)','Executed learned / rule progress','Root speed (m/s)']):axes[i,0].set_ylabel(label)
 axes[0,0].legend(fontsize=8,ncol=3);fig.suptitle('M5 fixed final checkpoints: unseen contact combinations, full release',fontsize=13);fig.tight_layout();fig.savefig(R/'analysis/m5-final-curves.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
