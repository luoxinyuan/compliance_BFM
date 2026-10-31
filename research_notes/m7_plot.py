"""Inspect full physical transition and release; no model invocation."""
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m7_analysis import RUN,read

def main():
 final=next(a for a in read(RUN/'compute-ledger.json')['attempts'] if a['stage']=='final' and a.get('result_status')=='completed');src=Path(final['path'])/'outputs'
 fig,axes=plt.subplots(5,2,figsize=(15,17),sharex='col')
 labels=['native_hold','force_rule','teacher','m6_alignment','m6_distill','alignment','distill'];colors=['.6','olive','black','cornflowerblue','violet','darkorange','teal']
 for col,name in enumerate(['sealed_mid','sealed_late']):
  for ctl,color in zip(labels,colors):
   e=read(src/(name+'__'+ctl+'.json.gz'));s=e['condition'];tr=s['transition'];rr=e['physics'];cs=e['control'];t=np.array([r['force_time'] for r in rr])-s['onset'];ct=np.array([c['time'] for c in cs])-s['onset'];active=np.array([r['active'] for r in rr]);err=np.array([r['error'] for r in rr])*100;err[~active]=np.nan
   f=np.linalg.norm(np.array([r['force'] for r in rr]),axis=1);w=np.array([r['wrist'] for r in rr])-np.array([r['root'] for r in rr]);i=np.argmin(abs(t));w=(w-w[i])*100
   alpha=[c['native_coords'][0] if c['native_coords'] is not None else c['alpha'] if ctl=='teacher' else c['rule_record']['alpha'] if ctl=='force_rule' else 0 for c in cs]
   for row,y,tt in [(0,err,t),(1,f,t),(2,w[:,0],t),(3,alpha,ct),(4,np.linalg.norm(np.array([r['qvel'][:3] for r in rr]),axis=1),t)]:axes[row,col].plot(tt,y,color=color,label=ctl,lw=1.3)
  for ax in axes[:,col]:
   ax.axvspan(tr['unload']-s['onset'],tr['restart']-s['onset'],alpha=.12,color='grey')
   for v in [0,tr['restart']-s['onset'],tr['reverse_end']-s['onset'],s['release']-s['onset']]:ax.axvline(v,ls=':',c='.5')
   ax.grid(alpha=.2)
  axes[0,col].set_title(name);axes[0,col].legend(fontsize=8,ncol=2);axes[-1,col].set_xlabel('Seconds from initial contact')
 for i,l in enumerate(['3D contact error (cm)','Actual force norm (N)','Wrist-root X increment (cm)','Executed native progress','Root speed (m/s)']):axes[i,0].set_ylabel(l)
 fig.suptitle('M7: same-episode physical reversal, fixed final checkpoints, full 8s recovery');fig.tight_layout();fig.savefig(RUN/'analysis/m7-transitions.png',dpi=140)
if __name__=='__main__':main()
