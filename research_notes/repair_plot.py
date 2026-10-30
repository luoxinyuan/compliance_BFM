"""Export compact sealed-test plots from existing physical records only."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(run):
 p=run/'physics-attempt-01/outputs'
 fig,axes=plt.subplots(2,2,figsize=(11,6),sharex='col',layout='constrained')
 colors={'standing':'#969696','contact':'#111111','A':'#1473b4','B':'#da731c'}
 for col,name in enumerate(['new_short03','new_long07']):
  for ctl in ['standing','contact','A','B']:
   ep=json.loads((p/(name+'__'+ctl+'.json')).read_text());c=ep['condition']
   label='Teacher' if ctl=='contact' else ctl
   ax=axes[0,col];ax.plot([r['time'] for r in ep['control']],[r['alpha'] for r in ep['control']],label=label,color=colors[ctl],linewidth=1.5)
   ph=ep['physics'];axes[1,col].plot([r['force_time'] for r in ph],np.linalg.norm([r['force'] for r in ph],axis=1),color=colors[ctl],linewidth=1.1)
  for row in range(2):
   ax=axes[row,col];ax.axvspan(c['disconnect_start'],c['disconnect_end'],color='#e99d20',alpha=.17);ax.axvline(10.4,color='#777777',linestyle=':',linewidth=1);ax.grid(alpha=.2);ax.set_xlim(5,12.4)
  axes[0,col].set_title(f"Unseen gap {c['disconnect_end']-c['disconnect_start']:.2f}s at {c['disconnect_start']:.1f}s")
  axes[0,col].set_ylim(-.02,1.04);axes[1,col].set_ylim(-.1,5.3);axes[1,col].set_xlabel('Simulation time (s)')
 axes[0,0].set_ylabel('Executed skill progress alpha');axes[1,0].set_ylabel('Actual wrist force norm (N)');axes[0,1].legend(ncol=4,fontsize=9,loc='upper left')
 fig.suptitle('One bounded supervised repair: fixed final checkpoints, real detach / reconnect')
 fig.savefig(run/'analysis/sealed-alpha-force.png',dpi=170);plt.close(fig)
if __name__=='__main__':main(Path(sys.argv[1]))
