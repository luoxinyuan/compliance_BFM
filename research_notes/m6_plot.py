"""Export true physical traces for the two sealed main conditions."""
import json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m6_analysis import RUN,read

def main():
 ledger=read(RUN/'compute-ledger.json');a=next(x for x in ledger['attempts'] if x['stage']=='final' and x.get('result_status')=='completed');src=Path(a['path'])/'outputs'
 fig,axs=plt.subplots(4,2,figsize=(13,12),sharex='col');colors=dict(native_hold='0.5',rule='tab:green',teacher='black',m5='tab:purple',alignment='tab:orange',distill='tab:blue',single_negative='tab:red',force_rule='tab:cyan')
 for col,name in enumerate(['test_pos','test_neg']):
  for ctl,color in colors.items():
   path=src/(name+'__'+ctl+'.json.gz')
   if not path.exists():continue
   e=read(path);r=e['physics'];c=e['control'];on=e['condition']['onset'];rel=e['condition']['release'];t=np.array([x['force_time']-on for x in r]);ct=np.array([x['time']-on for x in c]);active=np.array([x['active'] for x in r]);err=np.array([x['error']*100 for x in r]);err[~active]=np.nan
   axs[0,col].plot(t,err,color=color,label=ctl,lw=1);axs[1,col].plot(t,np.linalg.norm([x['force'] for x in r],axis=1),color=color,lw=1)
   w=np.array([x['wrist'] for x in r]);root=np.array([x['root'] for x in r]);ix=np.argmin(abs(t));axs[2,col].plot(t,(w[:,0]-root[:,0]-w[ix,0]+root[ix,0])*100,color=color,lw=1)
   alpha=[x['native_coords'][0] if x.get('native_coords') is not None else x['alpha'] if ctl in ['teacher','rule','force_rule'] else 0 for x in c];axs[3,col].plot(ct,alpha,color=color,lw=1)
  for row in range(4):axs[row,col].axvline(0,color='0.5',ls=':');axs[row,col].axvline(rel-on,color='0.5',ls=':');axs[row,col].grid(alpha=.2)
  axs[0,col].set_title(name);axs[3,col].set_xlabel('Time from contact onset (s)')
 for row,label in enumerate(['3D contact error (cm)','Actual force norm (N)','Wrist-root X increment (cm)','Native forward progress']):axs[row,0].set_ylabel(label)
 axs[0,0].legend(fontsize=8,ncol=2);fig.suptitle('M6 frozen final policies: sealed conditions, full release included');fig.tight_layout();fig.savefig(RUN/'analysis/m6-physical-curves.png',dpi=150)
if __name__=='__main__':main()
