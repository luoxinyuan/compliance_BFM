"""Plots original accepted updates and actual option coverage, no new evaluation."""
import os,time,json,hashlib
from pathlib import Path
START=time.monotonic();S=Path(__file__).resolve().parents[1];O=S/'runs/final-curves-01';O.mkdir(exist_ok=False)
os.environ['MPLCONFIGDIR']=str(S/'prep/mpl-final-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
coverage=json.loads((S/'runs/final-training-01/training-coverage.json').read_text());records={}
for arm in('E','I'):
 study=json.loads((S/f'runs/formal-{arm}-06/result.json').read_text())['study'];rr=[]
 for b in study['records']:
  ss=[v for v in b['metrics']['steps']if not v.get('auxiliary',False)];acc=[v for v in ss if v['accepted']];kk=[t['KL']for v in acc for t in v['trials']if t.get('acceptable')and t.get('alpha')==v.get('selected_alpha')]
  rr.append(dict(end_row=b['end_row'],phase=b['phase'],accepted_total=b['actor_accepted_total'],accepted=len(acc),proposals=len(ss),joint_max=max([k['joint']for k in kk],default=None),categorical_max=max([k['categorical']for k in kk],default=None),lambda_keep=b['metrics']['next_lambda'],old_logprob_error=b['metrics']['old_logprob_max_error'],collection_s=b['collection_s'],update_s=b['update_s'],commit_IO_s=b['commit_IO_s']))
 records[arm]=rr
fig,ax=plt.subplots(2,2,figsize=(11,7.5))
for a,color in(('E','tab:blue'),('I','tab:orange')):
 rr=records[a];x=[v['end_row']for v in rr];ax[0,0].plot(x,[v['accepted_total']for v in rr],label=a,color=color)
 ax[0,1].plot(x,[v['joint_max']for v in rr],color=color,label=a+' joint')
 ax[1,0].plot(x,[max(v['categorical_max']or 0,1e-12)for v in rr],color=color,label=a)
for aa in(ax[0,0],ax[0,1],ax[1,0]):aa.axvline(12000,c='gray',ls='--',lw=1);aa.set_xlabel('Actual online 50Hz transitions');aa.legend();aa.grid(alpha=.2)
ax[0,0].set_title('Accepted Actor updates (not auxiliary updates)');ax[0,1].axhline(.02,c='r',ls=':',label='locked joint KL');ax[0,1].set_title('Max accepted joint KL per real block');ax[1,0].set_yscale('log');ax[1,0].set_title('Categorical part of accepted joint KL')
labels=['standing','arm5','arm15','motion'];x=np.arange(4);width=.23
for j,(a,kind)in enumerate((('E','P3 training'),('I','P3 training'),('Dev','deterministic'))):
 if a=='Dev':v=np.array([6805,0,4825,0])/11630
 else:v=np.array([next(r['rows']for r in coverage[a]['option_rows']if r['phase']=='P3'and r['option']==k)for k in range(4)])/20000
 ax[1,1].bar(x+(j-1)*width,100*v,width,label=a+' '+kind)
ax[1,1].set_xticks(x,labels);ax[1,1].set_ylabel('% actual 20ms rows');ax[1,1].set_title('Sampled training != deterministic endpoint use');ax[1,1].legend(fontsize=8)
fig.suptitle('Contact-Goal E/I v2: optimization and command coverage\nFixed endpoints; no intermediate Dev curve or checkpoint selection',fontsize=12);fig.tight_layout();fig.savefig(O/'contact_goal_v2_final_curves.png',dpi=160);plt.close(fig)
(O/'curves.json').write_text(json.dumps(records,indent=2)+'\n')
meta=dict(status='CURVES_FROM_ARCHIVES_COMPLETE',wall_s=time.monotonic()-START,physics_ticks=0,policy_updates=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),ledger_merge_pending=True)
(O/'result.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
