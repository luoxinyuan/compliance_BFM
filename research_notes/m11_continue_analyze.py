"""Read-only statistics from M11 acquisition; no policy selection or new rollout."""
import json,os,statistics
from pathlib import Path
import numpy as np
from m11_metrics import summarize
from m11_selection import sha,atomic,completed_job
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-continuation-20261102')
def read(p):return json.loads(p.read_text())
def main():
 out=Path(os.environ['BFM_RECOVERY_OUTPUT']);report=dict(arms={},final={},notes=['One learner seed per arm; common checkpoint and optimizer. Stochastic final has3 seeds, not3 independent training runs.','Selection frozen before final; exposed old final is development only.','Actual shared force/physics/model unchanged.'])
 for arm in ['fixed','annealed']:
  p=completed_job(RUN,arm);logs=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()];eps=[e for r in logs for e in r['completed_episodes']];sel=read(p/'selection.json');assert sha(p/'best.pt')==sel['checkpoint_sha256'] and sha(Path(sel['evaluation_path']))==sel['evaluation_sha256']
  dev=[dict(batch=int(d.name.split('-')[1]),score=read(d/'score.json'),rows=read(d/'summary.json')) for d in sorted(p.glob('dev-*'))]
  o=[r['optimization'] for r in logs];g=[x for z in o for x in z['gradients']]
  report['arms'][arm]=dict(added_decisions=logs[-1]['added_decisions'],batches=len(logs),physics_steps=sum(r['physics_steps'] for r in logs),control_steps=sum(r['control_steps'] for r in logs),prefill_steps=sum(r['prefill_physics_steps'] for r in logs),optimizer_updates=logs[-1]['optimizer_updates'],completed_episodes=len(eps),failures=[e for e in eps if e['boundary']],selection=sel,dev=dev,max_old_logprob_error=max(x['pre_logprob_max_error'] for x in o),actual_std_first=logs[0]['actual_std_mean'],actual_std_last=logs[-1]['actual_std_mean'],gradients_mean={k:float(np.mean([x[k] for x in g])) for k in ['actor','critic','std','total','clip_scale']},last20_old_value_EV=float(np.mean([x['old_GAE_value_EV'] for x in o[-20:]])),last20_updated_value_EV=float(np.mean([x['updated_value_EV'] for x in o[-20:]])),last20_joint_KL=float(np.mean([x['epochs'][-1]['joint_gaussian_KL'] for x in o[-20:]])),last20_clip_fraction=float(np.mean([x['epochs'][-1]['clip_fraction'] for x in o[-20:]])))
 for arm in ['prior','fixed','annealed']:
  p=RUN/('final-'+arm+'-attempt-01/outputs/job/evaluation');rows=[summarize(f) for f in sorted(p.glob('*.json.gz'))];report['final'][arm]=rows;atomic(out/(arm+'-physical-metrics.json'),rows)
 atomic(out/'comparison.json',report)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,2,figsize=(13,11))
 for arm in ['fixed','annealed']:
  p=completed_job(RUN,arm);logs=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()];xs=[r['added_decisions'] for r in logs];ds=report['arms'][arm]['dev']
  axes[0,0].plot(xs,[r['actual_std_mean'] for r in logs],label=arm)
  axes[0,1].plot([d['batch']*768 for d in ds],[d['score']['rank'][1] for d in ds],'-o',label=arm)
  for sample,style in [('__mean','-'),('__sample202','--')]:
   rows=[next(r for r in d['rows'] if r['label']=='exposed_straight'+sample) for d in ds]
   axes[1,0].plot([d['batch']*768 for d in ds],[r['error_cm'] for r in rows],style+'o',label=arm+sample)
   axes[1,1].plot([d['batch']*768 for d in ds],[r['force_rms'] for r in rows],style+'o',label=arm+sample)
  axes[2,0].plot(xs,[np.mean([g['actor'] for g in r['optimization']['gradients']]) for r in logs],label=arm+' actor')
  axes[2,0].plot(xs,[np.mean([g['critic'] for g in r['optimization']['gradients']]) for r in logs],alpha=.5,label=arm+' critic')
  axes[2,1].plot(xs,[r['execution']['qPD_vs_prior_rms_deg'] for r in logs],label=arm+' actual qPD residual RMS')
 for ax,title in zip(axes.flat,['Executed std','Development mean/tail rank cost','Exposed failure: mean/sample error cm','Exposed failure: mean/sample force N','Pre-clipping gradient norm','Actual sampled qPD change vs prior (deg)']):ax.set_title(title);ax.set_xlabel('Additional RL decisions');ax.grid(alpha=.3);ax.legend(fontsize=8)
 fig.suptitle('M11 matched continuation: development curves; final never selects checkpoints');fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
 lines=['# M11 continuation raw comparison (pending scientific review)','', 'All model selections precede final tests. No winner is inferred merely from training reward.','', '|condition|M10 error cm / force N|fixed mean|annealed mean|','|---|---:|---:|---:|']
 for r in report['final']['prior']:
  if not r['label'].endswith('__mean'):continue
  label=r['label'];vals=[]
  for arm in ['prior','fixed','annealed']:
   x=next(z for z in report['final'][arm] if z['label']==label);vals.append(f"{x['error_cm']:.3f} / {x['force_rms']:.3f}" if x['error_cm'] is not None else f"no contact; root {1000*x['root_max_displacement']:.3f}mm")
  lines.append('|'+label+'|'+'|'.join(vals)+'|')
 (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');atomic(out/'result.json',dict(status='completed',report=str(out/'comparison.json'),policy_updated=False,final_used_for_selection=False))
if __name__=='__main__':main()
