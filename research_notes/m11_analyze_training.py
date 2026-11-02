"""Post hoc training diagnostics from immutable logs; no model or simulation calls."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True)
 logs=[json.loads(s) for s in (a.job/'training.jsonl').read_text().splitlines()]
 eps=[e for r in logs for e in r['completed_episodes']]
 dev=[]
 for d in a.job.glob('dev-*'):
  if not (d/'summary.json').exists():continue
  i=0 if d.name=='dev-initial' else int(d.name.split('-')[1]);rows=json.loads((d/'summary.json').read_text())
  dev.append(dict(iteration=i,score=float(np.mean([r['force_rms']+.03*r['error_cm']+r['cap_fraction']+20*(r['boundary'] is not None) for r in rows])),rows=rows))
 dev.sort(key=lambda v:v['iteration'])
 o=[r['optimization'] for r in logs]
 report=dict(training_commit=json.loads((a.job/'launch.json').read_text())['commit'],iterations=len(logs),high_level_decisions=logs[-1]['rl_decisions'],physics_steps=sum(r['physical_steps'] for r in logs),control_steps=sum(r['control_steps'] for r in logs),reset_and_visited_prefix_physics_steps=sum(r['prefill_physics_steps'] for r in logs),optimizer_minibatch_updates=logs[-1]['optimizer_updates'],completed_training_episodes=len(eps),recorded_training_failures=[e for e in eps if e['boundary']],dev_episodes=sum(len(d['rows']) for d in dev),dev_failures=[r for d in dev for r in d['rows'] if r['boundary']],max_pre_logprob_error=max(r['pre_logprob_max_error'] for r in o),last20_old_value_EV_mean=float(np.mean([r['old_GAE_value_EV'] for r in o[-20:]])),last20_post_value_EV_mean=float(np.mean([r['updated_value_EV'] for r in o[-20:]])),last20_joint_KL_mean=float(np.mean([r['epochs'][-1]['joint_gaussian_KL'] for r in o[-20:]])),last20_clip_fraction_mean=float(np.mean([r['epochs'][-1]['clip_fraction'] for r in o[-20:]])),initial_raw_std=o[0]['std_mean'],final_raw_std=o[-1]['std_mean'],selection=json.loads((a.job/'selection.json').read_text()),dev=dev,notes=['Physics steps exclude preparation and evaluation; prefix counts are separately measured.','Completed episode counts exclude unfinished episodes discarded at dev resets.','Seed0 physical resets plus randomized conditions; one learner seed, not multi-seed evidence.','Best selected by dev composite score, not final tests. Completed budget does not establish convergence.'])
 for f in ['initial.pt','best.pt','last.pt']:
  report[f+'_sha256']=hashlib.sha256((a.job/f).read_bytes()).hexdigest()
 (a.output/'training-analysis.json').write_text(json.dumps(report,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,2,figsize=(12,8));x=[r['iteration'] for r in logs]
 for name in ['straight','left','right','recovery']:
  xx=[d['iteration'] for d in dev];rr=[next(r for r in d['rows'] if r['label']==name+'__learned') for d in dev]
  ax[0,0].plot(xx,[r['force_rms'] for r in rr],'-o',label=name)
  ax[0,1].plot(xx,[r['error_cm'] for r in rr],'-o',label=name)
 for z in ax[0]:z.legend();z.axvline(report['selection']['iteration'],color='grey',linestyle=':');z.set_xlabel('PPO iteration (768 decisions each)')
 ax[0,0].set_ylabel('Dev actual force RMS (N)');ax[0,1].set_ylabel('Dev 3D error RMS (cm)')
 ax[1,0].plot(x,[r['old_GAE_value_EV'] for r in o],label='old value used for GAE');ax[1,0].plot(x,[r['updated_value_EV'] for r in o],label='post-update value');ax[1,0].legend();ax[1,0].set_ylabel('Explained variance (same batch)')
 ax[1,1].plot([d['iteration'] for d in dev],[d['score'] for d in dev],'-o');ax[1,1].set_ylabel('Predeclared dev composite (lower better)')
 for z in ax.flat:z.grid(alpha=.25)
 fig.suptitle('M11 privileged expert: development evidence, not final-test tuning');fig.tight_layout();fig.savefig(a.output/'learning-curves.png',dpi=150);plt.close(fig)
 print(json.dumps({k:v for k,v in report.items() if k!='dev'},indent=2))

if __name__=='__main__':main()
