"""CPU-only closeout of immutable v3/old-B archives; never selects from regression."""
import argparse,hashlib,json,time,sys,subprocess
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();t=time.monotonic();s=a.stage;o=a.output;o.mkdir(exist_ok=True,parents=True)
 sys.path.insert(0,str(s/'learning-runtime/research_notes'))
 from frl_v3_metrics import summarize,scientific_choice
 train=s/'runs/formal-v3-lambda033-02';old=s.parent/'frl-v2-fb-repair-20261110/runs/frl-formal-B-01';initial=read(train/'dev-0/summaries.json')
 def pack(rows):
  q=summarize(rows);f={}
  for fam in ('near','mid','stop','zero','none'):
   rr=[r for r in rows if r['family']==fam];d={}
   for key in ('error_cm','force_rms','cap_fraction','qpd_max_step_deg','q_actual_max_step_deg'):
    vals=[r['all'].get(key) for r in rr if r['all'].get(key) is not None];d[key]=float(np.median(vals)) if vals else None
   d['moving_wrist_dx_cm']=float(np.median([100*r['absolute']['moving_wrist_dx_m'] for r in rr]));d['children']=len(rr);f[fam]=d
  return {'absolute':q,'family_child_medians':f,'source_failures':sum(bool(r.get('source_failure')) for r in rows),'complete':sum(bool(r.get('complete')) for r in rows)}
 series=[];assets=[];paired=[]
 for k in (0,16,32,64,96,128):
  vr=read(train/f'dev-{k}/summaries.json');br=read(s/f'runs/oldB-newdev-{k:03d}/summaries.json')
  assert [(r['parent_id'],r['child'],r['condition'],r['source_state_sha256']) for r in vr]==[(r['parent_id'],r['child'],r['condition'],r['source_state_sha256']) for r in br]
  v,b=pack(vr),pack(br);v['scientific_coordinate']=scientific_choice(vr,initial)
  series.append({'iteration':k,'v3':v,'old_B':b})
  for arm,path in [('v3',train/f'candidate-{k}.pt' if k else s/'runs/formal-v3-lambda033-01/candidate-0.pt'),('old_B',old/f'candidate-{k}.pt')]:assets.append({'arm':arm,'iteration':k,'path':str(path.relative_to(s.parent)),'sha256':sha(path)})
  for arm,rows in [('v3',vr),('old_B',br)]:
   for r in rows:paired.append({'arm':arm,'iteration':k,**r})
 regression=[]
 for k in (64,128):
  vr=read(s/f'runs/v3-oldreg-{k:03d}/summaries.json');br=read(old/f'dev-{k}/summaries.json')
  # Archived v2 has no v3 absolute additions: preserve original per-child metrics.
  assert [(r['parent_id'],r['child']) for r in vr]==[(r['parent_id'],r['child']) for r in br]
  ff={}
  for fam in ('near','mid','stop','zero','none'):
   ff[fam]={arm:{key:float(np.median([r['all'][key] for r in rows if r['family']==fam and r['all'].get(key) is not None])) if any(r['family']==fam and r['all'].get(key) is not None for r in rows) else None for key in ('error_cm','force_rms')} for arm,rows in [('v3',vr),('old_B',br)]}
  regression.append({'iteration':k,'v3_absolute':summarize(vr),'family_child_medians':ff})
  for arm,rows in [('v3_oldreg',vr),('old_B_oldreg',br)]:
   for r in rows:paired.append({'arm':arm,'iteration':k,**r})
 learning=read(train/'learning.json');assert len(learning)==128 and learning[-1]['RL_decisions']==98304
 counts={'high_decisions':0,'control_steps':0,'physics_steps':0,'terminal_high_decisions':0,'truncated_high_decisions':0}
 risk=[]
 for k in range(1,129):
  rr=read(train/f'rollout-{k}-stats.json');counts['high_decisions']+=len(rr)
  for r in rr:
   for key in ('control_steps','physics_steps'):counts[key]+=r[key]
   counts['terminal_high_decisions']+=int(r.get('terminated',False));counts['truncated_high_decisions']+=int(r.get('truncated',False))
  risk.append({'iteration':k,'event_cost_sum':sum(r['risk_event_sum'] for r in rr),'optimized_reward_sum':sum(r['PPO_reward_native_discounted'] for r in rr)})
 counts.update(completed_episodes=learning[-1]['episodes'],coverage=learning[-1]['coverage'],episode_coverage=learning[-1]['episode_coverage'],accepted_actor_steps=sum(r['optimizer_updates'] for r in learning),critic_steps=sum(r['critic_fit']['optimizer_updates'] for r in learning),rejected_actor_candidates=sum(len(r['rejected_candidates']) for r in learning),max_pre_logprob_error=max(r['pre_logprob_max_error'] for r in learning),max_accepted_KL=max(e['joint_gaussian_KL'] for r in learning for e in r['epochs']),final_retention=learning[-1]['precontact_retention'])
 jobs=[]
 for p in sorted((s/'jobs').glob('*.json')):
  j=read(p)
  if 'elapsed_s' in j:jobs.append({k:j.get(k) for k in ('name','category','status','elapsed_s','runtime','returncode')})
 costs={cat:sum(j['elapsed_s'] for j in jobs if j['category']==cat) for cat in ('P1','P2')}
 disk=sum(p.stat().st_size for root in (s/'runs',s/'jobs',s/'prep') for p in root.rglob('*') if p.is_file())
 result={'status':'completed_partial_scientific_gain_no_promotion','runtime':'85af4353747bd000b50b9fbf06236e23c8d5ef4f','qualification_runtime':'cc1ddf6ee2fd4e5b7b02d7a3b791574f46396150','new_development':series,'old_regression':regression,'training_counts':counts,'models':assets,'jobs':jobs,'job_seconds_measured':costs,'preparation_reserve_seconds':{'P1':1800,'P2':1200},'P1_additional_CPU_seconds':81.63,'output_bytes_runs_jobs_prep':disk,'selection':{'old_relative_iteration':read(train/'selection.json')['iteration'],'scientific_iteration':read(train/'scientific-selection.json')['selected']['iteration'],'promoted':False},'scope':'one paired seed; new development, old regression; no sealed selection; stop children grouped for absolute metrics; family contact medians are child medians'}
 (o/'frl_v3_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 compact=[]
 for r in paired:
  fields=('error_cm','force_rms','cap_fraction','force_xyz_rms','error_xyz_rms_cm','error_xyz_signed_cm','signed_error_cm','slip_peak_mps','slip_p95_mps','yaw_deg_final','qpd_max_step_deg','q_actual_max_step_deg','root_delta_cm','foot_paths_cm','torque_saturation_max')
  compact.append({k:r[k] for k in ('arm','iteration','parent_id','family','child','boundary','complete')}|{'all':{k:r['all'].get(k) for k in fields if k in r['all']},'absolute':r.get('absolute'),'stages':{phase:{k:row[k] for k in fields if k in row} for phase,row in r['quality'].items()}})
 (o/'frl_v3_condition_results.json').write_text(json.dumps(compact,ensure_ascii=False,separators=(',',':'))+'\n')
 (o/'frl_v3_learning_curve.json').write_text(json.dumps({'learning':learning,'risk':risk},indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,3,figsize=(13,7));xx=[r['iteration'] for r in series]
 for arm,label in [('v3','v3 risk'),('old_B','old B')]:
  for akey,axis,scale in [('exceeding_parent_fraction',ax[0,0],100),('family_balanced_severity',ax[0,1],1),('absolute_yaw_p90',ax[0,2],1)]:axis.plot(xx,[r[arm]['absolute'][akey]*scale for r in series],'o-',label=label)
  for fam,axis in zip(('near','mid','stop'),ax[1]):axis.plot(xx,[r[arm]['family_child_medians'][fam]['error_cm'] for r in series],'o-',label=label)
 for axis,title in zip(ax.flat,['Slip-exceeding parents (%)','Integrated risk severity (s)','Absolute yaw p90 (deg)','Near contact error (cm)','Mid contact error (cm)','Stop/continue contact error (cm)']):axis.set_title(title);axis.set_xlabel('PPO batch');axis.grid(alpha=.25);axis.legend()
 fig.suptitle('Issue59: same new development parents, one paired seed; no promotion');fig.tight_layout();fig.savefig(o/'frl_v3_learning_curve.png',dpi=150);plt.close(fig)
 print(json.dumps({'series':[{k:v for k,v in x.items() if k=='iteration'}|{arm:{'absolute':{k:v for k,v in x[arm]['absolute'].items() if k not in ('parent_rows','family')},'families':x[arm]['family_child_medians']} for arm in ('v3','old_B')} for x in series if x['iteration'] in (32,128)],'regression':regression,'counts':counts,'costs':costs,'disk_GiB':disk/2**30,'selection':result['selection'],'analysis_s':time.monotonic()-t},ensure_ascii=False))
if __name__=='__main__':main()
