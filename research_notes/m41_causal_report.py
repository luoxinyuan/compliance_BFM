"""Recompute compact intervention, reward and sampling summaries from saved real physics."""
import json,gzip,sys,time
from pathlib import Path
import numpy as np
start=time.monotonic();R=Path(sys.argv[1]);P=R/sys.argv[2]
rows=json.loads((P/'summaries.json').read_text());out={'runs':[],'sampling':{},'same_state':[],'limitations':['One physical condition and seed0 initial state; repeated stochastic actions use 3 explicit seeds, not broad generalization.','Prefix hashes compare regenerated live runs, not qpos-only state copies.','Frozen-observation queries are not five real physics substeps.']}
queries=json.loads((P/'same-state.json').read_text())
for r in rows:
 with gzip.open(P/(r['label']+'.json.gz'),'rt') as f:e=json.load(f)
 p=e['physics'];ss=e['rl_stats'];returns=np.zeros(len(ss));v=0.
 for j in range(len(ss)-1,-1,-1):v=ss[j]['reward']+.99**(ss[j]['control_steps']/5)*v;returns[j]=v
 row={k:r.get(k) for k in ['label','prefix','suffix','switch','pulse','noise_seed','error_cm','force_rms','cap_fraction','episode_return','root5cm_time','root_velocity_005_200ms','foot_unload_100ms_time','root_x_cm_at14','qpd_max_deg','q_actual_max_deg','boundary','active_tail_error_cm','active_tail_force_rms']}
 row['parity']=r['intervention'];row['discounted_return']=float(returns[0]);row['reward_costs']={k:sum(.02*s['control_steps']*s['terms'][k] for s in ss) for k in ss[0]['terms']};row['phase_metrics']={}
 for name,lo,hi in [('pre',4,4.25),('early',4.25,6.5),('startup',6.5,8),('moving',8,12.05),('hold',12.05,14.05),('released',14.05,19.1)]:
  pp=[x for x in p if lo<=x['force_time']<hi];active=[x for x in pp if x['active']];F=np.array([x['force'] for x in active]);E=np.array([np.array(x['anchor'])-x['wrist'] for x in active]);root=np.array([x['root'] for x in pp]);vv=np.array([x['qvel'][:3] for x in pp])
  row['phase_metrics'][name]=dict(active_error_cm=100*float(np.sqrt(np.mean(np.sum(E*E,axis=1)))) if len(active) else None,force_rms=float(np.sqrt(np.mean(np.sum(F*F,axis=1)))) if len(active) else None,force_xyz_mean=F.mean(0).tolist() if len(active) else None,error_xyz_cm=(100*E.mean(0)).tolist() if len(active) else None,cap_fraction=float(np.mean(np.linalg.norm(F,axis=1)>=4.999)) if len(active) else None,root_displacement_cm=(100*(root[-1]-root[0])).tolist(),root_velocity_mean=vv.mean(0).tolist(),tilt_max=max(x['tilt_deg'] for x in pp),slip_max=max(x['contact_tangent_speed_max_mps'] for x in pp),min_height=min(x['height'] for x in pp))
 if r['label'] in ['I','A32','B32']:
  row['critic_vs_deterministic_return']=[dict(time=t,value=next(x['value'] for x in queries if x['prefix']==r['label'] and x['model']==r['label'] and abs(x['time']-t)<1e-5),return_mc=float(returns[round((t-4)*10)])) for t in [4,5,5.5,6,6.5,7,8]]
 out['runs'].append(row)
for model in ['I','A32','B32']:
 rr=[r for r in out['runs'] if r['prefix']==model and r['noise_seed'] is not None]
 out['sampling'][model]={k:dict(values=[r[k] for r in rr],mean=float(np.mean([r[k] for r in rr])),sample_std=float(np.std([r[k] for r in rr],ddof=1))) for k in ['error_cm','force_rms','episode_return','discounted_return','root_x_cm_at14']}
for prefix in ['I','A32','B32']:
 for t in [4,4.5,5,5.5,6,6.5,7,8]:
  for name in ['A16','A32','B16','B32','B48']:
   x=next(q for q in queries if q['prefix']==prefix and q['model']==name and abs(q['time']-t)<1e-5)
   out['same_state'].append({k:x[k] for k in ['prefix','time','model','raw_rms','raw_norm','z_deg','qpd_substep_rms_deg','value']})
out['analysis_seconds']=time.monotonic()-start
(R/'causal-summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
for r in out['runs']:print(r['label'],round(r['error_cm'],3),round(r['force_rms'],3),round(r['root_x_cm_at14'],2),r['root5cm_time'])
print('SAMPLING',json.dumps(out['sampling'],indent=2))
# Recompute training marginals and completed-episode early-contact coverage.
from collections import Counter,defaultdict
old=R.parent/'transition-curriculum-20261104';coverage={};completed={}
for arm,directory in [('A','A-ppo-03'),('B','B-ppo-04')]:
 logs=[json.loads(l) for l in (old/directory/'job/training.jsonl').read_text().splitlines()]
 family=Counter();phase_counts=Counter();episodes=Counter();cross=defaultdict(Counter);count=0;ratios=[];cos=[]
 for log in logs:
  family.update(log['coverage']['decisions_by_family']);phase_counts.update(log['coverage']['decisions_by_phase']);episodes.update(log['coverage']['completed_episodes_by_family'])
  for g in log['optimization']['gradient_components']:ratios.append(g['ratio']);cos.append(g['cosine'])
  for e in log['episodes']:
   s=e['condition'];n=e['rl_decisions'];count+=n
   for i in range(n):
    t=e['handoff']['time']+.1*i
    phase='no_contact' if not s['enabled'] else 'pre_contact' if t<s['onset'] else 'released' if t>=s['release'] else 'first_2s_of_contact' if t<s['onset']+2 else 'remaining_contact'
    cross[s['demand_family']][phase]+=1
 coverage[arm]=dict(decisions=sum(family.values()),family=dict(family),phase=dict(phase_counts),completed=dict(episodes),retention_ppo_ratio_median=float(np.median(ratios)),retention_ppo_ratio_range=[min(ratios),max(ratios)],retention_ppo_cos_median=float(np.median(cos)),stand_fraction=family['stand_walk']/sum(family.values()),release_fraction=phase_counts['released_recovery']/sum(family.values()))
 completed[arm]=dict(scope='Decision-grid reconstruction for completed episodes only; first2s is a clock window, not detected stepping. Partial episodes not covered.',completed_episode_decisions=count,all_decisions=sum(family.values()),completed_family_cross={k:dict(v) for k,v in cross.items()})
(R/'coverage-summary.json').write_text(json.dumps(coverage,indent=2));(R/'completed-phase-coverage.json').write_text(json.dumps(completed,indent=2))
