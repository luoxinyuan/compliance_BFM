"""Read-only arithmetic on saved PPO buffers; no model forward/training/physics/GPU."""
import pathlib,json,numpy as np,statistics,math
S=pathlib.Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/p4000-feedback-ppo-20261108');R=S/'runs/seed5201-32';logs=[json.loads(t) for t in (R/'training.jsonl').read_text().splitlines()];upto=len(logs)
out={};u=[r['optimization'] for r in logs]
out['update_budget']=dict(batches=upto,nominal_max_gradient_steps=upto*6*3,accepted_steps=sum(x['optimizer_updates'] for x in u),accepted_per_batch=[x['optimizer_updates'] for x in u],rejected_steps=sum(len(x['rejected_candidates']) for x in u),actor_early_stop_at_least=sum(x['optimizer_updates']<18 for x in u),critic_same_accepted_steps_as_actor=True,initial_value_only_steps=100,KL_minmax=[min(x['epochs'][-1]['joint_gaussian_KL'] for x in u),max(x['epochs'][-1]['joint_gaussian_KL'] for x in u)],mean_actor_update_degrees=float(np.mean([x['mean_update_z_angle']['mean'] for x in logs])),mean_sample_angle=float(np.mean([x['exploration']['z_sample_mean_angle_deg']['mean'] for x in logs])),mean_clip_fraction=float(np.mean([x['epochs'][-1]['clip_fraction'] for x in u])),mean_actor_gradient_clip_scale=float(np.mean([g['actor_clip_scale'] for x in u for g in x['gradients']])))
# Recompute original GAE from actual saved worker sequences without flatten mixing.
phases={};checks=[];all_terms={};cap_active=cap_total=0
for version in range(upto):
 workers=[]
 for w in range(3):
  p=R/f'worker-{w}/rollout-{version:04d}.npz';st=R/f'worker-{w}/rollout-{version:04d}-stats.json'
  if not(p.exists() and st.exists()):continue
  a=dict(np.load(p));stats=json.loads(st.read_text());assert len(stats)==len(a['reward'])
  carry=0.;adv=np.zeros(len(stats));mc=np.zeros(len(stats));nextret=float(a['next_value'][-1])
  for j in range(len(stats)-1,-1,-1):
   cont=1-float(a['done'][j]);delta=float(a['reward'][j])+float(a['discount'][j])*cont*float(a['next_value'][j])-float(a['value'][j]);carry=delta+float(a['discount'][j])*.95*cont*carry;adv[j]=carry;nextret=float(a['reward'][j])+float(a['discount'][j])*cont*nextret;mc[j]=nextret
  workers.append((a,stats,adv,mc))
 if len(workers)!=3:continue
 # First batch saved values precede the documented critic-only warmup; exclude from comparison to actual logged advantages.
 cat=np.stack([x[2] for x in workers],axis=1).reshape(-1);norm=(cat-cat.mean())/max(cat.std(),1e-6)
 for w,(a,stats,adv,mc) in enumerate(workers):
  for j,st in enumerate(stats):
   q=phases.setdefault(st['phase'],dict(n=0,pos_raw=0,pos_centered=0,flipped_pos_to_neg=0,raw_adv=[],centered_adv=[],td=[],mc_minus_v=[],mc=[],V=[],rewards=[],force_cost=[],force_cap_n=0,active_n=0,error_cm=[]));q['n']+=1;q['pos_raw']+=bool(adv[j]>0);q['pos_centered']+=bool(norm[j*3+w]>0);q['flipped_pos_to_neg']+=bool(adv[j]>0 and norm[j*3+w]<0);q['raw_adv'].append(float(adv[j]));q['centered_adv'].append(float(norm[j*3+w]));q['mc_minus_v'].append(float(mc[j]-a['value'][j]));q['mc'].append(float(mc[j]));q['V'].append(float(a['value'][j]));q['rewards'].append(float(a['reward'][j]));q['force_cost'].append(st['terms']['force']);q['active_n']+=st['active_steps'];q['force_cap_n']+=st['cap_steps'];
   if st['active_steps']:q['error_cm'].append(100*math.sqrt(st['error_squared']/st['active_steps']))
   for k,v in st['terms'].items():all_terms[k]=all_terms.get(k,0)+.02*st['control_steps']*v
   cap_active+=st['active_steps'];cap_total+=st['cap_steps']
 for family in set(t['demand_family'] for _,st,_,_ in workers for t in st):
  ix=[j*3+w for w,(_,st,_,_) in enumerate(workers) for j,t in enumerate(st) if t['demand_family']==family];vals=norm[ix];log=logs[version]['groups'].get('demand_family:'+family)
  if version>0 and log:checks.append(abs(vals.mean()-log['advantage_mean']))
out['phases']={}
for k,q in phases.items():
 v=np.array(q['V']);mc=np.array(q['mc']);out['phases'][k]=dict(n=q['n'],raw_positive=q['pos_raw']/q['n'],centered_positive=q['pos_centered']/q['n'],positive_to_negative=q['flipped_pos_to_neg']/q['n'],mean_raw_adv=float(np.mean(q['raw_adv'])),mean_centered_adv=float(np.mean(q['centered_adv'])),mean_bootstrapped_MC_minus_V=float(np.mean(q['mc_minus_v'])),value_EV_vs_bootstrapped_MC=float(1-np.var(mc-v)/max(np.var(mc),1e-8)),mean_reward=float(np.mean(q['rewards'])),force_cost_mean=float(np.mean(q['force_cost'])),force_cost_std=float(np.std(q['force_cost'])),cap_fraction=q['force_cap_n']/max(q['active_n'],1),mean_error_cm=float(np.mean(q['error_cm'])) if q['error_cm'] else None)
out['advantages_recomputed_family_max_difference_after_first_batch']=max(checks);out['reward_cost_sums']=all_terms;out['active_cap_fraction']=cap_total/max(cap_active,1);out['trace_weights']={str(t):(.99*.95)**(t/.1) for t in [.5,1.,2.,3.,5.]};out['limits']='MC bootstraps batch boundary from saved critic; firstbatch arrays before value warmup, raw adv not final firstbatch; no causal intervention or model training. Phase signs are diagnostics, not a proof of wrong gradient.'
p=S/'audit-readonly-20261108';p.mkdir(exist_ok=True);(p/'update-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
