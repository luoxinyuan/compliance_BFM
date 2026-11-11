"""Archive-only Issue58 diagnostics: no optimizer, simulator step, or sealed reads."""
import os,time,json,gzip,hashlib,math,traceback,itertools
from pathlib import Path
from collections import Counter,defaultdict,deque
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['P4000_CANONICAL_ROW_ACTOR']='1'
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_grad_enabled(False)
P=Path(os.environ['BFM_PROJECT_ROOT']);OLD=P/'experiments/frl-v2-fb-repair-20261110';ST=P/'experiments/frl-diagnostics-20261111';OUT=Path(os.environ.get('BFM_DIAGNOSTIC_OUT',str(ST/'results')));OUT.mkdir(exist_ok=True)
t0=time.monotonic();inputs={};timings={}
def read(p):
 assert 'sealed' not in str(p)
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist() if isinstance(a,np.ndarray) else str(a)))
def q(x):
 a=np.asarray(x,dtype=float)
 return dict(n=len(a),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max=float(a.max()),min=float(a.min()),mean=float(a.mean())) if len(a) else dict(n=0)
def rot(qp):
 w,x,y,z=np.asarray(qp[3:7]);return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def load(arm,k):
 p=OLD/f'runs/frl-formal-{arm}-01/candidate-{k}.pt';inputs[str(p.relative_to(P))]=sha(p)
 ck=torch.load(p,map_location='cpu',weights_only=False);return ck['model']
def mlp(x,s,p,n):
 for i in range(n):
  x=F.linear(x,s[f'{p}.{2*i}.weight'],s[f'{p}.{2*i}.bias'])
  if i<n-1:x=torch.tanh(x)
 return x
def mean(s,x):
 norm=((x[:,:1639]-s['base.center'])/s['base.scale']).clamp(-10,10)
 return torch.cat([mlp(a[None],s,'base.actor',3)+mlp(b[None],s,'branch',2) for a,b in zip(norm,x[:,1639:])])
def lp(mu,raw,std):return torch.distributions.Normal(mu,std).log_prob(raw).sum(-1)
def gae(a):
 r=torch.from_numpy(a['reward']);v=torch.from_numpy(a['value']);nv=torch.from_numpy(a['next_value']);d=torch.from_numpy(a['discount']);te=torch.from_numpy(a['terminated']);tr=torch.from_numpy(a['truncated']);out=torch.zeros_like(r);carry=torch.zeros_like(r[0])
 for t in reversed(range(len(r))):
  delta=r[t]+d[t]*(~te[t]).to(r.dtype)*nv[t]-v[t];carry=delta+d[t]*.95*(~(te[t]|tr[t])).to(r.dtype)*carry;out[t]=carry
 return out,(out-out.mean())/out.std(unbiased=False).clamp_min(1e-6)
def gate_scalars(row):
 ph=row['quality'];slip=max(v['slip_peak_mps'] for v in ph.values());yaw=abs(ph.get('released',list(ph.values())[-1])['yaw_deg_final']);return slip,yaw
models={'initial':('B',0),'A128':('A',128),'B64':('B',64),'B128':('B',128)}
# D2 gate audit: exact archived selector scalars, no new evaluation.
ss={m:read(OLD/f'runs/frl-formal-{a}-01/dev-{k}/summaries.json') for m,(a,k) in models.items()}
base={(r['parent_id'],r['child']):r for r in ss['initial']};gaterows=[];gategroups={}
for fam in ['near','mid','stop','zero','none']:
 rr=[r for r in ss['initial'] if r['family']==fam];slips=[gate_scalars(r)[0] for r in rr];yaws=[gate_scalars(r)[1] for r in rr]
 gategroups[fam]=dict(subepisodes=len(rr),parents=len(set(r['parent_id'] for r in rr)),slip=q(slips),abs_released_yaw=q(yaws),slip_threshold=q([s+max(.2,.5*s) for s in slips]),yaw_threshold=q([y+10 for y in yaws]),slip_threshold_family_percentile=q([100*np.mean(np.array(slips)<=s+max(.2,.5*s)) for s in slips]),yaw_threshold_family_percentile=q([100*np.mean(np.array(yaws)<=y+10) for y in yaws]),absolute_reference_slip_exceed=sum(s>.45 for s in slips),absolute_reference_yaw_exceed=sum(y>20 for y in yaws),self_gate_fail=0,physical_boundary=sum(bool(r['boundary']) for r in rr))
for m,rr in ss.items():
 for r in rr:
  b=base[(r['parent_id'],r['child'])];s,y=gate_scalars(r);bs,by=gate_scalars(b)
  gaterows.append(dict(model=m,parent_id=r['parent_id'],child=r['child'],family=r['family'],slip=s,abs_released_yaw=y,slip_limit=bs+max(.2,.5*bs),yaw_limit=by+10,slip_excess=s-bs-max(.2,.5*bs),yaw_excess=y-by-10,slip_gate_pass=s<=bs+max(.2,.5*bs),yaw_gate_pass=y<=by+10,peak_slip_phase=max(r['quality'],key=lambda p:r['quality'][p]['slip_peak_mps'])))
write(OUT/'quality-gates.json',dict(groups=gategroups,rows=gaterows,definition='exact original select scalars;32parents40subepisodes;init self relative gate is tautological;M1 .45m/s and20deg only reference, not replacement',native_same_condition='not found: native source-prefix only exists before task; P0B native cases differ in task/source and are not matched continuation'))
print('GATES_DONE',flush=True);timings['gate_s']=time.monotonic()-t0
# D3 actual optimized rewards, matched full development episodes only.
rewardrows=[];episode_recompute_error=0.;native_recompute_error=0.;reconstruction=[]
for m,(arm,k) in models.items():
 run=OLD/f'runs/frl-formal-{arm}-01/dev-{k}';states=[]
 for summary in ss[m]:
  label=summary['parent_id'][:12]+'-'+summary['child'];ep=read(run/(label+'.json.gz'));cs=ep['control'];phys=ep['physics'];spec=ep['condition'];start=spec['handoff_time'];R=0.;stage=defaultdict(float);terms=defaultdict(float)
  phases={}
  for name,v in summary['quality'].items():phases[name]=(v['start'],v['end'])
  for c in cs:
   dt=c['actual_duration_s'];re=-dt*sum(c['reward_terms'].values());native_recompute_error=max(native_recompute_error,abs(re-c['reward_integral']));w=.99**((c['time']-start)/.1);R+=w*re
   name=next((n for n,(aa,bb) in phases.items() if aa-1e-8<=c['time']<bb-1e-8),'unclassified');stage[name]+=w*re
   for name,v in c['reward_terms'].items():terms[name]+=-w*dt*v
  # Verify per100ms PPO reward saved in rl_stats independently from stored per20ms terms.
  ci=0
  for st in ep['rl_stats']:
   cc=cs[ci:ci+st['control_steps']];ci+=st['control_steps'];elapsed=0.;v=0.
   for c in cc:v+=.99**(elapsed/.1)*c['reward_integral'];elapsed+=c['actual_duration_s']
   episode_recompute_error=max(episode_recompute_error,abs(v-sum(.99**(jj*.02/.1)*rr for jj,rr in enumerate(st['native_rewards']))))
  slip,yaw=gate_scalars(summary);a=summary['all'];fam=summary['family'];bounds=(4.,4.) if fam=='near' else (8.,7.)
  taskok=a['error_cm'] is not None and a['error_cm']<=bounds[0] and a['force_rms']<=bounds[1] and not ep['boundary']
  if fam=='stop' and summary['child']=='main':
   h=summary['quality'].get('holding');taskok=bool(taskok and h and h['error_cm'] is not None and h['error_cm']<=8 and h['force_rms']<=6.5)
  category='not_applicable' if fam in ('zero','none') else ('failure' if not taskok else ('slipping_success' if slip>.45 else ('yaw_only_success' if yaw>20 else 'clean_success')))
  rewardrows.append(dict(model=m,parent_id=summary['parent_id'],child=summary['child'],family=fam,return_discounted=R,stage_returns=dict(stage),term_returns=dict(terms),error_cm=a['error_cm'],force=a['force_rms'],cap=a['cap_fraction'],slip=slip,yaw=yaw,boundary=ep['boundary'],category=category,physics_rows=len(phys),controls=len(cs)))
  # Reconstruct full high input from saved privileged row plus complete actual contact history.
  prefix=read(run/(label+'-source-prefix.json.gz'));hist=deque(prefix['audit']['force_history'],maxlen=100);priv=np.array([d['obs'] for d in ep['decisions']],dtype=np.float32);ci=0;max_hist_err=0.;xs=[];bodies=[]
  for j,st in enumerate(ep['rl_stats']):
   c=cs[ci];rot0=rot(c['pre_qpos']);h=np.zeros((100,3,3))
   for hh,sample in enumerate(hist):
    for contact in sample:h[-len(hist)+hh,contact['slot']]=rot0.T@np.array(contact['force'])/5
   enc=np.zeros((3,6));enc[0]=[1,1,0,0,0,0]
   max_hist_err=max(max_hist_err,float(np.max(abs(h[-25:].reshape(-1)-priv[j,825:1050]))))
   sx=np.r_[priv[j,:465],priv[j,565:821],h.reshape(-1),enc.reshape(-1)].astype(np.float32)
   assert sx.shape==(1639,);xs.append(np.r_[sx,priv[j]]);bodies.append(c['bfm_obs'])
   # next current boundary samples: next control has exact measured force; endpoint prepared once before next decision.
   for jj in range(st['control_steps']):
    ci+=1
    if ci<len(cs):
     nc=cs[ci];hist.append([dict(slot=0,force=nc['measured_force_xyz'])])
  states.extend(xs)
  reconstruction.append(dict(model=m,label=label,contact_history_max_error=max_hist_err,n=len(xs)))
 # Frozen mean reproduction on actual development states. Numerical backend CPU is diagnostic only.
 x=torch.tensor(np.asarray(states),dtype=torch.float32);state=load(arm,k);mu=mean(state,x);stored=[]
 for summary in ss[m]:
  label=summary['parent_id'][:12]+'-'+summary['child'];ep=read(run/(label+'.json.gz'));stored.extend(d['raw'] for d in ep['decisions'])
 err=float((mu-torch.tensor(stored)).abs().max());write(OUT/(m+'-dev-reconstruction.json'),dict(rows=len(x),raw_max_error=err,history_max_error=max(r['contact_history_max_error'] for r in reconstruction if r['model']==m)))
 np.savez_compressed(OUT/(m+'-dev-inputs.npz'),obs=x.numpy())
 print('DEV_DONE',m,len(x),err,flush=True)
# Strict matched ranking; no tradeoff ordering invented.
pairs=[];cs_pairs=[]
by=defaultdict(list)
for r in rewardrows:by[(r['parent_id'],r['child'])].append(r)
for key,rr in by.items():
 for a,b in itertools.combinations(rr,2):
  if a['family'] in ('zero','none'):continue
  va=np.array([a[k] for k in ['error_cm','force','cap','slip','yaw']]);vb=np.array([b[k] for k in ['error_cm','force','cap','slip','yaw']])
  direction=1 if (np.all(va<=vb) and np.any(va<vb)) else (-1 if (np.all(vb<=va) and np.any(vb<va)) else 0)
  dif=a['return_discounted']-b['return_discounted'];pairs.append(dict(parent_id=key[0],child=key[1],models=[a['model'],b['model']],quality_pareto_direction=direction,reward_difference=dif,agreement=bool(direction*dif>0) if direction else None))
  if {a['category'],b['category']}=={'clean_success','slipping_success'}:
   clean=a if a['category']=='clean_success' else b;slip=b if a['category']=='clean_success' else a
   cs_pairs.append(dict(parent_id=key[0],child=key[1],clean_model=clean['model'],slipping_model=slip['model'],slip_reward_higher=slip['return_discounted']>clean['return_discounted'],return_difference_slip_minus_clean=slip['return_discounted']-clean['return_discounted']))
ranked=[p for p in pairs if p['quality_pareto_direction']];write(OUT/'reward-ranking.json',dict(rows=rewardrows,category_counts=dict(Counter(r['category'] for r in rewardrows)),model_category_counts={m:dict(Counter(r['category'] for r in rewardrows if r['model']==m)) for m in models},matched_pairs=len(pairs),pareto_comparable=len(ranked),pareto_agree=sum(p['agreement'] for p in ranked),pareto_disagree=sum(not p['agreement'] for p in ranked),pareto_tradeoffs=len(pairs)-len(ranked),clean_slipping_pairs=cs_pairs,clean_slipping_count=len(cs_pairs),slipping_reward_higher=sum(p['slip_reward_higher'] for p in cs_pairs),per_native_integral_error=native_recompute_error,high_reward_error=episode_recompute_error,pairs=pairs))
write(OUT/'reconstruction.json',reconstruction);timings['gate_and_dev_s']=time.monotonic()-t0
print('RANKING_DONE',len(ranked),len(cs_pairs),flush=True)
# D1 existing actual sampled batches; proper before/after version, actual normalized PPO credit.
chains={}
for label,(arm,k) in {'B64':('B',64),'B128':('B',128),'A128':('A',128)}.items():
 run=OLD/f'runs/frl-formal-{arm}-01';z=np.load(run/f'rollout-{k}.npz');arr={a:z[a] for a in z.files};stat=read(run/f'rollout-{k}-stats.json');phy=read(run/f'physics-{k}.json.gz');ctrl=read(run/f'native-controls-{k}.json.gz');old=load(arm,k-1);new=load(arm,k);x=torch.from_numpy(arr['obs']);raw=torch.from_numpy(arr['action']);before=mean(old,x);after=mean(new,x);std=old['base.log_std'].exp();oldlp=lp(before,raw,std);lp_error=(oldlp-torch.from_numpy(arr['logp'])).abs().numpy();adv,normal=gae(arr);de=raw-before;dm=after-before;dot=(de*dm).sum(-1).numpy();lpd=(lp(after,raw,std)-oldlp).numpy();cos=((de*dm).sum(-1)/(de.norm(dim=-1)*dm.norm(dim=-1)).clamp_min(1e-30)).numpy();ang=lambda a,b:torch.rad2deg(torch.acos(((a.double()*b.double()).sum(-1)/(a.double().norm(dim=-1)*b.double().norm(dim=-1))).clamp(-1,1))).numpy()
 pc=0;cc=0;rows=[]
 for j,st in enumerate(stat):
  ph=phy[pc:pc+st['physics_steps']];pc+=st['physics_steps'];co=ctrl[cc:cc+st['control_steps']];cc+=st['control_steps'];active=all(p['active'] for p in ph);slip=max(p['stage_loaded_slip_mps'] for p in ph);yaw=max(abs(math.degrees(math.atan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0]))) for p in ph);e0=float(ph[0]['error']);e1=float(ph[-1]['error']);prog=None
  if j+1<len(stat) and stat[j+1]['parent_id']==st['parent_id'] and abs(stat[j+1]['decision_time_s']-st['decision_time_s']-.1)<1e-6:
   pp=arr['obs'][j,1639:];pn=arr['obs'][j+1,1639:];q0=co[0]['pre_qpos'];q1=ctrl[cc]['pre_qpos'];w0=np.asarray(q0[:3])+rot(q0)@pp[480:483];w1=np.asarray(q1[:3])+rot(q1)@pn[480:483];prog=float(w1[0]-w0[0])
  moving=st['phase']=='moving' and active;clean=slip<=.45 and yaw<=20 and max(p['joint_limit_excess_max_rad'] for p in ph)<=.02 and st['boundary'] is None
  favorable=bool(moving and clean and prog is not None and prog>0 and e1<e0)
  rows.append(dict(index=j,parent_id=st['parent_id'],family=st['family'],phase=st['phase'],time=st['decision_time_s'],active=active,moving=moving,clean=clean,observed_favorable_proxy=favorable,wrist_dx_m=prog,error_start=e0,error_end=e1,slip=slip,yaw=yaw,GAE_raw=float(adv[j]),PPO_adv=float(normal[j]),direction_dot=float(dot[j]),direction_cosine=float(cos[j]),sample_logprob_delta=float(lpd[j]),old_logprob_reproduction_error=float(lp_error[j])))
 def summarize(rr):
  fav=[r for r in rr if r['observed_favorable_proxy']];credit=[r for r in fav if r['PPO_adv']>0];ret=[r for r in credit if r['direction_dot']>0];return dict(rows=len(rr),moving=sum(r['moving'] for r in rr),clean_moving=sum(r['moving'] and r['clean'] for r in rr),favorable=len(fav),positive_raw_adv=sum(r['GAE_raw']>0 for r in fav),positive_PPO_adv=len(credit),credited_toward_sample=len(ret),credited_logprob_up=sum(r['sample_logprob_delta']>0 for r in credit),favorable_not_credited=len(fav)-len(credit),credited_not_toward=len(credit)-len(ret),adv_raw=q([r['GAE_raw'] for r in fav]),adv_normal=q([r['PPO_adv'] for r in fav]),credited_cosine=q([r['direction_cosine'] for r in credit]),credited_logprob_delta=q([r['sample_logprob_delta'] for r in credit]))
 result=dict(summary=summarize(rows),by_family={f:summarize([r for r in rows if r['family']==f]) for f in sorted(set(r['family'] for r in rows))},old_logprob_max_error=float(lp_error.max()),old_logprob_within_original_assertion=bool(lp_error.max()<3e-4),sample_z_angle_deg=q(ang(before,raw)),mean_update_z_angle_deg=q(ang(before,after)),rows=rows,proven_causal_good_action_fraction='unidentified: no same-state physical counterfactual per sampled action')
 log=read(run/'learning.json')[k-1];result['logged_update']=dict(accepted=log['optimizer_updates'],epochs=log['epochs'],raw_adv_std=log['raw_adv_std'],gradients=log['gradients']);result['GAE_std_reproduction_error']=abs(float(adv.std())-log['raw_adv_std'])
 if label=='B64':
  later=mean(load('B',128),x);late_dot=((later-before)*de).sum(-1).numpy();ix=[r['index'] for r in rows if r['observed_favorable_proxy'] and r['PPO_adv']>0];result['B128_on_B63_sampling_state']=dict(credited=len(ix),still_toward_sample=sum(late_dot[i]>0 for i in ix),z_change_deg=q(ang(before,later)),logprob_change=q((lp(later,raw,std)-oldlp).numpy()[ix]))
 # Exact sample/dev state overlap, never nearest-neighbor substitutions.
 dx=np.load(OUT/(label+'-dev-inputs.npz'))['obs'];trainhash={hashlib.sha256(v.tobytes()).hexdigest() for v in arr['obs']};devhash=[hashlib.sha256(v.tobytes()).hexdigest() for v in dx];result['exact_development_overlap']=dict(dev_rows=len(dx),exact_rows=sum(h in trainhash for h in devhash),training_rows=len(x),same_state_counterfactual_dev_samples=0)
 chains[label]=result;write(OUT/'exploration-chain.json',chains);print('CHAIN_DONE',label,result['summary'],flush=True)
# Frozen same-dev-state mean comparisons (no sampling/counterfactual attribution).
comparison={};states={m:load(*models[m]) for m in ['initial','A128','B64','B128']}
for host in ['B64','B128','A128']:
 x=torch.tensor(np.load(OUT/(host+'-dev-inputs.npz'))['obs']);mus={m:mean(s,x) for m,s in states.items()};pairs={}
 for a,b in itertools.combinations(mus,2):pairs[a+'->'+b]=dict(raw_l2=q((mus[b]-mus[a]).norm(dim=-1).numpy()),z_angle_deg=q(ang(mus[a],mus[b])))
 comparison[host]=dict(rows=len(x),pairs=pairs)
write(OUT/'dev-same-state-means.json',comparison)
write(OUT/'identity-manifest.json',dict(source='a687d528b1119413bc1a8f73eb5f199229e27d4d',model_files=inputs,script_sha256=sha(Path(__file__)),protocol_sha256=sha(ST/'prep/protocol.json'),runtime_historical='19b3117c8f808510a0c387e7f3de267a67cf7e90'))
write(OUT/'ledger.json',dict(status='completed',job_s=time.monotonic()-t0,timings=timings,device='CPU only;CUDA_VISIBLE_DEVICES empty',torch_threads=1,training=0,RL_decisions=0,new_physics_steps=0,sealed_reads=0,full_dev_episodes=160,diagnostic_actual_sampled_decisions=2304,claim_scope='retrospective observational archive audit, not new qualification or new sealed selection'))
print('COMPLETED',time.monotonic()-t0,flush=True)
