"""P4000 actor warm start + original transactional raw-Gaussian PPO, finite stage."""
import m11_ipc
import argparse,os,json,time,signal,multiprocessing as mp,subprocess,hashlib,shutil,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent

def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actor_hash(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():
  if k.startswith('actor.') or k in ['center','scale']:h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def cost(row):
 v=row['all'];return v['force_rms']+.03*(v['error_cm'] or 0)+v['cap_fraction']
def assess(rows,base):
 current={r['label'].split('__')[0]:r for r in rows};b={r['label'].split('__')[0]:r for r in base};delta=[cost(current[k])-cost(b[k]) for k in current];score=sum(cost(v) for v in current.values())/len(current)+.25*max(0.,max(delta));reasons=[]
 for k,r in current.items():
  v=r['all'];old=b[k]['all']
  if r['boundary']:reasons.append(k+':physical boundary')
  if v['error_cm'] is not None and (v['error_cm']>old['error_cm']+max(20.,.5*old['error_cm']) or v['force_rms']>old['force_rms']+2):reasons.append(k+':severe active deterioration')
  if k in ['short08','none'] and r['root_max_displacement']>b[k]['root_max_displacement']+.1:reasons.append(k+':unrequested extra motion')
 return dict(score=score,severe=bool(reasons),reasons=reasons,case_costs={k:cost(v) for k,v in current.items()},max_absolute_cost_increase=max(delta))

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--iterations',type=int,default=32);p.add_argument('--resume',type=Path);p.add_argument('--std',type=float,required=True);p.add_argument('--max-seconds',type=float,default=6000);a=p.parse_args();assert a.seed in [5201,5202] and a.iterations in [32,64] and a.std in [.008,.004];a.output.mkdir(exist_ok=False)
 os.environ.update(CUDA_VISIBLE_DEVICES='0',CLEAN_KIND='direct',P4000_SEED=str(a.seed),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 import torch,numpy as np
 from clean_policy import Expert,update
 from humanoidverse.force_compliance.parallel import receive,merge_rollouts
 from p4000_worker import launch
 plan=json.loads(Path(__file__).with_name('p4000_protocol.json').read_text());torch.set_num_threads(1);torch.manual_seed(a.seed);torch.set_grad_enabled(False)
 commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 atomic(a.output/'launch.json',dict(runtime=commit,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},protocol_sha256=sha(Path(__file__).with_name('p4000_protocol.json')),initial_sha256=plan['initial']['sha256']))
 ctx=mp.get_context('spawn');ps=[];cs=[];net=opt=None;iteration=decisions=updates=0;status='initializing';started=time.monotonic();best=float('inf');rollbacks=0;base=None
 def stop(*_):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop)
 def weights():return {k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
 def save(name):
  f=a.output/name;t=f.with_suffix('.tmp');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),input_dim=1088,kind='direct',iteration=iteration,decisions=decisions,optimizer_updates=updates,seed=a.seed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),runtime=commit,initial_sha256=plan['initial']['sha256'],actor_function_sha256=actor_hash(net),critic_trained_by_PPO=iteration>0,executed_std=a.std,rollbacks=rollbacks),t);t.replace(f);return f
 def evaluate(name,sampled=False):
  folder=a.output/name;folder.mkdir();schedule=[(k+'__'+('sample' if sampled else 'mean'),dict(v,sampling_seed=a.seed*101+17),'sample' if sampled else 'mean') for k,v in plan['dev'].items()];w=weights()
  for i,c in enumerate(cs):c.send(dict(op='evaluate',policy=w,schedule=schedule[i::len(cs)],output=str(folder)))
  rows=[]
  for c,proc in zip(cs,ps):rows+=receive(c,proc,timeout=1200)['summaries']
  atomic(folder/'summary.json',rows);return rows
 try:
  for i in range(3):
   pc,cc=ctx.Pipe();proc=ctx.Process(target=launch,args=(cc,'0',i,a.output/f'worker-{i}'));proc.start();cc.close();cs.append(pc);ps.append(proc)
  ready=[receive(c,p,900) for c,p in zip(cs,ps)];assert all(x['input_dim']==1088 for x in ready);atomic(a.output/'workers.json',ready)
  net=Expert(1088,'direct').cuda();initial=Path(plan['initial']['path']);assert sha(initial)==plan['initial']['sha256'];ck=torch.load(initial,map_location='cuda',weights_only=True);net.load_state_dict(ck['state_dict']);initial_actor=actor_hash(net)
  torch.manual_seed(a.seed+17);fresh=Expert(1088,'direct').cuda();net.critic.load_state_dict(fresh.critic.state_dict());del fresh;net.log_std.fill_(np.log(a.std));net.log_std.requires_grad_(False)
  opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=plan['training']['actor_lr']),dict(params=net.critic.parameters(),lr=plan['training']['critic_lr'])]);assert actor_hash(net)==initial_actor
  atomic(a.output/'initialization.json',dict(actor_normalizer_function_sha256=initial_actor,P4000_SHA=sha(initial),BC_critic_discarded=True,Adam_fresh=True,std=a.std,retention_weight=0.,no_online_expert=True))
  if a.resume:
   prev=a.resume.parent;ck=torch.load(a.resume,map_location='cuda',weights_only=True);assert ck['seed']==a.seed and ck['executed_std']==a.std and ck['initial_sha256']==plan['initial']['sha256'];net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);iteration=ck['iteration'];decisions=ck['decisions'];updates=ck['optimizer_updates'];rollbacks=ck['rollbacks'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu());base=json.loads((prev/'baseline-dev.json').read_text());sel=json.loads((prev/'selection.json').read_text());best=sel['score'];assert sha(prev/'best.pt')==sel['sha256'];shutil.copyfile(prev/'best.pt',a.output/'best.pt');atomic(a.output/'selection.json',dict(sel,inherited_selection=True,original_evaluation=sel['evaluation']));atomic(a.output/'resume.json',dict(path=str(a.resume),sha256=sha(a.resume),samplers='fresh actual source episodes, no fabricated prefix; new costs accounted',optimizer_loaded=True));atomic(a.output/'baseline-dev.json',base)
  else:
   save('initial.pt');base=evaluate('dev-0000');atomic(a.output/'baseline-dev.json',base);b=assess(base,base);best=b['score'];save('best.pt');atomic(a.output/'selection.json',dict(iteration=0,score=best,sha256=sha(a.output/'best.pt'),actor_function_sha256=actor_hash(net),evaluation=str(a.output/'dev-0000/summary.json'),rule=plan['selection']))
   evaluate('sample-0000',True)
  status='running'
  while iteration<a.iterations:
   if time.monotonic()-started>a.max_seconds-300:status='time_budget';break
   before_update=weights();t=time.monotonic()
   for c in cs:c.send(dict(op='rollout',steps=256,version=iteration,policy=before_update))
   results=[receive(c,p,1200) for c,p in zip(cs,ps)];assert all(r['version']==iteration for r in results);sample_seconds=time.monotonic()-t;batch={k:v.cuda() for k,v in merge_rollouts(results,.95).items()}
   # merge_rollouts flattens [time,worker], NOT worker-major. Keep metadata aligned.
   stats=[results[w]['stats'][j] for j in range(256) for w in range(3)]
   assert len(stats)==len(batch['reward'])==768
   warm=None
   if iteration==0:
    before_actor=actor_hash(net);targets=batch['returns'].detach();old_ev=float(1-(targets-net.value(batch['obs'])).var(unbiased=False)/targets.var(unbiased=False).clamp_min(1e-8))
    with torch.enable_grad():
     for _ in range(100):
      ix=torch.randint(len(targets),(256,),device='cuda');loss=.5*((net.value(batch['obs'][ix])-targets[ix])/10).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.critic.parameters(),1.);opt.step()
    assert actor_hash(net)==before_actor
    for r in results:
     r['batch']['value']=net.value(torch.as_tensor(r['batch']['obs'],device='cuda')).cpu().numpy();r['batch']['next_value']=net.value(torch.as_tensor(r['batch']['next_obs'],device='cuda')).cpu().numpy()
    warm=dict(value_only_updates=100,physical_samples_reused=768,extra_RL_samples=0,actor_unchanged=True,initial_target_EV=old_ev,post_EV=float(1-(targets-net.value(batch['obs'])).var(unbiased=False)/targets.var(unbiased=False).clamp_min(1e-8)),GAE='recomputed per-worker with newly fit current/next values; actor and actual old logprob unchanged')
    batch={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};atomic(a.output/'critic-warmup.json',warm)
   batch['raw_adv_std']=(batch['returns']-batch['value']).std();before_mean=net.distribution(batch['obs']).mean.clone();t=time.monotonic();u=update(net,opt,batch,epochs=6,minibatch=256);opt_seconds=time.monotonic()-t
   with torch.no_grad():
    after=net.distribution(batch['obs']);change=after.mean-before_mean;logchange=after.log_prob(batch['action']).sum(-1)-batch['logp'];z0=before_mean/before_mean.norm(dim=-1,keepdim=True).clamp_min(1e-6);z1=after.mean/after.mean.norm(dim=-1,keepdim=True).clamp_min(1e-6);angles=torch.rad2deg(torch.acos((z0*z1).sum(-1).clamp(-1,1)))
   groups={}
   for field in ['demand_family','phase']:
    for group in sorted(set(x[field] for x in stats)):
     ix=torch.tensor([i for i,x in enumerate(stats) if x[field]==group],device='cuda');adv=batch['advantages'][ix];lc=logchange[ix];groups[field+':'+group]=dict(decisions=len(ix),reward_mean=float(batch['reward'][ix].mean()),return_mean=float(batch['returns'][ix].mean()),advantage_mean=float(adv.mean()),positive_fraction=float((adv>0).float().mean()),positive_adv_logprob_change=float(lc[adv>0].mean()) if (adv>0).any() else None,negative_adv_logprob_change=float(lc[adv<0].mean()) if (adv<0).any() else None,mean_z_update_deg=float(angles[ix].mean()))
   iteration+=1;decisions+=768;updates+=u['optimizer_updates'];active=sum(x['active_steps'] for x in stats)
   row=dict(iteration=iteration,RL_decisions=decisions,optimizer_updates=updates,sampling_seconds=sample_seconds,optimization_seconds=opt_seconds,decisions_per_second=768/sample_seconds,physical_steps=sum(x['physics_steps'] for x in stats),BFM_control_steps=sum(x['control_steps'] for x in stats),reset_prefill_steps=sum(r['prefill_physics_steps'] for r in results),source_steps=sum(r['source_physics_steps'] for r in results),reward_mean=float(batch['reward'].mean()),reward_terms={k:float(np.mean([x['terms'][k] for x in stats])) for k in stats[0]['terms']},force_RMS=float((sum(x['active_force_squared'] for x in stats)/max(active,1))**.5),cap_fraction=sum(x['cap_steps'] for x in stats)/max(active,1),groups=groups,episodes=[e for r in results for e in r['episodes']],exploration={k:dict(mean=float(np.mean([x[k] for x in stats])),max=max(x[k] for x in stats)) for k in ['raw_mean_norm','raw_sample_norm','raw_noise_norm','z_sample_mean_angle_deg','executed_std']},mean_update_z_angle=dict(mean=float(angles.mean()),max=float(angles.max())),optimization=u,elapsed_s=time.monotonic()-started)
   with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   atomic(a.output/'status.json',dict(status=status,**row));save('latest.pt');print('P4000_PPO',a.seed,iteration,decisions,'force',round(row['force_RMS'],3),'reward',round(row['reward_mean'],4),'dps',round(row['decisions_per_second'],2),'KL',u['epochs'][-1]['joint_gaussian_KL'],flush=True)
   if iteration%8==0:
    rows=evaluate(f'dev-{iteration:04d}');v=assess(rows,base);candidate=save(f'candidate-{iteration:04d}.pt');v['iteration']=iteration
    if not v['severe'] and v['score']<best:
     best=v['score'];shutil.copyfile(candidate,a.output/'best.pt');atomic(a.output/'selection.json',dict(iteration=iteration,score=best,sha256=sha(a.output/'best.pt'),actor_function_sha256=actor_hash(net),evaluation=str(a.output/f'dev-{iteration:04d}/summary.json'),rule=plan['selection']))
    atomic(a.output/f'dev-{iteration:04d}/score.json',dict(v,best_score=best));print('P4000_DEV',a.seed,iteration,v['score'],v['severe'],'best',best,flush=True)
    if iteration in [16,32,48,64]:evaluate(f'sample-{iteration:04d}',True)
    if v['severe']:
     selected=torch.load(a.output/'best.pt',map_location='cuda',weights_only=True);net.load_state_dict(selected['state_dict']);opt.load_state_dict(selected['optimizer']);rollbacks+=1;lr=plan['training']['actor_lr']*.5**rollbacks;opt.param_groups[0]['lr']=lr;opt.param_groups[0]['nominal_actor_lr']=lr;atomic(a.output/f'rollback-{iteration:04d}.json',dict(reasons=v['reasons'],restored_iteration=selected['iteration'],restored_sha256=sha(a.output/'best.pt'),new_actor_lr=lr,current_real_sampler_states_retained=True))
  else:status='completed'
  save('last.pt')
 except KeyboardInterrupt:
  status='interrupted'
  if net is not None:save('interrupted.pt')
 except BaseException:
  import traceback
  status='failed';(a.output/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  for c in cs:
   try:c.send(dict(op='close'))
   except (BrokenPipeError,EOFError,OSError):pass
  for proc in ps:
   proc.join(timeout=10)
   if proc.is_alive():proc.terminate();proc.join(timeout=10)
  atomic(a.output/'result.json',dict(status=status,seed=a.seed,iterations=iteration,RL_decisions=decisions,PPO_optimizer_updates=updates,elapsed_s=time.monotonic()-started,workers=[dict(pid=p.pid,exitcode=p.exitcode) for p in ps],BFM_updated=False,Student_updated=False))
if __name__=='__main__':main()
