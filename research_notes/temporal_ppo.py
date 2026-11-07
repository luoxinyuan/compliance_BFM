"""Independent privileged Expert acquisition; native motion prior, finite stages."""
import m11_ipc
import argparse,json,os,time,signal,multiprocessing as mp,subprocess,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['independent','continuous'],required=True);p.add_argument('--parent',type=Path);p.add_argument('--kind',choices=['direct'],required=True);p.add_argument('--exploration',choices=['ordinary','guided'],required=True);p.add_argument('--stage',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--iterations',type=int,default=120);p.add_argument('--seed',type=int,default=101);p.add_argument('--execution-gain',type=float,default=1.0);p.add_argument('--resume',type=Path);p.add_argument('--domain',default='neighborhood',choices=['single','neighborhood']);p.add_argument('--max-seconds',type=float,default=28000);a=p.parse_args()
 assert 0<a.execution_gain<=16
 assert not a.output.exists();a.output.mkdir();os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CLEAN_KIND=a.kind,NARROW_DOMAIN=a.domain,NARROW_SEED=str(a.seed),NARROW_EXECUTION_GAIN=str(a.execution_gain),NATIVE_SEED=str(a.seed),TEMPORAL_ARM=a.arm,TEMPORAL_SOURCE_OFFSET='0')
 import torch,numpy as np
 from humanoidverse.force_compliance.parallel import receive,merge_rollouts
 from temporal_policy import Expert,rho_for
 from clean_policy import update
 from temporal_worker import launch
 plan=json.loads(Path(__file__).with_name('temporal_protocol.json').read_text());from native_env import dev_specs;plan['dev']=dev_specs();torch.set_num_threads(1);torch.manual_seed(a.seed);torch.set_grad_enabled(False)
 ctx=mp.get_context('spawn');ps=[];cs=[];net=opt=None;iteration=decisions=updates=start_iteration=start_decisions=collected_decisions=0;best=float('inf');status='initializing';started=time.monotonic();commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 atomic(a.output/'launch.json',dict(pid=os.getpid(),commit=commit,plan_sha256=sha(Path(__file__).with_name('temporal_protocol.json')),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},started=time.time()))
 def stop(*_):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop)
 def weights():return {k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
 def save(name):
  f=a.output/name;t=f.with_suffix('.tmp');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),input_dim=n,iteration=iteration,decisions=decisions,optimizer_updates=updates,seed=a.seed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),runtime=commit,source_student_online=False,old_actor_loaded=False,parameterization='native covariance-aligned affine mean head',arm=a.arm,rho=float(net.correlation),execution_gain=a.execution_gain,kind=a.kind,replay_dim=net.replay_dim),t);t.replace(f);return f
 def evaluate(schedule,name):
  d=a.output/name;d.mkdir();w=weights()
  for i,c in enumerate(cs):c.send(dict(op='evaluate',policy=w,schedule=schedule[i::len(cs)],output=str(d)))
  rows=[]
  for c,proc in zip(cs,ps):rows+=receive(c,proc,timeout=1200)['summaries']
  atomic(d/'summary.json',rows);return rows
 initial_rows=[]
 def select(rows):
  from temporal_selection import assess
  if not initial_rows:initial_rows.extend(rows)
  return assess(rows,initial_rows)['score']
 def dev():
  schedule=[(k+'__mean',v,'learned') for k,v in plan['dev'].items()]
  rows=evaluate(schedule,f'dev-{iteration:04d}');return rows,select(rows)
 try:
  os.environ['TEMPORAL_SOURCE_OFFSET']=str(1600000 if a.parent else 0)
  for i in range(3):
   parent,child=ctx.Pipe();proc=ctx.Process(target=launch,args=(child,a.gpu,i,a.output/f'worker-{i}'));proc.start();child.close();ps.append(proc);cs.append(parent)
  ready=[receive(c,p,900) for c,p in zip(cs,ps)];n=ready[0]['input_dim'];assert all(r['input_dim']==n for r in ready);assert all(r['contract']['model_sha256']==plan['physical']['BFM_sha256'] for r in ready);atomic(a.output/'workers.json',ready)
  net=Expert(n,rho=rho_for(a.arm)).cuda();net.log_std.requires_grad_(False)
  opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=plan['training']['actor_lr']),dict(params=net.critic.parameters(),lr=3e-4)])
  assert a.exploration=='guided' and a.resume
  native_root=Path(plan['initialization']['path']).parent
  assert sha(native_root/'native-directions.npy')==plan['exploration']['direction_sha256']
  directions=torch.from_numpy(np.load(native_root/'native-directions.npy')).cuda()
  ck=torch.load(a.resume,map_location='cuda',weights_only=True)
  if a.parent:
   assert ck['iteration']==16 and a.iterations==32 and ck['arm']==a.arm
   net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer'])
   iteration=ck['iteration'];decisions=ck['decisions'];updates=ck['optimizer_updates']
   torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
   initial_rows.extend(json.loads((a.parent/'dev-0000/summary.json').read_text()))
  else:
   assert a.resume.name=='native-initial.pt' and sha(a.resume)==plan['initialization']['sha256']
   assert ck['old_actor_loaded'] is False
   net.initialize_from_raw(ck['state_dict']);net.set_covariance(directions)
   torch.manual_seed(a.seed+17);fresh=Expert(n,rho=rho_for(a.arm)).cuda();net.critic.load_state_dict(fresh.critic.state_dict());del fresh
  atomic(a.output/'resume.json',dict(path=str(a.resume),sha256=sha(a.resume),optimizer_loaded=bool(a.parent),critic='same fresh seed within pair; extension retains actual state',rho=float(net.correlation),native_initial=plan['initialization']))
  assert a.iterations in [16,32]
  start_iteration=iteration;start_decisions=decisions
  save('initial.pt')
  atomic(a.output/'retention.json',dict(kind='none; first minimal covariance exploration comparison',old_high_actor_loaded=False,exploration=a.exploration,raw_covariance=(net.raw_L@net.raw_L.T).cpu().tolist(),raw_marginal_std=(net.raw_L.square().sum(-1).sqrt()).cpu().tolist()))
  rows,current_score=dev()
  if a.parent:
   selected=json.loads((a.parent/'selection.json').read_text());best=selected['score']
   shutil.copyfile(a.parent/'best.pt',a.output/'best.pt');atomic(a.output/'selection.json',selected)
  else:
   best=current_score;save('best.pt');atomic(a.output/'selection.json',dict(iteration=0,score=best,sha256=sha(a.output/'best.pt'),evaluation='dev-0000/summary.json',rule=plan['selection']))
  original=json.loads((native_root/'ppo-coordinate/dev-0000/summary.json').read_text())
  if not a.parent:
   by={r['label']:r for r in original};eq={k:max(abs(r[k]-by[r['label']][k]) for r in rows) for k in ['error_cm','force_rms','root_max_displacement']}
   atomic(a.output/'initial-physical-equivalence.json',dict(max_differences=eq,scope='same native mean, no old actor/no temporal noise in mean eval'));assert max(eq.values())<1e-5,eq
  iterations=a.iterations
  def coverage(results):
   from collections import Counter
   return dict(phase=dict(Counter(s['phase'] for r in results for s in r['stats'])),family=dict(Counter(s['demand_family'] for r in results for s in r['stats'])),completed=dict(Counter(e['condition'].get('demand_family','unknown') for r in results for e in r['episodes'])))
  history=[];rollbacks=0
  status='running'
  for it in range(start_iteration,iterations):
   if time.monotonic()-started>a.max_seconds-240:status='time_budget';break
   t=time.monotonic();w=weights()
   for c in cs:c.send(dict(op='rollout',steps=256,version=it,policy=w))
   results=[receive(c,p,1200) for c,p in zip(cs,ps)];sample_s=time.monotonic()-t
   assert all(r['version']==it for r in results)
   batch={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};batch['raw_adv_std']=(batch['returns']-batch['value']).std();collected_decisions+=len(batch['reward']);atomic(a.output/'sampled-decisions.json',dict(new_RL_decisions=collected_decisions,version=it,source='real worker batches before optimizer update'));t=time.monotonic();before_raw=net.raw_mean(batch['obs']).detach().clone();u=update(net,opt,batch);actual_delta=net.raw_mean(batch['obs']).detach()-before_raw;native_delta=actual_delta@directions.T;orthogonal_delta=actual_delta-native_delta@directions
   iteration+=1;decisions+=len(batch['reward']);updates+=u['optimizer_updates'];stats=[s for r in results for s in r['stats']];active=sum(s['active_steps'] for s in stats)
   task_advantage={}
   for family in sorted(set(s['demand_family'] for s in stats)):
    ix=torch.tensor([j for j,s in enumerate(stats) if s['demand_family']==family],device='cuda');task_advantage[family]=dict(decisions=len(ix),adv_mean=float(batch['advantages'][ix].mean()),positive_fraction=float((batch['advantages'][ix]>0).float().mean()),return_mean=float(batch['returns'][ix].mean()))
   row=dict(task_advantage=task_advantage,iteration=iteration,rl_decisions=decisions,optimizer_updates=updates,sampling_seconds=sample_s,optimization_seconds=time.monotonic()-t,decisions_per_second=len(batch['reward'])/sample_s,exploration=a.exploration,actual_raw_mean_change_rms=float(actual_delta.square().mean().sqrt()),native_span_mean_change_norm=float(native_delta.norm(dim=-1).mean()),orthogonal_mean_change_norm=float(orthogonal_delta.norm(dim=-1).mean()),raw_mean_norm=float(net.raw_mean(batch['obs']).norm(dim=-1).mean()),raw_marginal_std_mean=float(net.raw_L.square().sum(-1).sqrt().mean()),physical_steps=sum(s['physics_steps'] for s in stats),control_steps=sum(s['control_steps'] for s in stats),prefill_physics_steps=sum(r['prefill_physics_steps'] for r in results),source_physics_steps=sum(r['source_physics_steps'] for r in results),force_rms=(sum(s['active_force_squared'] for s in stats)/max(active,1))**.5,cap_fraction=sum(s['cap_steps'] for s in stats)/max(active,1),reward=float(batch['reward'].mean()),reward_terms={k:float(np.mean([s['terms'][k] for s in stats])) for k in stats[0]['terms']},execution={k:float(np.mean([s['execution'][k] for s in stats])) for k in stats[0]['execution']},episodes=[e for r in results for e in r['episodes']],coverage=coverage(results),optimization=u,elapsed_s=time.monotonic()-started)
   with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   atomic(a.output/'status.json',dict(status=status,**row));save('latest.pt');print('NARROW_UPDATE',iteration,decisions,'force',round(row['force_rms'],3),'rate',round(row['decisions_per_second'],2),'KL',u['epochs'][-1]['joint_gaussian_KL'],flush=True)
   if iteration%plan['training']['evaluate_every']==0:
    rows,score=dev();candidate=save(f'candidate-{iteration:04d}.pt')
    from temporal_selection import assess
    assessment=assess(rows,initial_rows);destructive=not assessment['admissible']
    if score<best and not destructive:
     best=score;shutil.copyfile(candidate,a.output/'best.pt');atomic(a.output/'selection.json',dict(iteration=iteration,score=score,sha256=sha(a.output/'best.pt'),evaluation=f'dev-{iteration:04d}/summary.json',rule=plan['selection']))
    atomic(a.output/f'dev-{iteration:04d}/score.json',dict(score=score,best=best,assessment=assessment));print('NARROW_DEV',iteration,score,best,flush=True)
    history.append(dict(iteration=iteration,score=score,best=best))
    if destructive:
     prior=torch.load(a.output/'best.pt',map_location='cuda',weights_only=True);net.load_state_dict(prior['state_dict']);opt.load_state_dict(prior['optimizer']);rollbacks+=1
     lr=plan['training']['actor_lr']*(.5**rollbacks);opt.param_groups[0]['lr']=lr;opt.param_groups[0]['nominal_actor_lr']=lr
     atomic(a.output/f'rollback-{iteration:04d}.json',dict(reason='preregistered physical-unit deterioration or safety boundary',assessment=assessment,restored_iteration=prior['iteration'],restored_sha256=sha(a.output/'best.pt'),new_actor_lr=lr,sampler_history='real current states retained, new batch under restored distribution'))
     # No asymmetric performance early stop; maximum32 batches each.

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
  for p in ps:
   p.join(timeout=10)
   if p.is_alive():p.terminate();p.join(timeout=10)
  atomic(a.output/'result.json',dict(status=status,iterations=iteration,decisions=decisions,new_decisions=decisions-start_decisions,collected_new_decisions=collected_decisions,updates=updates,elapsed_s=time.monotonic()-started,workers=[dict(pid=p.pid,exitcode=p.exitcode) for p in ps],bfm_updated=False,student_updated=False))
if __name__=='__main__':main()
