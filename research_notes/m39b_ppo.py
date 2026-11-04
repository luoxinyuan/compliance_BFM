"""Independent privileged Expert acquisition; native motion prior, finite stages."""
import m11_ipc
import argparse,json,os,time,signal,multiprocessing as mp,subprocess,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['structured','direct'],required=True);p.add_argument('--stage',choices=['ppo','calibrate'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--iterations',type=int,default=120);p.add_argument('--seed',type=int,default=101);p.add_argument('--execution-gain',type=float,default=1.0);p.add_argument('--resume',type=Path);p.add_argument('--domain',default='neighborhood',choices=['single','neighborhood']);p.add_argument('--max-seconds',type=float,default=28000);a=p.parse_args()
 assert 0<a.execution_gain<=16
 assert not a.output.exists();a.output.mkdir();os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CLEAN_KIND=a.kind,NARROW_DOMAIN=a.domain,NARROW_SEED=str(a.seed),NARROW_EXECUTION_GAIN=str(a.execution_gain))
 import torch,numpy as np
 from humanoidverse.force_compliance.parallel import receive,merge_rollouts
 from m39b_update import Expert,update
 from m39b_worker import launch
 plan=json.loads(Path(__file__).with_name('m39b_protocol.json').read_text());torch.set_num_threads(1);torch.manual_seed(a.seed);torch.set_grad_enabled(False)
 ctx=mp.get_context('spawn');ps=[];cs=[];net=opt=None;iteration=decisions=updates=0;best=float('inf');status='initializing';started=time.monotonic();commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 atomic(a.output/'launch.json',dict(pid=os.getpid(),commit=commit,plan_sha256=sha(Path(__file__).with_name('m39b_protocol.json')),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},started=time.time()))
 def stop(*_):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop)
 def weights():return {k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
 def save(name):
  f=a.output/name;t=f.with_suffix('.tmp');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),input_dim=n,iteration=iteration,decisions=decisions,optimizer_updates=updates,seed=a.seed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),runtime=commit,source_student_online=False,execution_gain=a.execution_gain,kind=a.kind),t);t.replace(f);return f
 def evaluate(schedule,name):
  d=a.output/name;d.mkdir();w=weights()
  for i,c in enumerate(cs):c.send(dict(op='evaluate',policy=w,schedule=schedule[i::len(cs)],output=str(d)))
  rows=[]
  for c,proc in zip(cs,ps):rows+=receive(c,proc,timeout=1200)['summaries']
  atomic(d/'summary.json',rows);return rows
 baseline_scores={}
 def select(rows):
  from m39b_metrics import cost
  values={r['label']:cost(r) for r in rows if r['label'].split('__')[0] in plan['dev']}
  if not baseline_scores:baseline_scores.update(values)
  relative=[v/max(baseline_scores[k],.1) for k,v in values.items()]
  return .5*max(relative)+sum(relative)/len(relative)
 def dev():
  schedule=[(k+'__mean',v,'learned') for k,v in plan['dev'].items()]
  rows=evaluate(schedule,f'dev-{iteration:04d}');return rows,select(rows)
 try:
  for i in range(3):
   parent,child=ctx.Pipe();proc=ctx.Process(target=launch,args=(child,a.gpu,i,a.output/f'worker-{i}'));proc.start();child.close();ps.append(proc);cs.append(parent)
  ready=[receive(c,p,900) for c,p in zip(cs,ps)];n=ready[0]['input_dim'];assert all(r['input_dim']==n for r in ready);atomic(a.output/'workers.json',ready)
  net=Expert(n).cuda();net.log_std.requires_grad_(False)
  if a.kind=='direct':net.actor[-1].bias.copy_(torch.tensor(ready[0]['standing'],device='cuda')/16)
  opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=plan['training']['actor_lr']),dict(params=net.critic.parameters(),lr=3e-4)])
  assert a.resume and sha(a.resume)==plan['initial']['sha256']
  ck=torch.load(a.resume,map_location='cuda',weights_only=True)
  from geometry_policy import expand
  expand(net,ck)
  # All model tensors inherited from common best64; Adam starts identically fresh.
  # Same inherited critic in both arms; fresh Adam only.
  atomic(a.output/'resume.json',dict(path=str(a.resume),sha256=sha(a.resume),optimizer_loaded=False,critic='same inherited best64 in both arms',std=float(net.log_std.exp().mean())))
  assert a.iterations<=64
  assert a.resume is not None, 'Initialize from separately verified preparation; not random latent exploration'
  save('initial.pt')
  from m39b_replay import Replay,calibrate
  replay=Replay(plan,ready[0]['contract'],ready[0]['standing'],a.output,os.environ.get('M39B_ARM','A'))
  if a.stage=='calibrate':
   w=weights()
   for c in cs:c.send(dict(op='rollout',steps=256,version=0,policy=w))
   results=[receive(c,p,1200) for c,p in zip(cs,ps)]
   batch={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};batch['raw_adv_std']=(batch['returns']-batch['value']).std()
   decisions=len(batch['reward']);torch.save({k:v.cpu() for k,v in batch.items()},a.output/'calibration-batch.pt')
   atomic(a.output/'calibration-coverage.json',dict(episodes=[e for r in results for e in r['episodes']],decisions=decisions,phase_counts=replay.count_phases(results)))
   calibrate(net,replay,batch,a.output,plan);status='completed_calibration';return
  calibration_path=Path(os.environ['M39B_CALIBRATION']);cal=json.loads(calibration_path.read_text());assert cal['passed'] and cal['initial_sha256']==plan['initial']['sha256'] and cal['data_sha256']==replay.data_sha
  weight=plan['replay']['raw_weight'] if replay.kind=='raw' else cal['action_weight']
  atomic(a.output/'retention.json',dict(kind=replay.kind,weight=weight,calibration=str(calibration_path),calibration_sha256=sha(calibration_path),same_initial_state_sha=replay.state_sha(net)))
  rows,best=dev();save('best.pt');atomic(a.output/'selection.json',dict(iteration=0,score=best,sha256=sha(a.output/'best.pt'),evaluation='dev-0000/summary.json',rule=plan['selection']));iterations=a.iterations
  history=[];rollbacks=0
  status='running'
  for it in range(iterations):
   if time.monotonic()-started>a.max_seconds-240:status='time_budget';break
   t=time.monotonic();w=weights()
   for c in cs:c.send(dict(op='rollout',steps=256,version=it,policy=w))
   results=[receive(c,p,1200) for c,p in zip(cs,ps)];sample_s=time.monotonic()-t
   assert all(r['version']==it for r in results)
   batch={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};batch['raw_adv_std']=(batch['returns']-batch['value']).std();t=time.monotonic();u=update(net,opt,batch,replay=replay,bc_weight=weight)
   iteration+=1;decisions+=len(batch['reward']);updates+=u['optimizer_updates'];stats=[s for r in results for s in r['stats']];active=sum(s['active_steps'] for s in stats)
   row=dict(iteration=iteration,rl_decisions=decisions,optimizer_updates=updates,sampling_seconds=sample_s,optimization_seconds=time.monotonic()-t,decisions_per_second=len(batch['reward'])/sample_s,physical_steps=sum(s['physics_steps'] for s in stats),control_steps=sum(s['control_steps'] for s in stats),prefill_physics_steps=sum(r['prefill_physics_steps'] for r in results),force_rms=(sum(s['active_force_squared'] for s in stats)/max(active,1))**.5,cap_fraction=sum(s['cap_steps'] for s in stats)/max(active,1),reward=float(batch['reward'].mean()),reward_terms={k:float(np.mean([s['terms'][k] for s in stats])) for k in stats[0]['terms']},execution={k:float(np.mean([s['execution'][k] for s in stats])) for k in stats[0]['execution']},episodes=[e for r in results for e in r['episodes']],retention_diagnostics=replay.diagnostics(net),coverage=replay.count_phases(results),optimization=u,elapsed_s=time.monotonic()-started)
   with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   atomic(a.output/'status.json',dict(status=status,**row));save('latest.pt');print('NARROW_UPDATE',iteration,decisions,'force',round(row['force_rms'],3),'rate',round(row['decisions_per_second'],2),'KL',u['epochs'][-1]['joint_gaussian_KL'],flush=True)
   if iteration%plan['training']['evaluate_every']==0:
    rows,score=dev();candidate=save(f'candidate-{iteration:04d}.pt')
    from m39b_metrics import cost
    worst=max(cost(r)/max(baseline_scores[r['label']],.1) for r in rows)
    reversed_rotation=any((r['label']=='pair_left__mean' and r['actual_torso_turn_deg']<=0) or (r['label']=='pair_right__mean' and r['actual_torso_turn_deg']>=0) for r in rows)
    destructive=any(r['boundary'] for r in rows) or worst>1.75 or reversed_rotation
    if score<best and not destructive:
     best=score;shutil.copyfile(candidate,a.output/'best.pt');atomic(a.output/'selection.json',dict(iteration=iteration,score=score,sha256=sha(a.output/'best.pt'),evaluation=f'dev-{iteration:04d}/summary.json',rule=plan['selection']))
    atomic(a.output/f'dev-{iteration:04d}/score.json',dict(score=score,best=best));print('NARROW_DEV',iteration,score,best,flush=True)
    history.append(dict(iteration=iteration,score=score,best=best))
    if destructive:
     prior=torch.load(a.output/'best.pt',map_location='cuda',weights_only=True);net.load_state_dict(prior['state_dict']);opt.load_state_dict(prior['optimizer']);rollbacks+=1
     lr=plan['training']['actor_lr']*(.5**rollbacks);opt.param_groups[0]['lr']=lr;opt.param_groups[0]['nominal_actor_lr']=lr
     atomic(a.output/f'rollback-{iteration:04d}.json',dict(reason='physical boundary, reversed developed rotation, or worst cost >1.75 initial',worst_ratio=worst,reversed_rotation=reversed_rotation,restored_iteration=prior['iteration'],restored_sha256=sha(a.output/'best.pt'),new_actor_lr=lr,sampler_history='real current states retained, new batch under restored distribution'))
     # No asymmetric performance early stop; both arms complete same64 decisions budget.

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
  atomic(a.output/'result.json',dict(status=status,iterations=iteration,decisions=decisions,updates=updates,elapsed_s=time.monotonic()-started,workers=[dict(pid=p.pid,exitcode=p.exitcode) for p in ps],bfm_updated=False,student_updated=False))
if __name__=='__main__':main()
