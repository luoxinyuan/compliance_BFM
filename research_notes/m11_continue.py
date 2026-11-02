"""M11 matched acquisition using the existing Task, worker and PPO implementation."""
import m11_ipc
import argparse,json,os,time,signal,subprocess,multiprocessing as mp
from pathlib import Path
from m11_selection import Selection,atomic,sha,continuation_rank,exploration_scale
ROOT=Path(__file__).resolve().parents[1]
PLAN=Path(__file__).with_name('m11_continuation_protocol.json')
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-continuation-20261102')

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True,choices=['zero','fixed','annealed','final-prior','final-fixed','final-annealed']);p.add_argument('--gpu',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--iterations',type=int,default=160);a=p.parse_args();assert not a.output.exists();a.output.mkdir()
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 import torch,numpy as np
 from m11_worker import launch
 from m11_policy import Expert,update
 from humanoidverse.force_compliance.parallel import receive,merge_rollouts
 plan=json.loads(PLAN.read_text());cfg=plan['continuation'];torch.set_num_threads(1);torch.set_grad_enabled(False)
 source=Path(cfg['source_checkpoint'])
 if a.job.startswith('final-') and a.job!='final-prior':
  arm=a.job.split('-')[1];source=RUN/(arm+'-attempt-01/outputs/job/best.pt')
 if a.job in ['zero','fixed','annealed','final-prior']:assert sha(source)==cfg['source_sha256']
 ck=torch.load(source,map_location='cuda',weights_only=True);assert ck['prior_sha']==plan['prior_sha256']
 net=Expert(ck['input_dim']).cuda();net.load_state_dict(ck['state_dict']);opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=3e-5),dict(params=[net.log_std],lr=1e-5),dict(params=net.critic.parameters(),lr=3e-4)]);opt.load_state_dict(ck['optimizer'])
 original_std=net.log_std.detach().clone();net.log_std.requires_grad_(False)
 # Keep the source optimizer groups/moments intact for provenance; freeze only std
 # identically in both arms. No stale std momentum can change sampled likelihoods.
 opt.param_groups[1]['lr']=0.
 torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
 commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 atomic(a.output/'launch.json',dict(job=a.job,pid=os.getpid(),commit=commit,source=str(source),source_sha256=sha(source),source_iteration=ck['iteration'],source_decisions=ck['decisions'],source_optimizer_updates=ck['optimizer_updates'],source_historical_best_score=ck['best_score'],plan_sha256=sha(PLAN),std_optimizer='frozen lr0; original moments preserved; no clamp during updates'))
 cs=[];ps=[];ctx=mp.get_context('spawn');start=time.monotonic();status='initializing';completed=decisions=updates=0;selection=Selection(a.output,dict(plan_sha256=sha(PLAN),rule=cfg['selection']));baseline=None
 def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
 signal.signal(signal.SIGTERM,stop)
 def weights():return {k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
 def save(name,evaluated_rank=None,evaluation=None):
  path=a.output/name;tmp=path.with_suffix('.tmp');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),input_dim=ck['input_dim'],iteration=ck['iteration']+completed,decisions=ck['decisions']+decisions,optimizer_updates=ck['optimizer_updates']+updates,continuation_batches=completed,continuation_decisions=decisions,best_score=evaluated_rank[1] if evaluated_rank is not None else (selection.record['rank'][1] if selection.record else None),evaluated_rank=evaluated_rank,evaluation_path=str(evaluation) if evaluation else None,evaluation_sha256=sha(evaluation) if evaluation else None,selection_scope=selection.scope,prior_sha=plan['prior_sha256'],source_sha256=sha(source),runtime_commit=commit,plan_sha=sha(PLAN),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),external_std=True),tmp);tmp.replace(path);return path
 def evaluate(schedule,name):
  folder=a.output/name;folder.mkdir();w=weights()
  for i,c in enumerate(cs):c.send(dict(op='evaluate',policy=w,schedule=schedule[i::3],output=str(folder)))
  rows=[]
  for c,proc in zip(cs,ps):rows+=receive(c,proc,timeout=1800)['summaries']
  for row in rows:row['case']=row['label'].split('__')[0]
  atomic(folder/'summary.json',rows);return rows,folder/'summary.json'
 def schedule(cases,mode='learned',samples=False,seeds=(202,)):
  out=[]
  for name,spec in cases.items():
   if not samples:out.append((name+'__mean',spec,mode));continue
   for seed in seeds:
    s=dict(spec,diagnostic_sampling=True,diagnostic_std_scale=1.,diagnostic_seed=seed,diagnostic_action_probes=False,diagnostic_read_audit=False)
    out.append((name+'__sample'+str(seed),s,mode))
  return out
 def development():
  rows,ev=evaluate(schedule(plan['dev'])+schedule(plan['dev'],samples=True),f'dev-{completed:04d}')
  mean=[r for r in rows if r['label'].endswith('__mean')];rank,details=continuation_rank(mean,baseline);candidate=save(f'candidate-{completed:04d}.pt',rank,ev)
  audit=torch.load(candidate,map_location='cpu',weights_only=True);assert audit['evaluated_rank']==rank and audit['best_score']==rank[1] and audit['evaluation_sha256']==sha(ev)
  selected=selection.consider(candidate,rank,ev,dict(batches=completed,added_decisions=decisions,source_sha256=sha(source),std_mean=float(net.log_std.exp().mean()),**details));atomic(ev.parent/'score.json',dict(rank=rank,selected=selected,checkpoint_sha256=sha(candidate),**details));print('M11_DEV',a.job,completed,rank,'selected',selected,flush=True)
 try:
  for i in range(3):
   parent,child=ctx.Pipe();proc=ctx.Process(target=launch,args=(child,a.gpu,i,a.output/f'worker-{i}'));proc.start();child.close();ps.append(proc);cs.append(parent)
  ready=[receive(c,proc,timeout=900) for c,proc in zip(cs,ps)];assert all(x['input_dim']==ck['input_dim'] for x in ready);atomic(a.output/'workers.json',ready)
  if a.job=='zero':
   evaluate(json.loads(Path(__file__).with_name('m11_zero_mean_plan.json').read_text()),'evaluation');status='completed'
  elif a.job.startswith('final-'):
   sch=schedule(plan['final'],mode='prior' if a.job=='final-prior' else 'learned')
   if a.job!='final-prior':sch+=schedule(plan['final'],samples=True,seeds=cfg['final_sample_seeds'])
   evaluate(sch,'evaluation');status='completed'
  else:
   rows,ev=evaluate(schedule(plan['dev'],mode='prior'),'baseline');baseline={r['case']:r for r in rows};development();status='running'
   for batch in range(a.iterations):
    if time.monotonic()-start>a.max_seconds:status='time_budget';break
    previous_std=net.log_std.exp().detach().clone();factor=exploration_scale(a.job,batch);net.log_std.copy_(original_std+np.log(factor));sampled_std=net.log_std.exp().detach().clone();schedule_KL=float((torch.log(sampled_std/previous_std)+(previous_std/sampled_std).square()/2-.5).sum());w=weights();version=ck['iteration']+batch;t=time.monotonic()
    for c in cs:c.send(dict(op='rollout',steps=256,version=version,policy=w))
    results=[receive(c,proc,timeout=1200) for c,proc in zip(cs,ps)];sampling=time.monotonic()-t
    assert all(x['version']==version and np.array_equal(x['executed_std'],sampled_std.cpu().numpy()) for x in results)
    data={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};data['raw_adv_std']=(data['returns']-data['value']).std();t=time.monotonic();u=update(net,opt,data);assert torch.equal(sampled_std,net.log_std.exp())
    completed+=1;decisions+=len(data['reward']);updates+=u['optimizer_updates'];stats=[s for x in results for s in x['stats']];active=sum(s['active_steps'] for s in stats)
    row=dict(arm=a.job,batch=completed,added_decisions=decisions,optimizer_updates=updates,sampling_seconds=sampling,optimization_seconds=time.monotonic()-t,decisions_per_second=len(data['reward'])/sampling,physics_steps=sum(s['physics_steps'] for s in stats),control_steps=sum(s['control_steps'] for s in stats),prefill_physics_steps=sum(x['prefill_physics_steps'] for x in results),force_rms=(sum(s['active_force_squared'] for s in stats)/max(active,1))**.5,cap_fraction=sum(s['cap_steps'] for s in stats)/max(active,1),reward=float(data['reward'].mean()),reward_terms={k:float(np.mean([s['terms'][k] for s in stats])) for k in stats[0]['terms']},std_factor=factor,schedule_joint_KL=schedule_KL,actual_std_mean=float(sampled_std.mean()),actual_std_min=float(sampled_std.min()),actual_std_max=float(sampled_std.max()),execution={k:float(np.mean([s['execution'][k] for s in stats])) for k in stats[0]['execution']},execution_max={k:max(s['execution_max'][k] for s in stats) for k in stats[0]['execution_max']},completed_episodes=[e for x in results for e in x['episodes']],optimization=u,elapsed_s=time.monotonic()-start)
    with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    atomic(a.output/'status.json',dict(status=status,**row));print('M11_TRAIN',a.job,completed,decisions,'std',round(row['actual_std_mean'],5),'force',round(row['force_rms'],3),'KL',round(u['epochs'][-1]['joint_gaussian_KL'],5),'rate',round(row['decisions_per_second'],2),flush=True)
    if completed%5==0:save('latest.pt')
    if completed%plan['evaluate_every']==0:development()
   else:status='completed'
   save('last.pt')
 except KeyboardInterrupt:
  status='interrupted';save('interrupted.pt')
 except BaseException:
  import traceback
  status='failed';(a.output/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  for c in cs:
   try:c.send(dict(op='close'))
   except (EOFError,BrokenPipeError,OSError):pass
  for proc in ps:
   proc.join(timeout=10)
   if proc.is_alive():proc.terminate();proc.join(timeout=10)
  result=dict(status=status,job=a.job,added_decisions=decisions,completed_batches=completed,added_optimizer_updates=updates,source_sha256=sha(source),elapsed_s=time.monotonic()-start,workers=[dict(pid=p.pid,exitcode=p.exitcode) for p in ps],bfm_updated=False,student_updated=False,selection=selection.record)
  atomic(a.output/'status.json',result);atomic(a.output/'result.json',result)
if __name__=='__main__':main()
