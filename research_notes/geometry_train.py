"""Independent privileged Expert acquisition; native motion prior, finite stages."""
import m11_ipc
import argparse,json,os,time,signal,multiprocessing as mp,subprocess,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['B1','B2'],required=True);p.add_argument('--kind',choices=['structured','direct'],required=True);p.add_argument('--stage',choices=['prepare','train','final'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--iterations',type=int,default=120);p.add_argument('--seed',type=int,default=101);p.add_argument('--execution-gain',type=float,default=1.0);p.add_argument('--resume',type=Path);p.add_argument('--domain',default='neighborhood',choices=['single','neighborhood']);p.add_argument('--max-seconds',type=float,default=28000);a=p.parse_args()
 assert 0<a.execution_gain<=16
 assert not a.output.exists();a.output.mkdir();os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CLEAN_KIND=a.kind,NARROW_DOMAIN=a.domain,NARROW_SEED=str(a.seed),NARROW_EXECUTION_GAIN=str(a.execution_gain))
 import torch,numpy as np
 from humanoidverse.force_compliance.parallel import receive,merge_rollouts
 from transfer_policy import Expert,update
 from geometry_worker import launch
 plan=json.loads(Path(__file__).with_name('geometry_protocol.json').read_text());plan['dev']={k:v for k,v in plan['dev'].items() if k in plan['train_'+os.environ['GEOMETRY_ARM']]+['no_contact']};torch.set_num_threads(1);torch.manual_seed(a.seed);torch.set_grad_enabled(False)
 ctx=mp.get_context('spawn');ps=[];cs=[];net=opt=None;iteration=decisions=updates=0;best=float('inf');status='initializing';started=time.monotonic();commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 atomic(a.output/'launch.json',dict(pid=os.getpid(),commit=commit,plan_sha256=sha(Path(__file__).with_name('geometry_protocol.json')),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},started=time.time()))
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
  values={r['label']:r.get('sum_force_norm_rms',r['force_rms'])+3*r['error_cm']/100+r['cap_fraction']+.04*(r.get('torso_heading_rms_deg') if r['condition'].get('geometry')=='torso_pair' else r.get('moving_heading_rms_deg') or 0)+20*bool(r['boundary']) for r in rows if r['label'].split('__')[0] in plan['dev']}
  if not baseline_scores:baseline_scores.update(values)
  relative=[v/max(baseline_scores[k],.1) for k,v in values.items()]
  return max(relative)+sum(relative)/len(relative)
 def dev():
  schedule=[(k+'__mean',v,'learned') for k,v in plan['dev'].items()]
  if iteration%10==0:schedule +=[(k+'__mean',v,'learned') for k,v in plan['regression'].items()]
  rows=evaluate(schedule,f'dev-{iteration:04d}');return rows,select(rows)
 try:
  for i in range(3):
   parent,child=ctx.Pipe();proc=ctx.Process(target=launch,args=(child,a.gpu,i,a.output/f'worker-{i}'));proc.start();child.close();ps.append(proc);cs.append(parent)
  ready=[receive(c,p,900) for c,p in zip(cs,ps)];n=ready[0]['input_dim'];assert all(r['input_dim']==n for r in ready);atomic(a.output/'workers.json',ready)
  net=Expert(n).cuda();net.log_std.requires_grad_(False)
  if a.kind=='direct':net.actor[-1].bias.copy_(torch.tensor(ready[0]['standing'],device='cuda')/16)
  opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=plan['training']['actor_lr']),dict(params=net.critic.parameters(),lr=3e-4)])
  if a.resume:
   ck=torch.load(a.resume,map_location='cuda',weights_only=True);assert ck['kind']==a.kind;assert sha(a.resume)==plan['teachers']['initial']['sha256'] if a.stage=='train' else True;assert ck['source_student_online'] is False;
   from geometry_policy import expand
   expand(net,ck)
   # Only an explicitly recorded narrow-domain continuation inherits its optimizer.
   # Fresh optimizer for new heading task; source weights remain fixed identity.
   atomic(a.output/'resume.json',dict(path=str(a.resume),sha256=sha(a.resume),source_iteration=ck.get('iteration',0),source_seed=ck.get('seed'),optimizer_loaded=False))
  if a.stage=='prepare':
   save('uninitialized.pt')
   rows=evaluate([('initial__mean',plan['single'],'learned'),('teacher__data',plan['single'],'m10_expert'),('standing',plan['single'],'standing')],'qualification')
   import gzip
   with gzip.open(a.output/'qualification/teacher__data.json.gz','rt') as f:data=json.load(f)
   samples=data['bootstrap_rows'];assert samples
   x=torch.tensor([r['obs'] for r in samples],device='cuda');y=torch.tensor([r['target'] for r in samples],device='cuda')
   atomic(a.output/'expression.json',dict(teacher_complete=data['boundary'] is None,max_tangent_norm=max(r['teacher']['tangent_norm'] for r in samples),max_teacher_angle=max(r['teacher']['angle_deg'] for r in samples),samples=len(samples),scope='algebraic own-state teacher coverage; physical learned execution follows'))
   if data['boundary'] is not None:raise RuntimeError('Teacher source failed on fixed task')
   if a.kind=='structured' and max(r['teacher']['tangent_norm'] for r in samples)>16.0001:raise RuntimeError('Teacher label outside declared structured tangent domain')
   net.center.copy_(x.mean(0));net.scale.copy_(x.std(0,unbiased=False).clamp_min(.1))
   warm=torch.optim.Adam(net.actor.parameters(),lr=3e-4);curve=[]
   with torch.enable_grad():
    for step in range(plan['training']['warmstart_steps']):
     ix=torch.randint(len(x),(128,),device='cuda');pred=net.distribution(x[ix]).mean
     if a.kind=='structured':loss=(pred[:,:2]-y[ix,:2]).square().mean()+(pred[:,2:]-y[ix,2:]).square().mean()*16
     else:loss=(torch.nn.functional.normalize(pred,dim=-1)-y[ix]).square().mean()*16
     warm.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1);warm.step()
     if step%100==0:curve.append(dict(step=step,loss=float(loss.detach())))
   atomic(a.output/'warmstart.json',dict(updates=plan['training']['warmstart_steps'],samples=len(x),curve=curve,data='qualification/teacher__data.json.gz',data_sha=sha(a.output/'qualification/teacher__data.json.gz'),future_partner_input=False))
   save('initial.pt');rows,best=dev();save('best.pt');save('last.pt')
   atomic(a.output/'selection.json',dict(iteration=0,score=best,sha256=sha(a.output/'best.pt'),evaluation='dev-0000/summary.json',rule=plan['selection']))
   status='completed';return
  assert a.iterations<=40
  assert a.resume is not None, 'Initialize from separately verified preparation; not random latent exploration'
  save('initial.pt')
  if a.stage=='final':
   evaluate([(k+'__'+m,v,m) for k,v in plan['final'].items() for m in ['standing','m10_expert','learned']], 'evaluation');status='completed';return
  rows,best=dev();save('best.pt');atomic(a.output/'selection.json',dict(iteration=0,score=best,sha256=sha(a.output/'best.pt'),evaluation='dev-0000/summary.json',rule=plan['selection']));iterations=a.iterations
  # Freeze effective initial own-trajectory raw commands for the B2 BC replay.
  import gzip
  rx=[]
  for f in sorted((a.output/'dev-0000').glob('*.json.gz')):
   with gzip.open(f,'rt') as stream: ep=json.load(stream)
   rx.extend(r['obs'] for r in ep['decisions'])
  rx=torch.tensor(rx,device='cuda');ry=net.distribution(rx).mean.detach();replay=(rx,ry)
  atomic(a.output/'replay.json',dict(samples=len(rx),source='dev-0000 initial effective trajectories',weight=plan['bc_weight'] if a.arm=='B2' else 0))
  status='running'
  for it in range(iterations):
   if time.monotonic()-started>a.max_seconds:status='time_budget';break
   t=time.monotonic();w=weights()
   for c in cs:c.send(dict(op='rollout',steps=256,version=it,policy=w))
   results=[receive(c,p,1200) for c,p in zip(cs,ps)];sample_s=time.monotonic()-t
   assert all(r['version']==it for r in results)
   batch={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};batch['raw_adv_std']=(batch['returns']-batch['value']).std();t=time.monotonic();u=update(net,opt,batch,replay=replay,bc_weight=plan['bc_weight'] if a.arm=='B2' else 0.)
   iteration+=1;decisions+=len(batch['reward']);updates+=u['optimizer_updates'];stats=[s for r in results for s in r['stats']];active=sum(s['active_steps'] for s in stats)
   row=dict(iteration=iteration,rl_decisions=decisions,optimizer_updates=updates,sampling_seconds=sample_s,optimization_seconds=time.monotonic()-t,decisions_per_second=len(batch['reward'])/sample_s,physical_steps=sum(s['physics_steps'] for s in stats),control_steps=sum(s['control_steps'] for s in stats),prefill_physics_steps=sum(r['prefill_physics_steps'] for r in results),force_rms=(sum(s['active_force_squared'] for s in stats)/max(active,1))**.5,cap_fraction=sum(s['cap_steps'] for s in stats)/max(active,1),reward=float(batch['reward'].mean()),reward_terms={k:float(np.mean([s['terms'][k] for s in stats])) for k in stats[0]['terms']},execution={k:float(np.mean([s['execution'][k] for s in stats])) for k in stats[0]['execution']},episodes=[e for r in results for e in r['episodes']],optimization=u,elapsed_s=time.monotonic()-started)
   with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   atomic(a.output/'status.json',dict(status=status,**row));save('latest.pt');print('NARROW_UPDATE',iteration,decisions,'force',round(row['force_rms'],3),'rate',round(row['decisions_per_second'],2),'KL',u['epochs'][-1]['joint_gaussian_KL'],flush=True)
   if iteration%plan['evaluate_every']==0:
    rows,score=dev();candidate=save(f'candidate-{iteration:04d}.pt')
    if score<best:
     best=score;shutil.copyfile(candidate,a.output/'best.pt');atomic(a.output/'selection.json',dict(iteration=iteration,score=score,sha256=sha(a.output/'best.pt'),evaluation=f'dev-{iteration:04d}/summary.json',rule=plan['selection']))
    atomic(a.output/f'dev-{iteration:04d}/score.json',dict(score=score,best=best));print('NARROW_DEV',iteration,score,best,flush=True)
    if iteration==40 and a.iterations>40:
     chosen=json.loads((a.output/'selection.json').read_text());ratios=[(r.get('sum_force_norm_rms',r['force_rms'])+3*r['error_cm']/100+r['cap_fraction']+.04*(r.get('torso_heading_rms_deg') if r['condition'].get('geometry')=='torso_pair' else r.get('moving_heading_rms_deg') or 0)+20*bool(r['boundary']))/baseline_scores[r['label']] for r in rows if r['label'] in baseline_scores]
     extend=chosen['iteration']>=30 and chosen['score']<=1.96 and max(ratios)<=1.15
     atomic(a.output/'extension-decision.json',dict(extend=extend,chosen=chosen,current_ratios=ratios,rule=plan['teacher_stop_training']['extension']))
     if not extend:status='completed_predefined_no_extension';break
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
