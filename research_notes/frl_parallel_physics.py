"""Bounded 1/2/4-process MuJoCo sampling with one central frozen inference service.
No optimizer, no new task, no PPO data production. Worker observation tensors remain
on CUDA in this first prototype; MuJoCo stepping is CPU. Unused local BFM copies
remain intact, so this is explicitly not a CPU-only-worker or memory-minimal system.
"""
import argparse,gzip,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent;P=S.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
from frl_inference_rpc import RemoteChannel,validate_request,preserved_row_actions
BASE_SHA='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
POLICY_SHA='112b3da596960e7064ba3a31f55f7c9da976101adcd866bedb15038501bbff1f'

def selected_cases():
 from frl_v2_task import manifests
 groups=manifests()['dev'];return [(groups[0],'main',groups[0]['spec']), (groups[8],'main',groups[8]['spec']),
   (groups[16],'main',groups[16]['spec']),(groups[16],'continue',groups[16]['continued_spec'])]

def prepare(e,m,out):
 import torch
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from frl_v2_task import configure
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');configure()

def tensor_array(value):return value.detach().cpu().contiguous().numpy().copy()

def worker_study(e,m,contract,out):
 import numpy as np,torch
 from multiprocessing.connection import Connection
 from frl_v2_task import Task
 from native_transfer import phase_report
 from dual_track_quality import quality
 prepare(e,m,out);assert contract['model_sha256']==BASE_SHA
 channel=RemoteChannel(WORKER_CONNECTION,POLICY_SHA)
 WORKER_CONNECTION.send(dict(kind='ready',pid=os.getpid(),base_sha=BASE_SHA,policy_sha=POLICY_SHA))
 assert WORKER_CONNECTION.recv()=={'kind':'start'},'No evaluation phase barrier'
 def act(obs,z,mean=True):
  assert mean is True,'This qualification never samples a low-level BFM actor'
  response=channel.call('bfm',dict(obs={k:tensor_array(v) for k,v in obs.items()},z=tensor_array(z)))
  return torch.from_numpy(response['action']).to(device=e.standing.device)
 m.act=act
 task=Task(e,m,contract,56017);rows=[];counts=dict(high_level_decisions=0,control_50Hz=0,physics_200Hz=0,source_physics_200Hz=0,reset_common_physics_200Hz=0)
 for case_index,(group,child,spec) in enumerate(selected_cases()):
  if case_index%ARGS.workers!=ARGS.worker_index:continue
  torch.manual_seed(56017);np.random.seed(56017)
  label=group['parent_id'][:12]+'-'+child;task.reset(spec,record=True,label=label)
  while not task.done:
   x=torch.cat([task.student_features(),task.features()],-1)
   response=channel.call('high',dict(obs=tensor_array(x)))
   raw=torch.from_numpy(response['mean']).to(device=e.standing.device)
   task.step(raw)
  rows.append(dict(parent_id=group['parent_id'],child=child,source_state_sha256=task.prefix_data['audit']['sha256'],boundary=task.boundary,quality=quality(task),**phase_report(task)))
  counts['high_level_decisions']+=len(task.decision_rows);counts['control_50Hz']+=len(task.controls);counts['physics_200Hz']+=len(task.physics)
  counts['source_physics_200Hz']+=len(task.prefix_data['physics']);counts['reset_common_physics_200Hz']+=600
  task.save(out)
 (out/'worker-physical.json').write_text(json.dumps(dict(rows=rows,counts=counts),indent=2))
 WORKER_CONNECTION.send(dict(kind='done',worker=ARGS.worker_index,counts=counts,cases=len(rows)))
 return dict(status='completed',worker=ARGS.worker_index,RL_training_decisions=0,agent_updates=0)

def broker_study(e,m,contract,out):
 import numpy as np,torch
 from multiprocessing import Pipe
 from multiprocessing.connection import wait
 from frl_v2_high import build
 from frl_v2_task import configure
 configure();checkpoint=S/'runs/frl-formal-A-01/candidate-0.pt'
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==POLICY_SHA
 net=build();net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);net.eval().requires_grad_(False)
 children=[];worker_processes=[];connections=[];sequence={};pending={};messages=[];sampling_start=None;startup=time.perf_counter()
 server_stats=dict(bfm_requests=0,high_requests=0,GPU_forward_calls=0,shadow_batch_calls=0,shadow_batch_errors=[],shadow_bfm_max_difference=0.,shadow_high_max_difference=0.,joint_logprob_max_difference=0.,received_group_sizes=[])
 probe_rng=torch.Generator(device='cuda');probe_rng.manual_seed(58083)
 def ready_read(conn,child,timeout):
  if not conn.poll(timeout):raise RuntimeError('Sampling worker bootstrap timeout')
  msg=conn.recv();assert msg.get('kind')=='ready',msg
  assert msg['base_sha']==BASE_SHA and msg['policy_sha']==POLICY_SHA
  assert child.poll() is None,'Worker exited at bootstrap'
  return msg
 try:
  # Sequential bootstrap prevents simultaneous model loading or calibration storms.
  # All workers are genuinely separate complete environments; no qpos-only state transfer.
  for i in range(ARGS.workers):
   parent,child_conn=Pipe(duplex=True);worker_out=out/f'worker-{i}'
   cmd=[sys.executable,str(Path(__file__)),'--role','worker','--workers',str(ARGS.workers),'--worker-index',str(i),'--fd',str(child_conn.fileno()),'--output',str(worker_out)]
   log=(out/f'worker-{i}.log').open('x')
   child=subprocess.Popen(cmd,pass_fds=(child_conn.fileno(),),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   child_conn.close();children.append((child,log));connections.append(parent);sequence[parent]=0
   worker_processes.append(dict(worker=i,pid=child.pid,command=cmd,started=time.time(),output=str(worker_out)))
   (out/'worker-processes.json').write_text(json.dumps(worker_processes,indent=2))
   messages.append(ready_read(parent,child,240))
  startup_s=time.perf_counter()-startup;sampling_start=time.perf_counter()
  for conn in connections:conn.send({'kind':'start'})
  active=list(connections);deadline=time.monotonic()+600
  while active:
   assert time.monotonic()<deadline,'Bounded sampling timeout'
   available=wait(active,timeout=.5)
   if not available:
    for child,_ in children:
     assert child.poll() in (None,0),'Sampling worker failed'
    continue
   # Bound gather latency to 0.5ms; no waiting for a full synchronized batch.
   gather_until=time.monotonic()+.0005
   while len(available)<len(active) and time.monotonic()<gather_until:
    more=wait([c for c in active if c not in available],timeout=max(0,gather_until-time.monotonic()))
    if not more:break
    available.extend(more)
   batch=[]
   for conn in available:
    request=conn.recv()
    if request['kind']=='fatal':raise RuntimeError(request)
    if request['kind']=='done':active.remove(conn);messages.append(request);continue
    sequence[conn]=validate_request(request,sequence[conn],POLICY_SHA);batch.append((conn,request))
   # Do not wait for every worker: reset, support transitions and episode lengths vary.
   for kind in ('high','bfm'):
    entries=[(c,q) for c,q in batch if q['kind']==kind]
    if not entries:continue
    server_stats['received_group_sizes'].append(len(entries))
    if kind=='bfm':
     data=[q['payload'] for _,q in entries];obs=[{k:torch.as_tensor(v,device='cuda') for k,v in d['obs'].items()} for d in data];zs=[torch.as_tensor(d['z'],device='cuda') for d in data]
     # Executed path stays canonical batch-one until genuine full-physics qualification.
     actions=preserved_row_actions(m.act,obs,zs);server_stats['GPU_forward_calls']+=len(actions);server_stats['bfm_requests']+=len(actions)
     if len(entries)>1 and server_stats['shadow_batch_calls']<32:
      try:
       joined={k:torch.cat([o[k] for o in obs],0) for k in obs[0]};candidate=m.act(joined,torch.cat(zs),mean=True)
       server_stats['shadow_bfm_max_difference']=max(server_stats['shadow_bfm_max_difference'],float((candidate-torch.cat(actions)).abs().max()))
      except (ValueError,AssertionError,RuntimeError) as exc:
       server_stats['shadow_batch_errors'].append(dict(batch=len(entries),error=repr(exc),executed=False))
      server_stats['shadow_batch_calls']+=1
     for (conn,request),action in zip(entries,actions):
      conn.send(dict(sequence=request['sequence'],policy_sha=POLICY_SHA,payload=dict(action=tensor_array(action))))
    else:
     xs=[torch.as_tensor(q['payload']['obs'],device='cuda') for _,q in entries]
     ds=[net.distribution(x) for x in xs];means=[d.mean.detach().clone() for d in ds];server_stats['high_requests']+=len(xs);server_stats['GPU_forward_calls']+=len(xs)
     if len(entries)>1:
      dist=net.distribution(torch.cat(xs));catmean=torch.cat(means);difference=float((dist.mean-catmean).abs().max())
      server_stats['shadow_high_max_difference']=max(server_stats['shadow_high_max_difference'],difference)
      # Probability-only diagnostic, never executed and never inserted into PPO data.
      raw=catmean+torch.cat([d.stddev for d in ds])*torch.randn(catmean.shape,device='cuda',generator=probe_rng)
      rowlp=torch.cat([d.log_prob(a).sum(-1) for d,a in zip(ds,raw.split(1))]);batchlp=dist.log_prob(raw).sum(-1)
      delta=float((batchlp-rowlp).abs().max());server_stats['joint_logprob_max_difference']=max(server_stats['joint_logprob_max_difference'],delta)
      assert delta<=3e-4,'Original probability tolerance exceeded; do not adopt batched high-level inference'
     for (conn,request),mean in zip(entries,means):
      conn.send(dict(sequence=request['sequence'],policy_sha=POLICY_SHA,payload=dict(mean=tensor_array(mean))))
  sampling_s=time.perf_counter()-sampling_start
  for meta,(child,_) in zip(worker_processes,children):
   rc=child.wait(timeout=60);meta.update(returncode=rc,elapsed_s=time.time()-meta['started']);assert rc==0,'Worker post-run frozen-weight or physical check failed'
  (out/'worker-processes.json').write_text(json.dumps(worker_processes,indent=2))
  counts={k:sum(msg['counts'][k] for msg in messages if msg.get('kind')=='done') for k in next(msg['counts'] for msg in messages if msg.get('kind')=='done')}
  row=dict(status='completed',workers=ARGS.workers,architecture='CPU MuJoCo workers retain CUDA observation tensors and unused frozen modules; one central inference service',execution='canonical batch-one; batching shadow only',startup_s=startup_s,sampling_s=sampling_s,counts=counts,
   high_level_decisions_per_s=counts['high_level_decisions']/sampling_s,control_50Hz_transitions_per_s=counts['control_50Hz']/sampling_s,server=server_stats,RL_training_decisions=0,agent_updates=0,policy_sha=POLICY_SHA,base_sha=BASE_SHA)
  (out/'parallel-result.json').write_text(json.dumps(row,indent=2));return row
 finally:
  for child,_ in children:
   if child.poll() is None:child.terminate()
  for child,log in children:
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   log.close()
  for conn in connections:conn.close()

def main():
 global ARGS,WORKER_CONNECTION
 ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['broker','worker'],required=True);ap.add_argument('--workers',type=int,choices=[1,2,4],required=True);ap.add_argument('--worker-index',type=int,default=0);ap.add_argument('--fd',type=int);ap.add_argument('--output',type=Path,required=True);ARGS=ap.parse_args()
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 ARGS.output.mkdir(parents=True,exist_ok=False)
 (ARGS.output/'launch.json').write_text(json.dumps(dict(runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),pid=os.getpid(),started=time.time(),role=ARGS.role,workers=ARGS.workers,worker_index=ARGS.worker_index,base_sha=BASE_SHA,policy_sha=POLICY_SHA,training=False),indent=2))
 if ARGS.role=='worker':
  from multiprocessing.connection import Connection
  WORKER_CONNECTION=Connection(ARGS.fd)
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.environ.setdefault('TMPDIR',str(S/'tmp'));os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(S/'torch-cache'))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=worker_study if ARGS.role=='worker' else broker_study
 try:m10_physics.main()
 except BaseException as exc:
  if ARGS.role=='worker':
   try:WORKER_CONNECTION.send(dict(kind='fatal',error=repr(exc),traceback=traceback.format_exc()))
   except (BrokenPipeError,EOFError,OSError):pass
  raise
if __name__=='__main__':main()
