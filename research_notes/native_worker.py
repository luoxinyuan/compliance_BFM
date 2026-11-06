"""Native-only source prefixes; PPO samples exact whitened full-rank Gaussians."""
import os,json,time,traceback
from pathlib import Path
CONNECTION=None;WORKER_ID=0

def launch(connection,gpu,index,output):
 global CONNECTION,WORKER_ID
 CONNECTION=connection;WORKER_ID=index;os.environ.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_GL='disable',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',BFM_RECOVERY_OUTPUT=str(output),PYTHONDONTWRITEBYTECODE='1',CLEAN_KIND='direct')
 output=Path(output);output.mkdir();os.chdir(output);log=open(output/'worker.log','a',buffering=1);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
 try:
  import m10_stage,m10_physics
  m10_stage.run_study=loop;m10_physics.main()
 except BaseException:
  err=traceback.format_exc();(output/'failure.txt').write_text(err)
  try:connection.send(dict(kind='error',error=err,worker_id=index))
  except (BrokenPipeError,EOFError):pass
 finally:connection.close()

def loop(env,model,contract,out):
 import torch,numpy as np
 from native_env import Task,install
 from native_policy import Expert
 from m11_runtime_speed import install as speed
 install();(out/'runtime-speed.json').write_text(json.dumps(speed(env)));torch.set_num_threads(1);seed=int(os.environ['NATIVE_SEED']);torch.manual_seed(seed*997+WORKER_ID);task=Task(env,model,contract,seed*997+WORKER_ID);obs=task.reset(record=True,label=f'train-w{WORKER_ID}-0');net=Expert(obs.shape[-1]).to(obs.device).eval();episodes=0
 CONNECTION.send(dict(kind='ready',pid=os.getpid(),input_dim=obs.shape[-1],standing=env.standing[0].cpu().tolist(),worker=WORKER_ID,contract=contract,initial_source_physics_steps=task.source_physics_total))
 while True:
  cmd=CONNECTION.recv()
  if cmd['op']=='close':break
  net.load_state_dict({k:torch.from_numpy(v).to(obs.device) for k,v in cmd['policy'].items()})
  if cmd['op']=='evaluate':
   summaries=[]
   for label,spec,mode in cmd['schedule']:
    obs=task.reset(spec,record=True,label=label)
    while not task.done:
     raw=net.raw_mean(obs)
     if mode=='standing':raw=env.standing/16
     elif mode.startswith('native:'):raw=task.native(mode.split(':',1)[1],float(env.tether.time))/16
     obs,_,_,_=task.step(raw)
    dest=Path(cmd['output']);dest.mkdir(exist_ok=True);summaries.append(task.save(dest))
   # Development reset is separate physical cost, never credited as PPO samples.
   obs=task.reset(record=True,label=f'train-w{WORKER_ID}-{episodes}')
   CONNECTION.send(dict(kind='evaluate',summaries=summaries));continue
  assert cmd['op']=='rollout';generator=torch.Generator(device=obs.device).manual_seed(seed*1000003+WORKER_ID*1001+cmd['version']*100003)
  batch={k:[] for k in ['obs','action','logp','value','next_value','reward','discount','done']};stats=[];eps=[];start=time.monotonic();prefill=task.total_prefill;source=task.source_physics_total
  with torch.no_grad():
   for _ in range(cmd['steps']):
    if task.done:obs=task.reset(record=True,label=f'train-w{WORKER_ID}-{episodes}')
    d=net.distribution(obs);u=d.mean+d.stddev*torch.randn(d.mean.shape,device=obs.device,generator=generator);raw=net.to_raw(u);lp=d.log_prob(u).sum(-1);value=net.value(obs);nxt,reward,done,st=task.step(raw);nv=net.value(nxt)
    # u is the actual sampled probability-space command; raw=L*u executed once.
    for k,v in dict(obs=obs[0],action=u[0],logp=lp[0],value=value[0],next_value=nv[0]).items():batch[k].append(v.detach().cpu().numpy())
    batch['reward'].append(reward);batch['discount'].append(.99**(st['control_steps']/5));batch['done'].append(done)
    st.update(time=float(env.tether.time),raw_sample_norm=float(raw.norm()),raw_mean_norm=float(net.raw_mean(obs).norm()),probability_u_norm=float(u.norm()),raw_execution_mapping_max=float((raw-net.to_raw(u)).abs().max()))
    stats.append(st);obs=nxt
    if done:
     eps.append(task.summary())
     if task.boundary or episodes%40==0:task.save(out)
     episodes+=1
  a={k:np.asarray(v,dtype=bool if k=='done' else np.float32) for k,v in batch.items()};np.savez_compressed(out/f'rollout-{cmd["version"]:04d}.npz',**a);(out/f'rollout-{cmd["version"]:04d}-stats.json').write_text(json.dumps(stats))
  CONNECTION.send(dict(kind='rollout',version=cmd['version'],batch=a,stats=stats,episodes=eps,seconds=time.monotonic()-start,prefill_physics_steps=task.total_prefill-prefill,source_physics_steps=task.source_physics_total-source))
 return dict(status='completed',episodes=episodes,physical_steps=task.total_steps,source_physics_steps=task.source_physics_total,old_actor_loaded=False)
