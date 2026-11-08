"""Original MuJoCo/BFM workers; actual raw Gaussian commands, no online expert override."""
import os,json,time,traceback,math
from pathlib import Path
CONNECTION=None;WORKER_ID=0

def launch(connection,gpu,index,output):
 global CONNECTION,WORKER_ID
 CONNECTION,WORKER_ID=connection,index
 os.environ.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_GL='disable',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',BFM_RECOVERY_OUTPUT=str(output),PYTHONDONTWRITEBYTECODE='1',CLEAN_KIND='direct')
 output=Path(output);output.mkdir();os.chdir(output);f=open(output/'worker.log','a',buffering=1);os.dup2(f.fileno(),1);os.dup2(f.fileno(),2)
 try:
  import m10_stage,m10_physics
  m10_stage.run_study=loop;m10_physics.main()
 except BaseException:
  err=traceback.format_exc();(output/'failure.txt').write_text(err)
  try:connection.send(dict(kind='error',error=err,worker_id=index))
  except (BrokenPipeError,EOFError):pass
 finally:connection.close()

def loop(env,model,contract,out):
 import numpy as np,torch
 from p4000_env import Task,install
 from clean_policy import Expert,direct_command
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from native_transfer import phase_report
 from locomotion_rollout import spherical
 install();torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'runtime-speed.json').write_text(json.dumps(speed(env)));folder=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/native-transfer-20261107/compatibility-prefix');(out/'backend.json').write_text(json.dumps(backend(env,model,folder)))
 seed=int(os.environ['P4000_SEED']);torch.manual_seed(seed*997+WORKER_ID);task=Task(env,model,contract,seed*997+WORKER_ID);obs=task.reset(record=True,label=f'train-w{WORKER_ID}-0');net=Expert(1088,'direct').to(obs.device).eval();episodes=0
 CONNECTION.send(dict(kind='ready',pid=os.getpid(),input_dim=obs.shape[-1],worker=WORKER_ID,standing=env.standing[0].cpu().tolist(),contract=contract,initial_source_physics_steps=task.source_physics_total))
 while True:
  cmd=CONNECTION.recv()
  if cmd['op']=='close':break
  net.load_state_dict({k:torch.from_numpy(v).to(obs.device) for k,v in cmd['policy'].items()})
  if cmd['op']=='evaluate':
   summaries=[]
   for label,spec,mode in cmd['schedule']:
    obs=task.reset(spec,record=True,label=label);g=torch.Generator(device=obs.device).manual_seed(int(spec.get('sampling_seed',seed*101+17)));rawstats=[]
    while not task.done:
     dist=net.distribution(obs);mean=dist.mean;raw=mean
     if mode=='sample':raw=mean+dist.stddev*torch.randn(mean.shape,device=obs.device,generator=g)
     zm=direct_command(mean,env.standing);zs=direct_command(raw,env.standing);rawstats.append(dict(raw_mean_norm=float(mean.norm()),raw_sample_norm=float(raw.norm()),z_sample_mean_angle_deg=float(torch.rad2deg(torch.acos(((zm*zs).sum()/256).clamp(-1,1))))))
     obs,_,_,_=task.step(raw)
    dest=Path(cmd['output']);dest.mkdir(exist_ok=True);row=task.save(dest);row.update(phase_report(task),mode=mode,command_statistics={k:dict(mean=float(np.mean([v[k] for v in rawstats])),max=max(v[k] for v in rawstats)) for k in rawstats[0]});summaries.append(row)
   obs=task.reset(record=True,label=f'train-w{WORKER_ID}-{episodes}')
   CONNECTION.send(dict(kind='evaluate',summaries=summaries));continue
  assert cmd['op']=='rollout';g=torch.Generator(device=obs.device).manual_seed(seed*1000003+WORKER_ID*1001+cmd['version']*100003)
  batch={k:[] for k in ['obs','next_obs','action','logp','value','next_value','reward','discount','done']};stats=[];eps=[];start=time.monotonic();prefill=task.total_prefill;source=task.source_physics_total
  for j in range(cmd['steps']):
   if task.done:obs=task.reset(record=True,label=f'train-w{WORKER_ID}-{episodes}')
   dist=net.distribution(obs);mean=dist.mean;raw=mean+dist.stddev*torch.randn(mean.shape,device=obs.device,generator=g);lp=dist.log_prob(raw).sum(-1);v=net.value(obs);prev=task.last_command.clone();zm=direct_command(mean,env.standing);zs=direct_command(raw,env.standing);nxt,re,done,st=task.step(raw);nv=net.value(nxt)
   for k,val in dict(obs=obs[0],next_obs=nxt[0],action=raw[0],logp=lp[0],value=v[0],next_value=nv[0]).items():batch[k].append(val.cpu().numpy())
   batch['reward'].append(re);batch['discount'].append(.99**(st['control_steps']/5));batch['done'].append(done)
   st.update(time=float(env.tether.time),raw_mean_norm=float(mean.norm()),raw_sample_norm=float(raw.norm()),raw_noise_norm=float((raw-mean).norm()),z_sample_mean_angle_deg=float(torch.rad2deg(torch.acos(((zm*zs).sum()/256).clamp(-1,1)))),executed_std=float(dist.stddev.mean()))
   if j%8==0:
    diffs=[];maxdiff=0.
    for k,c in enumerate(task.controls[-st['control_steps']:]):
     ob={key:torch.tensor(c['bfm_obs'][key],device=obs.device,dtype=torch.float32) for key in env.keys};q=task.qof(model.act(ob,spherical(prev,zm,(k+1)/5),mean=True));actual=torch.tensor(c['q_PD'],device=obs.device);d=torch.rad2deg(q-actual);diffs.append(float(d.square().mean().sqrt()));maxdiff=max(maxdiff,float(d.abs().max()))
    st['same_state_5substep_mean_sample_qpd']=dict(rms_deg=diffs,max_single_joint_deg=maxdiff,scope='fixed actual BFM observations, same actual last-z; not a physical alternative rollout')
   stats.append(st);obs=nxt
   if done:
    e=task.summary();eps.append(e)
    if task.boundary or episodes%50==0:task.save(out)
    episodes+=1
  a={k:np.asarray(v,dtype=bool if k=='done' else np.float32) for k,v in batch.items()};np.savez_compressed(out/f'rollout-{cmd["version"]:04d}.npz',**a);(out/f'rollout-{cmd["version"]:04d}-stats.json').write_text(json.dumps(stats))
  CONNECTION.send(dict(kind='rollout',version=cmd['version'],batch=a,stats=stats,episodes=eps,seconds=time.monotonic()-start,prefill_physics_steps=task.total_prefill-prefill,source_physics_steps=task.source_physics_total-source))
 return dict(status='completed',episodes=episodes,physical_steps=task.total_steps,source_physics_steps=task.source_physics_total,student_optimizer_updates=0)
