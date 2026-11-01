"""One original process-global MuJoCo adapter per spawned worker, as force_compliance.parallel."""
import os,json,time,traceback
from pathlib import Path
CONNECTION=None
WORKER_ID=0

def launch(connection,gpu,index,output):
    global CONNECTION,WORKER_ID
    CONNECTION=connection;WORKER_ID=index
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_GL='disable',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',BFM_RECOVERY_OUTPUT=str(output),PYTHONDONTWRITEBYTECODE='1')
    output=Path(output);output.mkdir();os.chdir(output) # per-worker short IPC paths used by original loader
    log=open(output/'worker.log','a',buffering=1);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    try:
        import m10_stage,m10_physics
        m10_stage.run_study=loop
        m10_physics.main()
    except BaseException:
        err=traceback.format_exc();(output/'failure.txt').write_text(err)
        try:connection.send(dict(kind='error',error=err,worker_id=index))
        except (BrokenPipeError,EOFError):pass
    finally:connection.close()

def loop(env,model,contract,out):
    import numpy as np,torch
    from m11_env import Task
    from m11_policy import Expert
    torch.set_num_threads(1);torch.manual_seed(1001+WORKER_ID);np.random.seed(1001+WORKER_ID)
    task=Task(env,model,contract,1001+WORKER_ID);obs=task.reset(record=True,label=f'train-w{WORKER_ID}-0');net=Expert(obs.shape[-1]).to(obs.device).eval();episodes=0
    CONNECTION.send(dict(kind='ready',pid=os.getpid(),input_dim=obs.shape[-1],worker=WORKER_ID))
    while True:
        cmd=CONNECTION.recv()
        if cmd['op']=='close':break
        net.load_state_dict({k:torch.from_numpy(v).to(obs.device) for k,v in cmd['policy'].items()})
        if cmd['op']=='evaluate':
            summaries=[]
            for label,spec,mode in cmd['schedule']:
                obs=task.reset(spec,record=True,label=label)
                while not task.done:
                    raw=net.distribution(obs).mean
                    obs,_,_,_=task.step(raw,mode=mode)
                dest=Path(cmd['output']);dest.mkdir(exist_ok=True);summaries.append(task.save(dest))
            obs=task.reset(record=True,label=f'train-w{WORKER_ID}-{episodes}')
            CONNECTION.send(dict(kind='evaluate',summaries=summaries));continue
        assert cmd['op']=='rollout'
        generator=torch.Generator(device=obs.device).manual_seed(1001+WORKER_ID+100003*cmd['version'])
        batch={k:[] for k in ['obs','action','logp','value','next_value','reward','discount','done']};stats=[];eps=[];start=time.monotonic();presteps=task.total_prefill
        for _ in range(cmd['steps']):
            if task.done:
                result=task.summary();eps.append(result)
                if task.boundary or episodes%50==0:task.save(out)
                episodes+=1;obs=task.reset(record=True,label=f'train-w{WORKER_ID}-{episodes}')
            with torch.no_grad():
                dist=net.distribution(obs);raw=dist.mean+dist.stddev*torch.randn(dist.mean.shape,device=obs.device,generator=generator);lp=dist.log_prob(raw).sum(-1);value=net.value(obs)
                nxt,reward,done,st=task.step(raw);nv=net.value(nxt)
            for k,v in dict(obs=obs[0],action=raw[0],logp=lp[0],value=value[0],next_value=nv[0]).items():batch[k].append(v.detach().cpu().numpy())
            batch['reward'].append(reward);batch['discount'].append(.99**(st['control_steps']/5));batch['done'].append(done)
            stats.append(st);obs=nxt
        a={k:np.asarray(v,dtype=bool if k=='done' else np.float32) for k,v in batch.items()}
        CONNECTION.send(dict(kind='rollout',version=cmd['version'],batch=a,stats=stats,episodes=eps,seconds=time.monotonic()-start,prefill_physics_steps=task.total_prefill-presteps))
    if task.boundary:task.save(out)
    return dict(status='completed',episodes=episodes,physical_steps=task.total_steps,prefill_steps=task.total_prefill,student_optimizer_updates=0)
