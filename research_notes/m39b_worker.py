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
    from m39b_env import Task,install_frame
    install_frame()
    from clean_policy import Expert
    from m11_runtime_speed import install
    (out/'runtime-speed.json').write_text(json.dumps(install(env),indent=2))
    torch.set_num_threads(1);torch.manual_seed(1001+WORKER_ID+int(os.environ['NARROW_SEED'])*997);np.random.seed(1001+WORKER_ID+int(os.environ['NARROW_SEED'])*997)
    task=Task(env,model,contract,1001+WORKER_ID+int(os.environ['NARROW_SEED'])*997);task.neighborhood=os.environ.get('NARROW_DOMAIN')=='neighborhood';obs=task.reset(record=True,label=f'train-w{WORKER_ID}-0');net=Expert(obs.shape[-1]).to(obs.device).eval();episodes=0;reset_serial=0
    CONNECTION.send(dict(kind='ready',pid=os.getpid(),input_dim=obs.shape[-1],worker=WORKER_ID,standing=env.standing[0].cpu().tolist(),contract=contract))
    while True:
        cmd=CONNECTION.recv()
        if cmd['op']=='close':break
        if cmd['op']=='snapshot_partial':
            task.label+='-before-matched-reset';partial=task.save(out)
            CONNECTION.send(dict(kind='snapshot',worker=WORKER_ID,discarded_partial_episode=partial));continue
        net.load_state_dict({k:torch.from_numpy(v).to(obs.device) for k,v in cmd['policy'].items()})
        if cmd['op']=='evaluate':
            summaries=[]
            for label,spec,mode in cmd['schedule']:
                task.requested_mode=mode;obs=task.reset(spec,record=True,label=label,query_reference=(mode=='m10_expert'))
                diagnostic=bool(spec.get('diagnostic_sampling'))
                if diagnostic:
                    from m11_diagnostics import monitor_point_read,action_probe
                    read_audit=monitor_point_read(task) if spec.get('diagnostic_read_audit',True) else {};probes=[];probe_times=[5.5,7.,8.,10.]
                    noise=torch.Generator(device=obs.device).manual_seed(spec['diagnostic_seed'])
                while not task.done:
                    dist=net.distribution(obs);raw=torch.zeros_like(dist.mean) if spec.get('diagnostic_zero_mean') else dist.mean
                    if diagnostic:
                        if spec.get('diagnostic_action_probes',True) and probe_times and float(task.e.tether.time)>=probe_times[0]:
                            probes.append(action_probe(task,net,obs));probe_times.pop(0)
                        epsilon=torch.randn(raw.shape,device=obs.device,generator=noise)
                        raw=raw+spec['diagnostic_std_scale']*dist.stddev*epsilon
                    if spec.get('perturb_axis') is not None and 7.0<=float(task.e.tether.time)<7.5:
                        raw=raw.clone();raw[:,spec['perturb_axis']]=spec.get('perturb_value',.3)
                    obs,_,_,_=task.step(raw,mode=mode)
                dest=Path(cmd['output']);dest.mkdir(exist_ok=True);row=task.save(dest)
                from m39b_metrics import extra
                row.update(extra(task))
                if diagnostic:
                    row['point_read_audit']=read_audit
                    (dest/(label+'-action-probes.json')).write_text(json.dumps(probes,indent=2))
                summaries.append(row)
            task.requested_mode='learned'
            obs=task.reset(record=True,label=f'train-r{reset_serial}-w{WORKER_ID}-{episodes}')
            CONNECTION.send(dict(kind='evaluate',summaries=summaries));continue
        assert cmd['op']=='rollout'
        generator=torch.Generator(device=obs.device).manual_seed(1001+WORKER_ID+int(os.environ['NARROW_SEED'])*997+int(os.environ['NARROW_SEED'])*1000003+100003*cmd['version'])
        batch={k:[] for k in ['obs','action','logp','value','next_value','reward','discount','done']};stats=[];eps=[];start=time.monotonic();presteps=task.total_prefill
        for _ in range(cmd['steps']):
            if task.done:
                result=task.summary();eps.append(result)
                if task.boundary or episodes%50==0:task.save(out)
                episodes+=1;obs=task.reset(record=True,label=f'train-r{reset_serial}-w{WORKER_ID}-{episodes}')
            with torch.no_grad():
                dist=net.distribution(obs);raw=dist.mean+dist.stddev*torch.randn(dist.mean.shape,device=obs.device,generator=generator);lp=dist.log_prob(raw).sum(-1);value=net.value(obs)
                nxt,reward,done,st=task.step(raw);nv=net.value(nxt)
            for k,v in dict(obs=obs[0],action=raw[0],logp=lp[0],value=value[0],next_value=nv[0]).items():batch[k].append(v.detach().cpu().numpy())
            batch['reward'].append(reward);batch['discount'].append(.99**(st['control_steps']/5));batch['done'].append(done)
            stats.append(st);obs=nxt
        a={k:np.asarray(v,dtype=bool if k=='done' else np.float32) for k,v in batch.items()}
        CONNECTION.send(dict(kind='rollout',version=cmd['version'],executed_std=net.log_std.exp().detach().cpu().numpy(),batch=a,stats=stats,episodes=eps,seconds=time.monotonic()-start,prefill_physics_steps=task.total_prefill-presteps))
    if task.boundary:task.save(out)
    return dict(status='completed',episodes=episodes,physical_steps=task.total_steps,prefill_steps=task.total_prefill,student_optimizer_updates=0)
