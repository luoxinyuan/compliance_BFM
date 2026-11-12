"""Isolated child entry. All monkeypatches scoped to this process and recorded."""
import os,sys,time,json,argparse,subprocess,hashlib,traceback,signal,resource,shutil
STOP_REQUESTED=False
def request_stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,m,c,out):
    import torch,numpy as np
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.agent import Agent
    from research.interaction_fb.schema import deploy
    from research.interaction_fb.checkpoint import restore
    from research.interaction_fb.journal import save,adopt
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import global_rng
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
    env=Environment(e,m,c);env.reset(0)
    env.proxy.start=env.task.last_command.clone();env.proxy.target=env.proxy.start.clone()
    obs=env.proxy.enrich(env.task.extended_observation())
    torch.manual_seed(6201);np.random.seed(6201)
    agent=Agent(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'),ARGS.arm);env.proxy.agent=agent
    if ARGS.envelope_lock and not ARGS.resume and not ARGS.envelope_qualification and ARGS.arm!='A':
        from research.interaction_fb.envelope import Envelope
        agent.envelope=Envelope(json.loads(ARGS.envelope_lock.read_text())['operator'],agent.goal.device)
    if ARGS.mode=='pilot':
        agent.exploration_std=ARGS.noise['std'];agent.exploration_clip=ARGS.noise['clip']
    # Exact sampling contract check without consuming training RNG or modifying weights.
    from dual_track_task import global_rng,restore_global_rng
    saved_rng=global_rng()
    with torch.no_grad():
        mean=agent.action(obs,sample=False)[0]
        sampled=agent.action(obs,sample=True)[0]
        after_rng=global_rng()
        restore_global_rng(saved_rng)
        if agent.envelope is None:
            expected=(mean+(agent.exploration_std*torch.randn_like(mean)).clamp(-agent.exploration_clip,agent.exploration_clip)).clamp(-1,1)
        else:
            zi=agent.zi(1);_,z,_=agent.action(obs,sample=False)
            from research.interaction_fb.schema import robot
            base=agent.base._actor(agent.base._normalize(robot(obs)),z,agent.base.cfg.actor_std).mean
            adapter=agent.nets['adapter'](torch.cat([deploy(obs),z,zi],-1))
            noise=(agent.exploration_std*torch.randn_like(base)).clamp(-agent.exploration_clip,agent.exploration_clip)
            expected=agent.envelope(base,(base+adapter+noise).clamp(-1,1))[0]
        assert torch.equal(sampled,expected),'Actual sampling differs from declared one-draw contract'
        assert tree_digest(global_rng())==tree_digest(after_rng),'Sampling consumed extra random draws'
        restore_global_rng(saved_rng)
        assert torch.equal(mean,agent.action(obs,sample=False)[0])
    restore_global_rng(saved_rng)
    # Permission test: all hidden truth can change without changing deploy input/action.
    hidden={k:v.clone() for k,v in obs.items()};hidden['privileged_state']+=100
    for lo,hi in [(9,15),(18,22)]:hidden['contact_privileged_v1'][:,lo:hi]+=100
    with torch.no_grad():
        assert torch.equal(deploy(obs),deploy(hidden))
        assert torch.equal(agent.action(obs)[0],agent.action(hidden)[0])
    write(out/'interface.json',dict(hidden_input_invariance_exact=True,params={k:sum(p.numel() for p in v.parameters()) for k,v in agent.nets.items()},
        frozen_base=True,exploration=dict(std=agent.exploration_std,clip=agent.exploration_clip),optimizer_names=list(agent.opts),robot_keys={k:list(v.shape) for k,v in e.obs.items()},
        global_hooks=['native_env.install:geometry_env.install_contact','dual_track_task.Task:clean_env.advance','rolling_backend.install:m8_stage.read'],
        isolated_one_Task_process=True))
    # Compare all common tensors at initialization, not just output-layer zeros.
    from diagnostic_fb_resume import tree_digest
    common={k:tree_digest(v.state_dict()) for k,v in agent.nets.items()}
    write(out/'alignment.json',dict(arm=ARGS.arm,common_parameter_digests=common,
        deploy_shape=list(deploy(obs).shape),initial_action_digest=tree_digest(agent.action(obs)[0]),
        contract='same initializer draw order; no hidden interaction prompt in A/B'))
    if ARGS.mode=='eval':
        from research.interaction_fb.evaluation import run
        from research.interaction_fb.journal import load_state
        from research.interaction_fb.env import specs
        if ARGS.model:agent.load(load_state(ARGS.model)['agent'])
        if ARGS.reference_bc500:
            assert ARGS.arm=='A' and not ARGS.model
            from clean_policy import Expert
            path=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/interactive-contact-pilot-20261108/runs/bc-fit-01/old_data_only-500.pt')
            assert hashlib.sha256(path.read_bytes()).hexdigest()=='a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e'
            ck=torch.load(path,map_location='cuda',weights_only=True);net=Expert(1639,'direct').cuda()
            net.load_state_dict(ck['state_dict'],strict=True);net.eval().requires_grad_(False)
            agent.command=lambda ob,target=False:net.distribution(env.task.student_features()).mean
            env.proxy.reference_original=True
        if ARGS.manifest:
            assert ARGS.manifest.name==ARGS.split+'.json','No sealed evaluation in preparation/first read'
            audit=json.loads((ARGS.manifest.parent/'audit.json').read_text())
            assert hashlib.sha256(ARGS.manifest.read_bytes()).hexdigest()==audit['splits'][ARGS.split]['file_sha256']
            data=json.loads(ARGS.manifest.read_text());cases=[]
            for g in data['parents']:
                for sp in g['specs']:
                    sp=dict(sp,split=ARGS.split)
                    if g['family']=='stop':sp['pair_branch_at']=g['specs'][0]['stop_time']
                    cases.append(sp)
        else:
            cases=[specs()[2],specs()[3],specs()[5]]
            for case in cases[:2]:case['pair_branch_at']=cases[0]['stop_time']
        return run(env,agent,out,cases)
    if ARGS.mode=='pilot' or ARGS.failure_probe or ARGS.envelope_qualification:
        data=json.loads(ARGS.manifest.read_text())
        env.specs=[dict(sp,split='train') for g in data['parents'] for sp in g['specs']]
        # Initialization stays canonical/common; training replay starts from real locked source.
        env.rows=[];env.native=[];env.physical_stream=[];env.episodes=[];env.next_cursor=0
        env.reset(0)
    round0=0
    if ARGS.resume:
        round0=restore(ARGS.resume,agent,env)
        if not ARGS.failure_probe and not ARGS.envelope_qualification:
            if ARGS.mode=='pilot':assert dict(std=agent.exploration_std,clip=agent.exploration_clip)==ARGS.noise,'Resume exploration contract mismatch'
            adopt(out/'checkpoints',ARGS.resume)
    else:save(out/'checkpoints',0,agent,env,ARGS.runtime,'initial')
    if ARGS.envelope_qualification:
        from research.interaction_fb.envelope_qualification import run
        lock=json.loads(ARGS.envelope_lock.read_text());case=next(c for c in lock['qualification_states'] if c['id']==ARGS.envelope_qualification)
        if ARGS.qual_bundle:
            results=[]
            for branch in case['branches']:
                agent.action=Agent.action.__get__(agent,Agent);agent.last_action_audit=None;agent.envelope=None
                restore(ARGS.resume,agent,env)
                dest=out/branch['id'];dest.mkdir()
                result=run(env,agent,dest,lock,case,branch['mode'],branch['sample']);results.append(result)
                write(out/'qualification-bundle.json',results)
                # Immediate safety stop on the proposed execution path; old on failures retained as controls.
                if branch['mode']=='envelope' and result['status']!='completed_3s':break
            return dict(status='qualification_bundle',results=results)
        return run(env,agent,out,lock,case,ARGS.envelope_mode,not ARGS.qual_mean)
    if ARGS.failure_probe:
        assert ARGS.resume and ARGS.manifest.name=='train.json'
        frozen=tree_digest(agent.state());start_cursor=env.next_cursor
        start_rng=tree_digest(global_rng());start_physics=len(env.physical_stream)
        env.proxy.stochastic=ARGS.failure_probe not in ('mean','offmean')
        if ARGS.failure_probe in ('offsample','offmean'):
            from research.interaction_fb.schema import robot
            original_action=agent.action
            def adapter_off(obs,target=False,sample=False,z_override=None):
                _,z,zi=original_action(obs,target=target,sample=False,z_override=z_override)
                a=agent.base._actor(agent.base._normalize(robot(obs)),z,agent.base.cfg.actor_std).mean.clamp(-1,1)
                if sample:
                    a=(a+(agent.exploration_std*torch.randn_like(a)).clamp(-agent.exploration_clip,agent.exploration_clip)).clamp(-1,1)
                return a,z,zi
            agent.action=adapter_off
        if ARGS.failure_probe in ('ankle','uniform'):
            original_action=agent.action
            def ankle_diagnostic(obs,target=False,sample=False,z_override=None):
                a,z,zi=original_action(obs,target=target,sample=False,z_override=z_override)
                if sample:
                    draw=torch.randn_like(a)
                    if ARGS.failure_probe=='ankle':
                        noise=(.05*draw).clamp(-.3,.3);noise[:,[5,11]]*=.2
                    else:noise=(.01*draw).clamp(-.06,.06)
                    a=(a+noise).clamp(-1,1)
                return a,z,zi
            agent.action=ankle_diagnostic
        # Diagnostic-only observation of the same actual state, without replacing actions.
        observed_actions=[];execute_action=agent.action
        from research.interaction_fb.schema import robot,deploy
        import mujoco
        names=c['dof_names'];joint_ids=[mujoco.mj_name2id(e.sim.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
        joint_contract=[dict(name=n,qposadr=int(e.sim.model.jnt_qposadr[j]),qveladr=int(e.sim.model.jnt_dofadr[j]),range=e.sim.model.jnt_range[j].tolist(),limited=bool(e.sim.model.jnt_limited[j])) for n,j in zip(names,joint_ids)]
        def observed_action(obs,target=False,sample=False,z_override=None):
            mean,z,zi=execute_action(obs,target=target,sample=False,z_override=z_override)
            actual,zz,ii=execute_action(obs,target=target,sample=sample,z_override=z_override)
            assert torch.equal(z,zz) and torch.equal(zi,ii)
            base=agent.base._actor(agent.base._normalize(robot(obs)),z,agent.base.cfg.actor_std).mean
            correction=agent.nets['adapter'](torch.cat([deploy(obs),z,zi],-1)) if agent.arm!='A' else torch.zeros_like(base)
            off=torch.as_tensor(c['q_PD_offset'],device=mean.device);scale=agent.scale
            observed_actions.append(dict(adapter_applied=ARGS.failure_probe not in ('offsample','offmean'),time=float(env.task.e.tether.time),mean=mean.detach().cpu().tolist(),actual=actual.detach().cpu().tolist(),base_mean=base.detach().cpu().tolist(),adapter=correction.detach().cpu().tolist(),mean_q_PD=(mean*scale+off).detach().cpu().tolist(),actual_q_PD=(actual*scale+off).detach().cpu().tolist(),base_q_PD=(base.clamp(-1,1)*scale+off).detach().cpu().tolist(),pre_qpos=e.sim.data.qpos.tolist(),pre_qvel=e.sim.data.qvel.tolist(),slot=obs['_slot'].detach().cpu().tolist()))
            return actual,zz,ii
        agent.action=observed_action
        status='completed_3s';error=None
        try:
            with torch.no_grad():
                for _ in range(30):env.collect_call()
        except RuntimeError as exc:
            if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
            status='hard_physics_boundary';error=str(exc)
        assert tree_digest(agent.state())==frozen, 'Diagnostic changed frozen model'
        trace=dict(physics=env.physical_stream[start_physics:],controls=env.task.controls,
                   commands=env.commands,actions=env.proxy.actions,task_spec=env.task.spec)
        torch.save(trace,out/'probe-trace.pt')
        write(out/'same-state-actions.json',dict(contract=joint_contract,controls=observed_actions,scope='frozen_same_state_adapter_off_only_when_explicitly_requested'))
        result=dict(status=status,error=error,mode=ARGS.failure_probe,restored_round=round0,
                    original_cursor=start_cursor,end_cursor=env.next_cursor,updates=agent.updates,
                    frozen_agent_digest=frozen,restored_rng_digest=start_rng,
                    trace_sha256=hashlib.sha256((out/'probe-trace.pt').read_bytes()).hexdigest(),
                    boundary=env.task.boundary,physics=trace['physics'])
        write(out/'probe.json',result)
        return {k:v for k,v in result.items() if k!='physics'}
    records=[]
    try:
        for r in range(round0+1,ARGS.rounds+1):
            begun=time.monotonic();sample=time.monotonic()
            with torch.no_grad():
                for i in range(50):
                    env.collect_call()
                    if ARGS.inject and r==2 and i==2:raise InterruptedError('planned mid-round injection')
            if ARGS.mode=='pilot':assert env.next_cursor==r*250,'Partial control step: preserve failure, do not miscount'
            collect_s=time.monotonic()-sample;update=time.monotonic()
            with torch.enable_grad():metrics=[agent.update(env.rows,env.native) for _ in range(2)]
            torch.cuda.synchronize();update_s=time.monotonic()-update;io=time.monotonic()
            manifest=save(out/'checkpoints',r,agent,env,ARGS.runtime)
            io_s=time.monotonic()-io
            rec=dict(round=r,transitions=env.next_cursor,updates=agent.updates,collect_s=collect_s,update_s=update_s,io_s=io_s,
                host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,free_disk_bytes=shutil.disk_usage(out).free,
                total_s=time.monotonic()-begun,peak_cuda_bytes=torch.cuda.max_memory_allocated(),metrics=metrics,
                contact_rows=sum(bool(x['observation']['contact_privileged_v1'][0,21]) for x in env.rows),
                fingerprints=dict(agent=tree_digest(agent.state()),replay=tree_digest(env.rows),rng=tree_digest(global_rng()),
                    physics_stream=tree_digest(env.physical_stream),qpos=tree_digest(e.sim.data.qpos),qvel=tree_digest(e.sim.data.qvel),history=tree_digest(e.base.history_handler.history)))
            records.append(rec);write(out/'rounds.json',records);print('COMMITTED',r,rec['total_s'],flush=True)
            if STOP_REQUESTED:return dict(status='normal_stop_after_committed_round',rounds=records)
    except Exception as exc:
        write(out/'failure-evidence.json',dict(type=type(exc).__name__,error=str(exc),cursor=env.next_cursor,updates=agent.updates,
            rounds=records,physics=env.task.physics[-20:],last_controls=env.task.controls[-5:]))
        # Preserve uncommitted work separately. Never call it a completed round.
        torch.save(dict(agent=agent.state(),rows=env.rows,commands=env.commands,actions=env.proxy.actions,
                        task_spec=env.task.spec,physics=env.task.physics,controls=env.task.controls),out/'uncommitted-stop.pt')
        if isinstance(exc,InterruptedError):return dict(status='injected_stop',rounds=records)
        raise
    return dict(status='trial_completed',rounds=records)

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
    global ARGS
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--inject',action='store_true');p.add_argument('--arm',choices=['A','B','C'],required=True);p.add_argument('--mode',choices=['bench','eval','pilot'],default='bench');p.add_argument('--model',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--split',choices=['train','dev'],default='dev');p.add_argument('--reference-bc500',action='store_true');p.add_argument('--approval-lock',type=Path);p.add_argument('--failure-probe',choices=['sample','mean','ankle','uniform','offsample','offmean']);p.add_argument('--envelope-lock',type=Path);p.add_argument('--envelope-qualification');p.add_argument('--envelope-mode',choices=['on','off','envelope']);p.add_argument('--qual-mean',action='store_true');p.add_argument('--qual-bundle',action='store_true');ARGS=p.parse_args()
    if ARGS.mode=='pilot':
        assert ARGS.approval_lock is not None
        lock=json.loads(ARGS.approval_lock.read_text())
        assert lock['status']=='approved_by_Yi' and lock.get('approval_reference'), 'Pilot requires explicit revised-lock approval'
        assert ARGS.manifest and ARGS.manifest.name=='train.json'
        assert lock['transitions_per_arm']%250==0 and lock['updates_per_round']==2 and lock['batch_size']==32
        assert hashlib.sha256(ARGS.manifest.read_bytes()).hexdigest()==lock['train_file_sha256']
        ARGS.rounds=lock['transitions_per_arm']//250
        ARGS.noise=lock.get('exploration',dict(std=.05,clip=.3))
        assert ARGS.noise in (dict(std=.05,clip=.3),dict(std=.01,clip=.06))
    else:ARGS.rounds=3
    ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    ARGS.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(BFM_RECOVERY_OUTPUT=str(ARGS.output),P4000_CANONICAL_ROW_ACTOR='1',CLEAN_KIND='direct',BFM_RECORD_NOMINAL_DIAGNOSTIC='0')
    import mujoco
    original=mujoco.mj_step;counts={'actual':0,'reserved':0}
    def counted(*a,**kw):
        n=int(kw.get('nstep',1));counts['reserved']+=n;write(ARGS.output/'physics-counter.json',counts)
        result=original(*a,**kw);counts['actual']+=n;return result
    mujoco.mj_step=counted
    os.chdir(ARGS.output)
    import m10_stage,m10_physics
    m10_physics.OUT=ARGS.output;m10_stage.run_study=study
    try:m10_physics.main()
    finally:write(ARGS.output/'physics-counter.json',counts)
if __name__=='__main__':main()
