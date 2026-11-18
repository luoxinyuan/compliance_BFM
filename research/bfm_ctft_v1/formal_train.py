"""Fixed-N CTFT v1.1 U/T over the qualified native learning and Task path.

The shared warm start is the original formal U Actor-frozen round-8192.
The v1.1 behavior choice affects g0 collection only. Each new round
still commits full lossless state after its 1024 rows and eight updates.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(path,value):
    from cas_checkpoint import durable
    durable(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def study(e,original,contract,process_out):
    import torch
    from training_live import TrainingLive
    from training_data import episode
    from contact_replay import cat
    from learning_checkpoint import digest,file_sha,one_update,set_rng
    from gradient_housekeeping import clear_consumed
    from cas_checkpoint import learner_state,restore_learner
    from production_commit import RoundCommitter
    from packed_checkpoint import PackedTensorStore
    from f1_mjdata_contract import canonical,metadata
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from resource_guard import check
    args=json.loads(os.environ['CTFT_FORMAL_ARGS'])
    out=Path(args['output'])
    revision=json.loads((STAGE/'locks/bfm-ctft-v1a-signed.json').read_text())
    assert revision['approved'] and revision['version']=='1.1'
    speed(e)
    (process_out/'backend-prefix').mkdir()
    backend(e,original,process_out/'backend-prefix')
    reset_common(e,original,original.act)
    lock=json.loads((STAGE/'locks/common-N-resource-lock-output250-v4.json').read_text())
    n=lock['N']
    assert lock['status']=='COMMON_N_RESOURCE_LOCK' and n==131072
    plan_path=STAGE/'locks/exogenous-training-plan-v2.json'
    reset_path=STAGE/'locks/source-reset-domain-v1.json'
    plan=json.loads(plan_path.read_text());reset=json.loads(reset_path.read_text())
    records=json.loads((STAGE/'runs/s0-source-admission-01/result.json').read_text())['train']
    coeff=json.loads((STAGE/'runs/s0-online-probe-01/agent-identity.json').read_text())['original_training_config']['aux_rewards_scaling']
    reference=STAGE/'runs/s0-cas-reference-02/reference'
    initial_rows=revision['shared_rows']
    assert initial_rows==8192 and revision['formal_N_per_arm']==n
    milestones=[initial_rows,n//4,n//2,3*n//4,n]
    live=TrainingLive(e,STAGE,coeff,records)
    stops={'requested':False}
    def request_stop(signum,frame): stops['requested']=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    warm_name='round-8192'
    warm=PackedTensorStore(STAGE/'runs/formal-U-01/store')
    warm_manifest=json.loads((warm.root/warm_name/'manifest.json').read_text())
    assert warm_manifest['anchor'] and warm_manifest['complete']
    assert warm_manifest['state_sha256']==revision['warm_state_sha256']
    warm_model=warm.metadata(warm_name)['learner']['model']
    b0_state={k.removeprefix('_actor.'):warm.unpack(v) for k,v in warm_model.items()
              if k.startswith('_actor.')}
    committer=RoundCommitter(out/'store',total_rows=n,milestone_rows=milestones,
                             create=not args['resume'],initial_rows=initial_rows)
    if not args['resume']:
        alias=committer.adopt(initial_rows,warm.root,warm_name)
        assert alias['newly_written_blob_bytes']==0
        write(out/'common-warm-start.json',dict(source_state_sha256=warm_manifest['state_sha256'],
              alias=alias,shared_rows=initial_rows,shared_updates=64,arm=args['arm']))
    package=committer.load_latest()
    if not args['resume']:
        assert package['learner']['step']==64 and not package['learner']['actor_enabled']
        assert package['scheduler']['collected']==initial_rows
    else:
        assert package['revision']=='1.1' and package['arm']==args['arm']

    def exact_restore(saved):
        with torch.enable_grad():
            model,replay_buffers=restore_learner(saved['learner'])
        live.restore(saved['task'])
        set_rng(saved['learner']['rng'])
        meta=metadata(e.sim.model,e.sim.data)
        assert digest(canonical(live.snapshot(),meta))==digest(canonical(saved['task'],meta))
        return model,replay_buffers

    agent,buffers=exact_restore(package)
    initial_actor_sha=package['initial_actor_sha']
    assert digest(b0_state)==initial_actor_sha==revision['frozen_actor_sha256']
    b0_actor=copy.deepcopy(agent._model._actor)
    b0_actor.load_state_dict(b0_state,strict=True)
    b0_actor.requires_grad_(False)
    def behavior(obs,z,mode):
        with torch.no_grad():
            if mode==0:
                model=agent._model
                with torch.autocast(device_type=model.device,dtype=model.amp_dtype,enabled=model.cfg.amp):
                    return b0_actor(model._normalize(obs),z,model.cfg.actor_std).mean.float()
            assert mode==1
            return agent.act(obs,z,mean=False)

    scheduler=copy.deepcopy(package['scheduler'])
    segments=copy.deepcopy(package['learner']['replay_segments'])
    blocks=json.loads((out/'blocks.json').read_text()) if (out/'blocks.json').exists() else []
    blocks=[b for b in blocks if b['collected']<=scheduler['collected']]
    if not args['resume']:
        assert digest(agent._model._actor.state_dict())==initial_actor_sha
        ob,z=live.observation(),live.command()
        assert torch.equal(behavior(ob,z,0),agent.act(ob,z,mean=True))
        set_rng(package['learner']['rng'])
        write(out/'v1a-identity.json',dict(
            warm_state_sha256=warm_manifest['state_sha256'],frozen_actor_sha256=initial_actor_sha,
            qualification='exact B0 mean on actual warm handoff observation',
            shared_rows=initial_rows,mode_0='frozen B0 mean',mode_1='current actor original stochastic',
            plan_sha256=file_sha(plan_path),reset_sha256=file_sha(reset_path)))
    else:
        write(out/f'resume-{time.time_ns()}.json',dict(collected=scheduler['collected'],exact_learner_and_Task_restore=True))

    def next_action(state):
        if live.done:
            spec=episode(plan,reset,state['source_index'],args['arm'])
            state['source_index']+=1
            live.start(spec)
            state['episodes'].append(dict(spec=spec,controls=0,complete=False))
        before,z=live.observation(),live.command()
        return before,z,behavior(before,z,live.spec['mode'])

    def restore_canary():
        """Non-formal short trajectory and update used only for restart proof."""
        st=copy.deepcopy(scheduler)
        evidence=[]
        ticks=[]
        for _ in range(12):
            before,z,action=next_action(st)
            evidence.append(live.step(action,before,z))
            ticks.extend(copy.deepcopy(e.trace[-4:]))
        with torch.enable_grad():
            updated=one_update(agent,buffers,scheduler['updates'])
        return dict(transition_sha256=digest(evidence),physics_sha256=digest(ticks),
                    task_sha256=digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data))),
                    update_model_optimizer_rng=updated['state'],tested_controls=12,
                    update_metrics=updated['metrics'],sample_trace=updated['sample_trace'],
                    formal_updates=0,formal_rows=0)

    if args['resume'] and scheduler['collected']==initial_rows+1024:
        expected=json.loads((out/'first-round-uninterrupted-canary.json').read_text())
        optimizer_before={key:digest(value) for key,value in package['learner']['optimizers'].items()}
        actual=restore_canary()
        assert actual==expected,'FIRST_ROUND_ACTUAL_NEXT_TRAJECTORY_OR_UPDATE_NOT_EXACT'
        optimizer_changed=[key for key,value in package['learner']['optimizers'].items()
                           if digest(value)!=optimizer_before[key]]
        write(out/f'first-round-canary-match-{time.time_ns()}.json',dict(
            tested_controls=12,same_next_update=True,
            optimizer_sources_mutated_in_deserialized_package=optimizer_changed))
        # Adam can retain CPU step-tensor references from load_state_dict.
        # The deserialized package is therefore scratch after the canary.
        # Reopen the authoritative immutable package rather than reuse any
        # Python objects that might have been changed by an optimizer.
        del agent,buffers,package
        package=committer.load_latest()
        assert package['scheduler']['collected']==initial_rows+1024
        assert package['learner']['step']==64+8
        agent,buffers=exact_restore(package)
        write(out/'first-round-recovery-qualified.json',dict(
            status='PASS',committed_rows=scheduler['collected'],
            production_state_restored_again=True,production_state_reloaded_from_disk=True,
            canary_rows_entered_replay=0))
    if args['resume'] and scheduler['collected']>initial_rows+1024:
        assert json.loads((out/'first-round-recovery-qualified.json').read_text())['status']=='PASS'
    del package,warm_model,b0_state

    def full_state():
        state=learner_state(agent,step=scheduler['updates'],replay_segments=segments,
            replay_capacity=buffers['train'].capacity,replay_idx=buffers['train']._idx,
            reference=reference,replay_quotas=buffers['train'].quotas)
        return dict(learner=state,task=live.snapshot(),scheduler=copy.deepcopy(scheduler),
                    initial_actor_sha=initial_actor_sha,formal_training=True,arm=args['arm'],N=n,
                    revision='1.1',shared_rows=initial_rows)
    physics=[]
    def persist_physics(label):
        nonlocal physics
        if not physics:return
        path=out/f'physics-{label}.pkl.gz'
        assert not path.exists()
        with gzip.open(path,'xb',compresslevel=1) as f:pickle.dump(physics,f,protocol=5)
        physics=[]
    def output_guard():
        # Conservative fixed S0 base plus actual new formal directories. Do
        # not repeatedly stat every old stage artifact at each training step.
        total=lock['stage_storage']['logical_bytes']+2*2**30
        for directory in STAGE.glob('runs/formal-*-*'):
            for root,dirs,names in os.walk(directory,followlinks=False):
                for name in names:
                    p=Path(root)/name
                    if not p.is_symlink():total+=p.stat().st_size
        assert total<250*2**30,'STAGE_OUTPUT_GUARD'
        return total
    try:
        while scheduler['collected']<n:
            check(STAGE)
            assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC']),'ARM_JOB_LIMIT'
            assert not stops['requested'],'USER_NORMAL_STOP'
            started=time.monotonic();rows=[]
            for i in range(1024):
                assert not stops['requested'],'USER_NORMAL_STOP'
                assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC']),'ARM_JOB_LIMIT'
                before,z,action=next_action(scheduler)
                rows.append(live.step(action,before,z))
                physics.extend(copy.deepcopy(e.trace[-4:]))
                scheduler['collected']+=1
                scheduler['episodes'][-1]['controls']+=1
                scheduler['episodes'][-1]['complete']=live.done
            collect_s=time.monotonic()-started
            data=cat(rows);buffers['train'].extend(data)
            rowpath=out/f'actual-rows-{scheduler["collected"]:06d}.pt'
            assert not rowpath.exists();torch.save(data,rowpath)
            g0_count=int((data['ctft_replay_phase']==0).sum())
            segments.append(dict(path=str(rowpath),sha256=file_sha(rowpath),rows=1024,
                provenance=dict(g0_frozen_b0_mean=g0_count,g1_current_actor_sample=1024-g0_count,
                                discriminator='actual ctft_replay_phase==0 on v1.1 new segments')))
            persist_physics(str(scheduler['collected']))
            collected=scheduler['collected']
            assert collected>initial_rows
            count=8
            agent.actor_enabled=True
            update_s=[];metrics_record=[]
            for _ in range(count):
                torch.cuda.synchronize();t=time.monotonic()
                with torch.enable_grad(): metrics=agent.update(buffers,scheduler['updates'])
                torch.cuda.synchronize();update_s.append(time.monotonic()-t)
                assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
                metrics_record.append({k:float(v.detach().mean().cpu()) if torch.is_tensor(v) else v for k,v in metrics.items()})
                scheduler['updates']+=1
            clear_consumed(agent)
            save=committer.commit(collected,full_state())
            free,total=torch.cuda.mem_get_info()
            assert torch.cuda.max_memory_reserved()<=.88*total and free>=max(6*2**30,.08*total)
            block=dict(collected=collected,updates=scheduler['updates'],actual_update_count=count,
                collect_s=collect_s,update_s=update_s,save=save,total_s=time.monotonic()-started,
                actual_phase_rows={k:len(v) for k,v in buffers['train'].members.items()},
                actual_episodes=copy.deepcopy(scheduler['episodes']),metrics=metrics_record,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),output_conservative_bytes=output_guard())
            blocks.append(block);write(out/'blocks.json',blocks)
            write(out/'progress.json',dict(status='COMMITTED',arm=args['arm'],N=n,
                collected=collected,updates=scheduler['updates'],completed_rounds=len(blocks),
                latest_manifest_sha256=file_sha(out/'store'/f'round-{collected}'/'manifest.json')))
            print('CTFT_FORMAL_COMMIT',args['arm'],collected,scheduler['updates'],block['total_s'],flush=True)
            if args['qualify_first_round']:
                assert collected==initial_rows+1024
                write(out/'first-round-uninterrupted-canary.json',restore_canary())
                return dict(status='PAUSED_FOR_FRESH_PROCESS_RECOVERY',arm=args['arm'],
                            actual_rows=collected,actual_update_groups=scheduler['updates'])
            if collected in [16384,n//4,n//2,3*n//4,n]:
                # Mandatory finite-N growth recheck, using actual complete
                #rounds. No reward/Dev criterion enters this resource gate.
                measured_max=max(b['total_s'] for b in blocks[-8:])
                actual_wall=float(os.environ['CTFT_PRIOR_WALL'])+time.monotonic()-float(os.environ['CTFT_RUN_START_MONO'])
                projected_arm=actual_wall+(n-collected)/1024*measured_max*1.25+600.
                trace_bytes=sum(p.stat().st_size for p in out.glob('physics-*.pkl.gz'))
                row_bytes=sum(p.stat().st_size for p in out.glob('actual-rows-*.pt'))
                peak_pack=max([lock['measured']['first_full_pack_bytes']]+[
                    b['save']['newly_written_blob_bytes'] for b in blocks])
                projected_output=revision['base_stage_output_bytes']+1.25*(
                    11*peak_pack+2*(n-initial_rows)*(row_bytes+trace_bytes)/(collected-initial_rows)+8*2**30)
                check_record=dict(
                    actual_rows=collected,source_episodes=scheduler['episodes'],
                    instantaneous_physics_hard_guards_active=True,
                    no_development_or_sealed_read=True,
                    function_metrics='Fullactual causal rows + immutable5ms physics segments; no latent similarity success claim',
                    projected_arm_s=projected_arm,projected_output_bytes=projected_output,
                    trace_bytes=trace_bytes,actual_row_bytes=row_bytes,
                    recent_full_round_costs=blocks[-8:])
                write(out/f'train-only-check-{collected}.json',check_record)
                assert projected_arm<=43200,'FIXED_N_REMAINDER_TIME_INFEASIBLE'
                assert projected_output<=250*2**30,'FIXED_N_REMAINDER_OUTPUT_INFEASIBLE'
        return dict(status='FORMAL_FIXED_ENDPOINT_COMMITTED',arm=args['arm'],N=n,
                    actual_rows=scheduler['collected'],actual_update_groups=scheduler['updates'])
    except BaseException as exc:
        # Never delete/overwrite a prior round or label partial rows committed.
        persist_physics('failure-'+str(time.time_ns()))
        torch.save(dict(Task=live.snapshot(),partial_actual_rows=locals().get('rows',[]),
                        scheduler=scheduler,error=repr(exc)),out/f'failure-{time.time_ns()}.pt')
        failure=full_state()
        failure['pending_actual_rows']=locals().get('rows',[])
        failure['error']=repr(exc)
        failure['not_a_committed_round']=True
        failure_store=PackedTensorStore(out/'failure-store')
        base=copy.deepcopy(committer.store.metadata(committer.permanent[-1]))
        def rebase(v):
            if isinstance(v,dict):
                if 'file' in v and 'raw_sha256' in v:
                    v['file']=os.path.relpath(committer.store.root/v['file'],failure_store.root)
                for x in v.values():rebase(x)
            elif isinstance(v,(list,tuple)):
                for x in v:rebase(x)
        rebase(base)
        # Independent immutable failure package; atomic round latest stays at
        #the last complete committed boundary. Never resumes a half-update.
        failure_store.save('failure-'+str(time.time_ns()),failure,base=base,anchor=True)
        write(out/'stopped.json',dict(reason=repr(exc),actual_rows=scheduler['collected'],
            last_committed=committer.completed[-1],no_automatic_scientific_restart=True))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['U','T'],required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--qualify-first-round',action='store_true')
    a=p.parse_args()
    assert a.resume!=a.qualify_first_round
    a.output=a.output.resolve()
    if a.resume: assert a.output.exists()
    else:a.output.mkdir(parents=True,exist_ok=False)
    from resource_guard import check
    write(a.output/f'preflight-{time.time_ns()}.json',check(STAGE))
    prior=sum(json.loads(f.read_text())['wall_s'] for f in a.output.glob('cost-*.json'))
    assert prior<43200
    start=time.monotonic()
    process_out=a.output/'attempts'/str(time.time_ns())
    process_out.mkdir(parents=True,exist_ok=False)
    os.environ.update(CTFT_FORMAL_ARGS=json.dumps(dict(arm=a.arm,resume=a.resume,
        output=str(a.output),qualify_first_round=a.qualify_first_round)),
        CTFT_PRIOR_WALL=str(prior),CTFT_RUN_START_MONO=str(start),
        BFM_RECOVERY_OUTPUT=str(process_out),CCL_DEADLINE_MONOTONIC=str(start+43200-prior))
    write(a.output/f'launch-{time.time_ns()}.json',dict(pid=os.getpid(),arm=a.arm,
        runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        started_unix=time.time(),formal_training=True))
    import mujoco
    raw=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));counts['reserved']+=n
        assert counts['reserved']<=2000000,'FORMAL_ARM_PHYSICS_BUDGET'
        result=raw(*args,**kw);counts['actual']+=n;return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=process_out
    try:m10_physics.main()
    finally:write(a.output/f'cost-{time.time_ns()}.json',dict(wall_s=time.monotonic()-start,
        physics=counts,formal_training=True,arm=a.arm))


if __name__=='__main__':main()
