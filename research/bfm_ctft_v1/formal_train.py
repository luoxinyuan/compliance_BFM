"""Fixed-N U/T driver over the qualified native CTFT learning/Task path.

No learning equation, replay quota, source/partner plan or physical gate is
defined here. Each1024-row round commits a complete lossless state before its
completion is reported. S0 learner states are never formal initialization.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import pickle
import random
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


def study(e,original,contract,out):
    import numpy as np
    import torch
    from agent import build
    from training_live import TrainingLive
    from training_data import PhaseReplay,episode
    from native_reference import load_reference
    from contact_replay import cat
    from learning_checkpoint import digest,file_sha
    from gradient_housekeeping import clear_consumed
    from cas_checkpoint import learner_state,restore_learner
    from production_commit import RoundCommitter
    from packed_checkpoint import PackedTensorStore
    from f1_mjdata_contract import canonical,metadata
    from locomotion_rollout import UnsafePhysics
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from resource_guard import check
    speed(e)
    (out/'backend-prefix').mkdir()
    backend(e,original,out/'backend-prefix')
    reset_common(e,original,original.act)
    args=json.loads(os.environ['CTFT_FORMAL_ARGS'])
    lock=json.loads((STAGE/'locks/common-N-resource-lock-output250-v4.json').read_text())
    n=lock['N']
    assert lock['status']=='COMMON_N_RESOURCE_LOCK' and n==131072
    plan_path=STAGE/'locks/exogenous-training-plan-v2.json'
    reset_path=STAGE/'locks/source-reset-domain-v1.json'
    plan=json.loads(plan_path.read_text());reset=json.loads(reset_path.read_text())
    records=json.loads((STAGE/'runs/s0-source-admission-01/result.json').read_text())['train']
    coeff=json.loads((STAGE/'runs/s0-online-probe-01/agent-identity.json').read_text())['original_training_config']['aux_rewards_scaling']
    scale=json.loads((STAGE/'locks/common-aux-scale-v1.json').read_text())['value']
    reference=STAGE/'runs/s0-cas-reference-02/reference'
    milestones=[0,8192,n//4,n//2,3*n//4,n]
    live=TrainingLive(e,STAGE,coeff,records)
    stops={'requested':False}
    def request_stop(signum,frame): stops['requested']=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    if args['resume']:
        committer=RoundCommitter(out/'store',total_rows=n,milestone_rows=milestones,create=False)
        package=committer.load_latest()
        with torch.enable_grad(): agent,buffers=restore_learner(package['learner'])
        live.restore(package['task'])
        assert digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data)))==digest(canonical(package['task'],metadata(e.sim.model,e.sim.data)))
        scheduler=copy.deepcopy(package['scheduler'])
        segments=copy.deepcopy(package['learner']['replay_segments'])
        blocks=json.loads((out/'blocks.json').read_text()) if (out/'blocks.json').exists() else []
        # Atomic committed state is authoritative after a write interruption.
        blocks=[b for b in blocks if b['collected']<=scheduler['collected']]
        initial_actor_sha=package['initial_actor_sha']
        write(out/f'resume-{time.time_ns()}.json',dict(collected=scheduler['collected'],exact_learner_and_Task_restore=True))
    else:
        random.seed(681117);np.random.seed(681117)
        with torch.enable_grad(): agent,identity=build(aux_scale=scale,compile_execution=False)
        assert identity['source_sha256']=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
        assert all(not x['state'] for x in agent.optimizer_dict.values())
        write(out/'identity.json',identity)
        replay=PhaseReplay(capacity=min(1000000,2*n),quotas=plan['replay_quotas'])
        buffers=dict(train=replay,expert_slicer=load_reference(reference))
        first=episode(plan,reset,0,args['arm']);live.start(first)
        scheduler=dict(collected=0,updates=0,source_index=1,
                       episodes=[dict(spec=first,controls=0,complete=False)])
        segments=[];blocks=[]
        initial_actor_sha=digest(agent._model._actor.state_dict())
        committer=RoundCommitter(out/'store',total_rows=n,milestone_rows=milestones)
    def full_state():
        state=learner_state(agent,step=scheduler['updates'],replay_segments=segments,
            replay_capacity=buffers['train'].capacity,replay_idx=buffers['train']._idx,
            reference=reference,replay_quotas=buffers['train'].quotas)
        return dict(learner=state,task=live.snapshot(),scheduler=copy.deepcopy(scheduler),
                    initial_actor_sha=initial_actor_sha,formal_training=True,arm=args['arm'],N=n)
    if not args['resume']:
        save=committer.commit(0,full_state())
        write(out/'step0.json',dict(status='FRESH_ORIGINAL_STEP0_FULLY_COMMITTED',save=save,
            original_weights=True,fresh_five_Adam=True,fresh_replay=True,no_S0_learner_inherited=True,
            plan_sha256=file_sha(plan_path),reset_sha256=file_sha(reset_path)))
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
                if live.done:
                    spec=episode(plan,reset,scheduler['source_index'],args['arm'])
                    scheduler['source_index']+=1;live.start(spec)
                    scheduler['episodes'].append(dict(spec=spec,controls=0,complete=False))
                before=live.observation();z=live.command()
                with torch.no_grad(): action=agent.act(before,z,mean=False)
                rows.append(live.step(action,before,z))
                physics.extend(copy.deepcopy(e.trace[-4:]))
                scheduler['collected']+=1
                scheduler['episodes'][-1]['controls']+=1
                scheduler['episodes'][-1]['complete']=live.done
            collect_s=time.monotonic()-started
            data=cat(rows);buffers['train'].extend(data)
            rowpath=out/f'actual-rows-{scheduler["collected"]:06d}.pt'
            assert not rowpath.exists();torch.save(data,rowpath)
            segments.append(dict(path=str(rowpath),sha256=file_sha(rowpath)))
            persist_physics(str(scheduler['collected']))
            collected=scheduler['collected']
            count=64 if collected==8192 else 8 if collected>8192 else 0
            agent.actor_enabled=collected>8192
            update_s=[];metrics_record=[]
            for _ in range(count):
                torch.cuda.synchronize();t=time.monotonic()
                with torch.enable_grad(): metrics=agent.update(buffers,scheduler['updates'])
                torch.cuda.synchronize();update_s.append(time.monotonic()-t)
                assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
                metrics_record.append({k:float(v.detach().mean().cpu()) if torch.is_tensor(v) else v for k,v in metrics.items()})
                scheduler['updates']+=1
            if collected<=8192:assert digest(agent._model._actor.state_dict())==initial_actor_sha
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
                projected_output=lock['stage_storage']['logical_bytes']+1.25*(
                    15*peak_pack+2*n*(row_bytes+trace_bytes)/collected+8*2**30)
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
    a=p.parse_args()
    if a.resume: assert a.output.exists()
    else:a.output.mkdir(parents=True,exist_ok=False)
    from resource_guard import check
    write(a.output/f'preflight-{time.time_ns()}.json',check(STAGE))
    prior=sum(json.loads(f.read_text())['wall_s'] for f in a.output.glob('cost-*.json'))
    assert prior<43200
    start=time.monotonic()
    os.environ.update(CTFT_FORMAL_ARGS=json.dumps(dict(arm=a.arm,resume=a.resume)),
        CTFT_PRIOR_WALL=str(prior),CTFT_RUN_START_MONO=str(start),
        BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(start+43200-prior))
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
    m10_stage.run_study=study;m10_physics.OUT=a.output
    try:m10_physics.main()
    finally:write(a.output/f'cost-{time.time_ns()}.json',dict(wall_s=time.monotonic()-start,
        physics=counts,formal_training=True,arm=a.arm))


if __name__=='__main__':main()
