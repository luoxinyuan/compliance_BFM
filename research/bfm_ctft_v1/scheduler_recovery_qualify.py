"""Separate scheduler-count erratum and actual cross-reset continuation.

Preserves the original10252 pack. Only episode counters are corrected; all
model/Adam/replay/RNG/Task tensors retain exact hashes. Then physically finish
the actual episode, reset from the real next source, sample12 more controls
and perform one full native update. Reference/resume use the same fixed pack.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def study(e,original,contract,out):
    import torch
    from packed_checkpoint import PackedTensorStore
    from cas_checkpoint import restore_learner
    from production_commit import validate_scheduler
    from learning_checkpoint import digest,one_update
    from training_live import TrainingLive
    from training_data import episode
    from contact_replay import cat
    from f1_mjdata_contract import canonical,metadata
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from locomotion_rollout import UnsafePhysics
    speed(e)
    (out/'backend-prefix').mkdir()
    backend(e,original,out/'backend-prefix')
    reset_common(e,original,original.act)
    args=json.loads(os.environ['CTFT_SCHEDULER_ARGS'])
    origin=PackedTensorStore(STAGE/'runs/s0-production-reference-01/store')
    if args['mode']=='reference':
        x=origin.load('round-10252')
        before=digest({k:v for k,v in x.items() if k!='scheduler'})
        old_count=sum(q['controls'] for q in x['scheduler']['episodes'])
        assert x['scheduler']['collected']-old_count==12
        x['scheduler']['episodes'][-1]['controls']=x['task']['live_control']
        x['scheduler']['episodes'][-1]['complete']=x['task']['done']
        validate_scheduler(x)
        assert digest({k:v for k,v in x.items() if k!='scheduler'})==before
        store=PackedTensorStore(out/'store')
        base=copy.deepcopy(origin.metadata('round-10252'))
        def rebase(v):
            if isinstance(v,dict):
                if 'file' in v and 'raw_sha256' in v:
                    v['file']=os.path.relpath(origin.root/v['file'],store.root)
                for q in v.values():
                    rebase(q)
            elif isinstance(v,(list,tuple)):
                for q in v:
                    rebase(q)
        rebase(base)
        _,cost=store.save('round-10252',x,base=base,anchor=True)
        assert cost['newly_written_blob_bytes']==0
        write(out/'scheduler-erratum.json',dict(status='COUNTER_ONLY_FIXED',
            old_total_episode_controls=old_count,new_total_episode_controls=x['scheduler']['collected'],
            identical_learner_Task_replay_RNG=True,old_package_preserved=True,save=cost,
            scope='Disposable S0 qualifier only; no formalU/T data or old signed scientific result changed'))
    else:
        store=PackedTensorStore(args['source'])
        x=store.load('round-10252')
        validate_scheduler(x)
    with torch.enable_grad():
        agent,buffers=restore_learner(x['learner'])
    coefficients=json.loads((STAGE/'runs/s0-online-probe-01/agent-identity.json').read_text())['original_training_config']['aux_rewards_scaling']
    records=json.loads((STAGE/'runs/s0-source-admission-01/result.json').read_text())['train']
    live=TrainingLive(e,STAGE,coefficients,records)
    live.restore(x['task'])
    assert digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data)))==digest(canonical(x['task'],metadata(e.sim.model,e.sim.data)))
    plan=json.loads((STAGE/'locks/exogenous-training-plan-v2.json').read_text())
    reset=json.loads((STAGE/'locks/source-reset-domain-v1.json').read_text())
    scheduler=copy.deepcopy(x['scheduler'])
    count=live.controls-live.i+12
    assert count<900 and buffers['train']._idx+count<buffers['train'].capacity
    rows=[]
    physics=[]
    reset_events=[]
    for _ in range(count):
        assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
        if live.done:
            spec=episode(plan,reset,scheduler['source_index'],'U')
            scheduler['source_index']+=1
            live.start(spec)
            scheduler['episodes'].append(dict(spec=spec,controls=0,complete=False))
            reset_events.append(dict(after_actual_rows=len(rows),next_source=spec))
        before=live.observation()
        z=live.command()
        with torch.no_grad():
            action=agent.act(before,z,mean=False)
        try:
            rows.append(live.step(action,before,z))
        except UnsafePhysics as exc:
            torch.save(dict(Task=live.snapshot(),actual_rows=rows,physics=physics,boundary=exc.row),out/'hard-boundary.pt')
            raise
        physics.extend(copy.deepcopy(e.trace[-4:]))
        scheduler['collected']+=1
        scheduler['episodes'][-1]['controls']+=1
        scheduler['episodes'][-1]['complete']=live.done
    assert reset_events
    buffers['train'].extend(cat(rows))
    with torch.enable_grad():
        u=one_update(agent,buffers,x['learner']['step'])
    scheduler['updates']+=1
    assert scheduler['collected']==buffers['train']._idx
    validate_scheduler(dict(scheduler=scheduler,learner=dict(step=x['learner']['step']+1),task=live.snapshot()))
    torch.save(dict(actual_rows=rows,physics=physics,reset_events=reset_events,scheduler=scheduler),out/'actual-cross-reset-evidence.pt')
    answer=dict(rows=digest(rows),physics=digest(physics),scheduler=digest(scheduler),
        Task=digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data))),
        learner={k:u[k] for k in ['state','sample_trace','metrics']})
    write(out/'continuation.json',answer)
    if args['mode']=='resume':
        expected=json.loads((Path(args['reference'])/'continuation.json').read_text())
        bad=[k for k in answer if answer[k]!=expected[k]]
        write(out/'comparison.json',dict(status='PASS' if not bad else 'FAIL',different=bad,
              real_controls=count,actual_full_episode_reset=len(reset_events),full_native_updates=1,formal_training=False))
        assert not bad,bad
    return dict(status='SCHEDULER_REFERENCE_COMPLETE' if args['mode']=='reference' else 'SCHEDULER_CROSS_RESET_RESTORE_EXACT',
                actual_controls=count,actual_full_episode_reset=len(reset_events),actual_updates=1,formal_training=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['reference','resume'],required=True)
    p.add_argument('--source')
    p.add_argument('--reference')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    from resource_guard import check
    check(STAGE)
    os.environ.update(CTFT_SCHEDULER_ARGS=json.dumps(dict(mode=a.mode,source=a.source,reference=a.reference)),
        BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1200))
    write(a.output/'launch.json',dict(pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),started_unix=time.time()))
    import mujoco
    raw=mujoco.mj_step
    counts=dict(actual=0,reserved=0)
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));counts['reserved']+=n
        assert counts['reserved']<=6000
        result=raw(*args,**kw);counts['actual']+=n
        return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study
    m10_physics.OUT=a.output
    start=time.monotonic()
    try:
        m10_physics.main()
    finally:
        write(a.output/'cost.json',dict(wall_s=time.monotonic()-start,physics=counts,formal_updates=0))


if __name__=='__main__':
    main()
