"""Actual fresh8192 warm-up then two full source-task/update/commit blocks.

S0 disposable only. Formal U/T never load this Actor, replay or Adam. Exact
complete checkpoint includes actual causal rows and the generalized K callback.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def study(e,original,contract,out):
    import torch
    from agent import build
    from training_live import TrainingLive
    from training_data import PhaseReplay,episode
    from native_reference import load_reference
    from contact_replay import cat
    from learning_checkpoint import digest,file_sha,one_update
    from gradient_housekeeping import clear_consumed
    from packed_checkpoint import PackedTensorStore,durable
    from cas_checkpoint import learner_state,restore_learner
    from f1_mjdata_contract import canonical,metadata
    from locomotion_rollout import UnsafePhysics
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    speed(e);(out/'backend-prefix').mkdir();backend(e,original,out/'backend-prefix');reset_common(e,original,original.act)
    args=json.loads(os.environ['CTFT_ROUND_ARGS']);phase=args['mode'];target=10240
    reset=json.loads((STAGE/'locks/source-reset-domain-v1.json').read_text())
    plan=json.loads((STAGE/'locks/exogenous-training-plan-v2.json').read_text())
    records=json.loads((STAGE/'runs/s0-source-admission-01/result.json').read_text())['train']
    coeff=json.loads((STAGE/'runs/s0-online-probe-01/agent-identity.json').read_text())['original_training_config']['aux_rewards_scaling']
    live=TrainingLive(e,STAGE,coeff,records)
    if phase=='reference':
        scale=json.loads((STAGE/'locks/common-aux-scale-v1.json').read_text())['value']
        with torch.enable_grad():agent,identity=build(aux_scale=scale,compile_execution=False)
        write(out/'identity.json',identity)
        replay=PhaseReplay(capacity=target+1024,quotas=plan['replay_quotas'])
        reference=STAGE/'runs/s0-cas-reference-02/reference';buffers=dict(train=replay,expert_slicer=load_reference(reference))
        store=PackedTensorStore(out/'store');source_index=0;updates=0;collected=0;segments=[];episodes=[];blocks=[];base=None
        frozen=digest(agent._model._actor.state_dict())
        while collected<target:
            start=time.monotonic();rows=[]
            for i in range(1024):
                assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
                if live.done:
                    spec=episode(plan,reset,source_index,'U');source_index+=1;live.start(spec)
                    episodes.append(dict(spec=spec,controls=0,complete=False))
                before=live.observation();z=live.command()
                with torch.no_grad():action=agent.act(before,z,mean=False)
                try:row=live.step(action,before,z)
                except UnsafePhysics as exc:
                    torch.save(dict(Task=live.snapshot(),partial_rows=rows,requested_action=action.detach().cpu(),boundary=exc.row),out/'hard-boundary.pt')
                    write(out/'hard-boundary.json',dict(status='REAL_PHYSICAL_STOP',collected=collected+i,updates=updates,source=live.spec))
                    raise
                rows.append(row);episodes[-1]['controls']+=1
                if live.done:episodes[-1]['complete']=True
            data=cat(rows);replay.extend(data);collected+=1024
            path=out/f'actual-rows-{collected:06d}.pt';torch.save(data,path);segments.append(dict(path=str(path),sha256=file_sha(path)))
            collect_s=time.monotonic()-start
            count=64 if collected==8192 else 8 if collected>8192 else 0
            agent.actor_enabled=collected>8192;update_s=[]
            for _ in range(count):
                torch.cuda.synchronize();start_update=time.monotonic()
                with torch.enable_grad():metrics=agent.update(buffers,updates)
                torch.cuda.synchronize();update_s.append(time.monotonic()-start_update);updates+=1
                assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
            if collected<=8192:assert digest(agent._model._actor.state_dict())==frozen,'Actor changed in8192 warmup'
            housekeeping=clear_consumed(agent) if any(p.grad is not None for p in agent._model.parameters()) else dict(noop_all_gradients_already_None=True)
            state=learner_state(agent,step=updates,replay_segments=segments,replay_capacity=replay.capacity,replay_idx=replay._idx,
                reference=reference,replay_quotas=replay.quotas)
            package=dict(learner=state,task=live.snapshot(),scheduler=dict(collected=collected,updates=updates,source_index=source_index,episodes=episodes),
                housekeeping=housekeeping,formal_training=False)
            # Probe keeps only snapshots necessary to measure changed full
            # learner commits; every1024 actual rows remain immutable on disk.
            save=None
            if collected>=8192:
                base,save=store.save(f'round-{collected}',package,base=base,anchor=collected==8192)
            blocks.append(dict(collected=collected,collect_s=collect_s,actual_update_count=count,update_s=update_s,save=save,
                total_s=time.monotonic()-start,phase_unique={k:len(v) for k,v in replay.members.items()}))
            write(out/'blocks.json',blocks);write(out/'episodes.json',episodes)
        write(out/'probe-summary.json',dict(status='FRESH8192_AND_TWO_JOINT_ROUNDS_COMMITTED',formal_training=False,
            actual_rows=collected,updates=updates,blocks=blocks,first7_warm_collection_blocks_no_core_snapshot=True,
            no_formal_N_locked=True,not_full_formal_retention_driver=True))
        expected_step=updates
    else:
        store=PackedTensorStore(Path(args['source']));package=store.load('round-10240')
        with torch.enable_grad():agent,buffers=restore_learner(package['learner'])
        live.restore(package['task']);expected_step=package['learner']['step']
        assert digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data)))==digest(canonical(package['task'],metadata(e.sim.model,e.sim.data)))
    rows=[];e.trace=[]
    for _ in range(12):
        assert not live.done,'Probe ended at episode boundary; no invented continuation'
        before=live.observation();z=live.command()
        with torch.no_grad():action=agent.act(before,z,mean=False)
        rows.append(live.step(action,before,z))
    buffers['train'].extend(cat(rows))
    with torch.enable_grad():u=one_update(agent,buffers,expected_step)
    answer=dict(rows=digest(rows),physics=digest(e.trace),Task=digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data))),
        learner={k:u[k] for k in ['state','sample_trace','metrics']})
    write(out/'continuation.json',answer)
    if phase=='resume':
        expected=json.loads((Path(args['reference'])/'continuation.json').read_text());bad=[k for k in answer if answer[k]!=expected[k]]
        write(out/'comparison.json',dict(status='PASS' if not bad else 'FAIL',different=bad,actual_PhaseReplay_TrainingLive_native_update=True))
        assert not bad,('New source/phase replay checkpoint mismatch',bad)
    return dict(status='ROUND_PROBE_REFERENCE_COMPLETE' if phase=='reference' else 'ROUND_PROBE_FULL_RESTORE_EXACT',formal_training=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['reference','resume'],required=True)
    p.add_argument('--source');p.add_argument('--reference');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from resource_guard import check
    check(STAGE)
    os.environ.update(CTFT_ROUND_ARGS=json.dumps(dict(mode=a.mode,source=a.source,reference=a.reference)),
        BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1200))
    write(a.output/'launch.json',dict(pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),started_unix=time.time()))
    import mujoco
    raw=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));counts['reserved']+=n;assert counts['reserved']<=50000
        result=raw(*args,**kw);counts['actual']+=n;return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output;begun=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'cost.json',dict(wall_s=time.monotonic()-begun,physics=counts,formal_updates=0))


if __name__=='__main__':main()
