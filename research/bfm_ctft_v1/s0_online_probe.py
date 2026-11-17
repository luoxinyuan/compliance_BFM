"""Bounded one-slot real collection + native update + full commit benchmark.

Disposable S0 only: no formal source sampler, training size or scientific result
is inferred from this three-block wiring/throughput run. Worker scaling remains
a separate gate, not a claim of implemented parallel physics.
"""
import argparse
import hashlib
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


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def study(e,original,contract,out):
    import numpy as np
    import torch
    from agent import build
    from contact_replay import build_buffers,Sequences,cat
    from contact_live import ContactLive
    from learning_checkpoint import atomic_save,fingerprint,file_sha,digest
    from humanoidverse.agents.buffers.transition import DictBuffer
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from locomotion_rollout import UnsafePhysics
    speed(e);(out/'backend-prefix').mkdir();backend(e,original,out/'backend-prefix')
    reset_common(e,original,original.act)
    lock=json.loads((STAGE/'locks/common-aux-scale-v1.json').read_text())
    torch.set_float32_matmul_precision('high')
    with torch.enable_grad():agent,identity=build(aux_scale=lock['value'],compile_execution=False)
    write(out/'agent-identity.json',identity)
    _,groups,refs=build_buffers(ROOT,identity['original_training_config']['aux_rewards_scaling'])
    write(out/'reference-provenance.json',refs)
    replay=DictBuffer(capacity=8192,device='cpu')
    buffers=dict(train=replay,expert_slicer=Sequences(groups))
    live=ContactLive(e,STAGE,identity['original_training_config']['aux_rewards_scaling'])
    plan=json.loads((out/'protocol.json').read_text())
    episodes=[];schedule_index=0;blocks=[];update_count=0;collected=0
    model_before={k:p.detach().clone() for k,p in agent._model._actor.named_parameters()}
    for block in range(3):
        begun=time.monotonic();reset_s=0.;rows=[];episode=episodes[-1] if episodes else None
        try:
            for _ in range(1024):
                assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
                if live.done:
                    start=time.monotonic();spec=plan['schedule'][schedule_index];schedule_index+=1
                    live.start(spec);reset_s+=time.monotonic()-start
                    episode=dict(index=schedule_index-1,spec=spec,actual_controls=0,complete=False,
                        active_ticks=0,released_ticks=0,mode=spec['mode'],parent_identity='standing' if spec['mode'] else f"original_walk_case_{spec['case']}")
                    episodes.append(episode)
                before=live.observation();z=live.command()
                with torch.no_grad():action=agent.act(before,z,mean=False)
                assert torch.isfinite(action).all()
                row=live.step(action,before,z)
                rows.append(row);collected+=1;episode['actual_controls']+=1
                episode['active_ticks']+=sum(bool(p['active']) for p in e.trace[-4:])
                episode['released_ticks']+=sum(live.spec['mode']==1 and p['time']>=e.tether.ctft_partner_spec['release_s'] for p in e.trace[-4:])
                if live.done:episode['complete']=True
        except UnsafePhysics as exc:
            # A real safety event is not an engineering retry or a skipped source.
            torch.save(dict(task=live.snapshot(),partial_actual_rows=rows,episodes=episodes,boundary=exc.row),out/'hard-boundary.pt')
            write(out/'boundary.json',dict(status='REAL_HARD_PHYSICS_STOP',row=exc.row,collected=collected,updates=update_count,episodes=episodes))
            raise
        torch.cuda.synchronize();collect_s=time.monotonic()-begun
        replay.extend(cat(rows))
        agent.actor_enabled=True
        timings=[];metrics=[]
        for j in range(8):
            torch.cuda.synchronize();start=time.monotonic()
            with torch.enable_grad():m=agent.update(buffers,update_count)
            torch.cuda.synchronize();timings.append(time.monotonic()-start);update_count+=1
            assert all(torch.isfinite(v).all() for v in m.values() if torch.is_tensor(v))
            metrics.append({k:float(v.detach().cpu()) for k,v in m.items() if torch.is_tensor(v) and v.numel()==1})
        # Two complete immutable commits, including scheduler/live source phase.
        save_s=0.;saved_bytes=0
        if block<2:
            start=time.monotonic();pending=out/f'checkpoint-{block+1}.pending';pending.mkdir()
            physical=live.snapshot()
            torch.save(physical,pending/'task.pt')
            write(pending/'scheduler.json',dict(collected=collected,update_count=update_count,block=block,
                schedule_index=schedule_index,episodes=episodes,logical_batch=1024,num_envs=1,completed_in_round=1024))
            lm=atomic_save(agent,buffers,groups,pending/'learner',step=update_count)
            files={p.name:dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in [pending/'task.pt',pending/'scheduler.json']}
            files['learner/manifest.json']=dict(sha256=file_sha(pending/'learner/manifest.json'),bytes=(pending/'learner/manifest.json').stat().st_size)
            saved_bytes=sum(p.stat().st_size for p in pending.rglob('*') if p.is_file())
            write(pending/'manifest.json',dict(complete=True,protocol='CTFT_S0_ONLINE_FULL_STATE_V1',files=files,
                initial_learner=lm['expected_initial'],physical_sha256=digest(physical),
                source_callback='contact_live.ContactLive.guarded + explicit native/contact measure binding',
                bytes=saved_bytes,formal_training=False))
            for p in pending.rglob('*'):
                if p.is_file():
                    with p.open('rb') as f:os.fsync(f.fileno())
            os.rename(pending,out/f'checkpoint-{block+1}')
            save_s=time.monotonic()-start
        record=dict(block=block,actual_new_rows=1024,collected=collected,unique_replay_rows=len(replay),
            collect_s_including_source_restore=collect_s,restore_source_s=reset_s,
            complete_joint_update_s=timings,updates=8,update_metrics=metrics,
            save_s=save_s,saved_bytes=saved_bytes,
            actor_parameter_max_abs_delta=max(float((p-model_before[k]).abs().max()) for k,p in agent._model._actor.named_parameters()),
            total_block_s=time.monotonic()-begun,
            cuda_allocated=torch.cuda.max_memory_allocated(),cuda_reserved=torch.cuda.max_memory_reserved())
        blocks.append(record);write(out/'blocks.json',blocks);write(out/'episodes.json',episodes)
        free,total=torch.cuda.mem_get_info()
        assert torch.cuda.max_memory_reserved()<=.88*total and free>=max(6*2**30,.08*total)
        assert __import__('shutil').disk_usage(out).free>=100*2**30
        print('CTFT_S0_ONLINE_BLOCK',block,record['total_block_s'],collect_s,sum(timings),save_s,flush=True)
    return dict(status='S0_SINGLE_SLOT_ONLINE_THROUGHPUT_MEASURED',disposable=True,formal_training=False,
        actual_new_rows=collected,complete_updates=update_count,num_envs=1,
        completed_episodes=sum(p['complete'] for p in episodes),episodes=episodes,blocks=blocks,
        worker_scaling_qualified=False,formal_N_locked=False,
        initialization='original source + fresh aux and Adam; no probe model inherited')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    schedule=[dict(mode=1,force=True),dict(mode=1,force=False),dict(mode=0,case=1),
        dict(mode=1,force=True),dict(mode=1,force=True),dict(mode=0,case=2),dict(mode=1,force=True)]
    write(a.output/'protocol.json',dict(scope='S0 disposable complete-cost probe, NOT formal U/T and NOT formal warm-up',
        schedule=schedule,blocks=3,new_actual_rows_per_block=1024,complete_updates_per_block=8,logical_batch=1024,
        num_envs=1,seed=681117,actor_std='matching original config; original mean=False sampling',
        frozen_scale_lock='common-aux-scale-v1.json',wall_cap_s=1800,physics_cap=20000,
        original_algorithm_changes=False,training_data='newly executed rows only; native CPR reference separately archived actual g0'))
    write(a.output/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1800))
    import mujoco
    raw=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));counts['reserved']+=n;assert counts['reserved']<=20000
        x=raw(*args,**kw);counts['actual']+=n;return x
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output
    begun=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'cost.json',dict(wall_s=time.monotonic()-begun,physics=counts,formal_training_transitions=0))


if __name__=='__main__':main()
