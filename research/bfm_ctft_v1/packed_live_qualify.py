"""Actual complete-state packed recovery; no formal training or reused init.

Engineering only: continue an explicitly disposable S0 online probe. Original
raw package is held in-memory, two packed generations are written, and a new
process must reproduce actual released-phase physics and a native CPR update.
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
    from packed_checkpoint import PackedTensorStore
    from cas_checkpoint import learner_state,restore_learner
    from learning_checkpoint import file_sha,fingerprint,one_update,digest
    from contact_live import ContactLive
    from contact_replay import cat
    from f1_mjdata_contract import canonical,metadata
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    speed(e);(out/'backend-prefix').mkdir();backend(e,original,out/'backend-prefix');reset_common(e,original,original.act)
    args=json.loads(os.environ['CTFT_PACKED_ARGS']);mode=args['mode']
    root=out/'store' if mode=='reference' else Path(args['store']);store=PackedTensorStore(root)
    if mode=='reference':
        source=STAGE/'runs/s0-online-probe-01/checkpoint-2';raw=source/'learner'
        m=json.loads((raw/'manifest.json').read_text())
        for name,v in m['files'].items():assert file_sha(raw/name)==v['sha256']
        state=torch.load(raw/'learner.pt',map_location='cpu',weights_only=False)
        state['model']=torch.load(raw/'model.pt',map_location='cpu',weights_only=False)
        state['initial_fingerprint']=m['expected_initial']
        replay_path=out/'actual-initial-replay.pt';torch.save(state.pop('replay'),replay_path)
        segments=[dict(path=str(replay_path),sha256=file_sha(replay_path))]
        state['replay_segments']=segments;state['reference']=str(STAGE/'runs/s0-cas-reference-02/reference')
        state.pop('expert_groups');state.pop('replay_full')
        with torch.enable_grad():agent,buffers=restore_learner(state)
        task=torch.load(source/'task.pt',weights_only=False)
        step=int(state['step']);write(out/'raw-provenance.json',dict(
            purpose='Disposable S0 snapshot, never formal U/T initializer',source_manifest_sha256=file_sha(source/'manifest.json'),
            reference_manifest_sha256=file_sha(Path(state['reference'])/'manifest.json'),
            initial_model_and_all_Adam_exact=True,source_step=step,source_actual_rows=buffers['train']._idx))
    else:
        pack=store.load('round-3');state=pack['learner'];task=pack['task'];step=int(state['step'])
        with torch.enable_grad():agent,buffers=restore_learner(state)
    coefficients=json.loads((STAGE/'runs/s0-online-probe-01/agent-identity.json').read_text())['original_training_config']['aux_rewards_scaling']
    live=ContactLive(e,STAGE,coefficients);live.restore(task)
    assert fingerprint(agent)==state['initial_fingerprint']
    assert digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data)))==digest(canonical(task,metadata(e.sim.model,e.sim.data)))
    def run(n,interrupt=False):
        rows=[];e.trace=[]
        for _ in range(n):
            before=live.observation();z=live.command()
            with torch.no_grad():action=agent.act(before,z,mean=False)
            rows.append(live.step(action,before,z))
        if interrupt:
            write(out/'interrupted-uncommitted.json',dict(actual_rows=n,committed_rows=buffers['train']._idx,
                latest_sha256=file_sha(root/'latest.json'),actual_physics=len(e.trace),no_update=True))
            import signal
            signal.raise_signal(signal.SIGINT)
            raise AssertionError('SIGINT did not interrupt')
        buffers['train'].extend(cat(rows))
        with torch.enable_grad():updated=one_update(agent,buffers,step)
        return dict(actual_rows_sha256=digest(rows),physics_sha256=digest(e.trace),
            full_task_sha256=digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data))),
            learner={k:updated[k] for k in ['state','sample_trace','metrics']},
            committed_replay_rows=buffers['train']._idx),cat(rows)
    if mode=='interrupt':run(5,interrupt=True)
    if mode=='reference':
        reference=state['reference']
        def package():return dict(learner=learner_state(agent,step=step,replay_segments=segments,
            replay_capacity=buffers['train'].capacity,replay_idx=buffers['train']._idx,reference=reference),task=live.snapshot(),
            scheduler=dict(committed_rows=buffers['train']._idx,step=step,disposable=True))
        before=fingerprint(agent);base,stats=store.save('round-2',package(),anchor=True);assert fingerprint(agent)==before
        write(out/'save-2.json',stats)
        _,new=run(12);step+=1
        replay_path=out/'actual-next-replay.pt';torch.save(new,replay_path);segments.append(dict(path=str(replay_path),sha256=file_sha(replay_path)))
        pack=package();before=fingerprint(agent);_,stats=store.save('round-3',pack,base=base);assert before==fingerprint(agent)
        write(out/'save-3.json',stats)
        assert store.load('round-2')['scheduler']['step']==step-1
        latest=(root/'latest.json').read_bytes()
        try:store.save('injected-write-failure',pack,base=base,inject='after_first_chunk')
        except RuntimeError as exc:assert 'INJECTED_DURING_ACTUAL_FULL_STATE_WRITE' in str(exc)
        else:raise AssertionError('Full-state transaction failure did not fire')
        assert (root/'latest.json').read_bytes()==latest
        assert digest(store.load('round-3'))==digest(pack)
        write(out/'failure-injection.json',dict(full_state_write_interrupted=True,latest_unchanged=True,full_committed_package_exact=True))
    answer,_=run(12);write(out/'continuation.json',answer)
    if mode=='resume':
        expected=json.loads((Path(args['reference'])/'continuation.json').read_text())
        different=[k for k in answer if answer[k]!=expected[k]]
        write(out/'comparison.json',dict(status='PASS' if not different else 'FAIL',different=different,
            initial_full_Task_and_learner_exact=True,actual_released_physics_and_native_CPR_update_exact=not different,
            formal_training=False))
        assert not different,('Packed fresh-process continuation differs',different)
    return dict(status='PACKED_REFERENCE_COMPLETE' if mode=='reference' else 'PACKED_FRESH_PROCESS_EXACT',
        formal_training=False,actual_new_rows=24 if mode=='reference' else 12,complete_updates=2 if mode=='reference' else 1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['reference','interrupt','resume'],required=True)
    p.add_argument('--store');p.add_argument('--reference');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CTFT_PACKED_ARGS=json.dumps(dict(mode=a.mode,store=a.store,reference=a.reference)),
        BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1800))
    from resource_guard import check
    check(STAGE)
    write(a.output/'launch.json',dict(pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),started_unix=time.time()))
    import mujoco
    raw=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kwargs):
        n=int(kwargs.get('nstep',1));counts['reserved']+=n;assert counts['reserved']<=6000
        result=raw(*args,**kwargs);counts['actual']+=n;return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output;begun=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'cost.json',dict(wall_s=time.monotonic()-begun,physics=counts,formal_updates=0))


if __name__=='__main__':main()
