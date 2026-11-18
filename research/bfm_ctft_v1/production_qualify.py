"""Actual Task/PhaseReplay/native-update qualification of round retention.

Adopts immutable S0 packs without counting them as new samples or updates.
One additional actual12-control/update checkpoint exercises the real round
policy. Fresh-process continuation must match another12 controls/full update.
Formal U/T never inherit this engineering learner.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'research_notes'), str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import m11_ipc


def write(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')


def study(e, original, contract, out):
    import torch
    from production_commit import RoundCommitter
    from cas_checkpoint import learner_state, restore_learner
    from packed_checkpoint import PackedTensorStore
    from training_live import TrainingLive
    from contact_replay import cat
    from learning_checkpoint import digest, one_update, file_sha
    from gradient_housekeeping import clear_consumed
    from f1_mjdata_contract import canonical, metadata
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    speed(e)
    (out/'backend-prefix').mkdir()
    backend(e, original, out/'backend-prefix')
    reset_common(e, original, original.act)
    args = json.loads(os.environ['CTFT_PRODUCTION_QUALIFY_ARGS'])
    origin = STAGE/'runs/s0-round-probe-01/store'
    if args['mode'] == 'reference':
        committer = RoundCommitter(out/'store', total_rows=11264,
                                  milestone_rows=[8192, 11264], initial_rows=8192)
        imports = []
        for rows in [8192, 9216, 10240]:
            imports.append(dict(rows=rows, cost=committer.adopt(rows, origin, f'round-{rows}')))
        write(out/'adopted-actual-packs.json', imports)
    else:
        committer = RoundCommitter(Path(args['source']), total_rows=11264,
                                  milestone_rows=[8192, 11264], initial_rows=8192, create=False)
    package = committer.load_latest()
    with torch.enable_grad():
        agent, buffers = restore_learner(package['learner'])
    coefficients = json.loads((STAGE/'runs/s0-online-probe-01/agent-identity.json').read_text())['original_training_config']['aux_rewards_scaling']
    records = json.loads((STAGE/'runs/s0-source-admission-01/result.json').read_text())['train']
    live = TrainingLive(e, STAGE, coefficients, records)
    live.restore(package['task'])
    assert digest(canonical(live.snapshot(), metadata(e.sim.model, e.sim.data))) == digest(canonical(package['task'], metadata(e.sim.model, e.sim.data)))
    step = package['learner']['step']
    scheduler = package['scheduler']
    segments = list(package['learner']['replay_segments'])

    def actual_next():
        nonlocal step
        rows = []
        e.trace = []
        for _ in range(12):
            assert time.monotonic() < float(os.environ['CCL_DEADLINE_MONOTONIC'])
            assert not live.done
            before = live.observation()
            z = live.command()
            with torch.no_grad():
                action = agent.act(before, z, mean=False)
            rows.append(live.step(action, before, z))
        data = cat(rows)
        buffers['train'].extend(data)
        with torch.enable_grad():
            u = one_update(agent, buffers, step)
        step += 1
        return dict(rows=digest(rows), physics=digest(e.trace),
                    Task=digest(canonical(live.snapshot(), metadata(e.sim.model, e.sim.data))),
                    learner={k:u[k] for k in ['state', 'sample_trace', 'metrics']}), data

    if args['mode'] == 'reference':
        first, data = actual_next()
        write(out/'first-actual-continuation.json', first)
        path = out/'actual-next-12rows.pt'
        torch.save(data, path)
        segments.append(dict(path=str(path), sha256=file_sha(path)))
        clear = clear_consumed(agent)
        scheduler = dict(scheduler, collected=10252, updates=step)
        scheduler['episodes'][-1]['controls']=live.i
        scheduler['episodes'][-1]['complete']=live.done
        new = dict(learner=learner_state(agent, step=step, replay_segments=segments,
                       replay_capacity=buffers['train'].capacity, replay_idx=buffers['train']._idx,
                       reference=package['learner']['reference'], replay_quotas=buffers['train'].quotas),
                   task=live.snapshot(), scheduler=scheduler, housekeeping=clear,
                   formal_training=False, engineering_retention_qualification=True)
        measured = committer.commit(10252, new)
        assert (committer.root/'round-9216/retired.json').exists()
        assert not (committer.root/'round-9216/state.pkl').exists()
        assert (origin/'round-9216/state.pkl').exists(), 'Original engineering evidence must remain'
        write(out/'actual-retention.json', dict(status='FULL_ACTUAL_ROUND_COMMIT_AND_RETIRE_PASS',
              save=measured, original_inputs_preserved=True, latest_two_complete=True,
              retired_only_successful_alias=True, no_old_asset_deleted=True))
    answer, _ = actual_next()
    write(out/'continuation.json', answer)
    if args['mode'] == 'resume':
        expected = json.loads((Path(args['reference'])/'continuation.json').read_text())
        bad = [k for k in answer if answer[k] != expected[k]]
        write(out/'comparison.json', dict(status='PASS' if not bad else 'FAIL', different=bad,
              actual_Task_PhaseReplay_round_driver_full_update=True, formal_training=False))
        assert not bad, bad
    return dict(status='PRODUCTION_ROUND_REFERENCE_COMPLETE' if args['mode']=='reference' else 'PRODUCTION_ROUND_RESTORE_EXACT',
                formal_training=False, actual_new_rows=24 if args['mode']=='reference' else 12,
                actual_new_agent_updates=2 if args['mode']=='reference' else 1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['reference', 'resume'], required=True)
    p.add_argument('--source')
    p.add_argument('--reference')
    a=p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from resource_guard import check
    check(STAGE)
    from stage_storage import footprint
    disk=footprint(STAGE)
    write(a.output/'stage-storage-before.json', disk)
    margin=6*2**30 if a.mode=='reference' else 64*2**20
    assert disk['logical_bytes']+margin < 120*2**30, 'Actual stage output plus conservative next-write reserve exceeds120GiB'
    os.environ.update(CTFT_PRODUCTION_QUALIFY_ARGS=json.dumps(dict(mode=a.mode, source=a.source, reference=a.reference)),
                      BFM_RECOVERY_OUTPUT=str(a.output), CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1800))
    write(a.output/'launch.json', dict(pid=os.getpid(), runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(), started_unix=time.time()))
    import mujoco
    raw=mujoco.mj_step
    counts=dict(actual=0, reserved=0)
    def counted(*args, **kw):
        n=int(kw.get('nstep',1))
        counts['reserved']+=n
        assert counts['reserved']<=6000
        result=raw(*args, **kw)
        counts['actual']+=n
        return result
    mujoco.mj_step=counted
    import m10_stage, m10_physics
    m10_stage.run_study=study
    m10_physics.OUT=a.output
    begun=time.monotonic()
    try:
        m10_physics.main()
    finally:
        write(a.output/'cost.json', dict(wall_s=time.monotonic()-begun, physics=counts, formal_updates=0))


if __name__=='__main__':
    main()
