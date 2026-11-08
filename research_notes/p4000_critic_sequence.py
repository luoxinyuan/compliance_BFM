"""Finite authorized critic follow-up after the exact original owned sequence exits."""
import json, os, subprocess, time, math
from pathlib import Path
from p4000_sequence import job, remaining

ROOT = Path(__file__).resolve().parents[1]
S = ROOT.parent


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(p)


def owned_alive(identity):
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != identity['boot_id']:
        raise RuntimeError('Host boot changed; no automatic launch')
    path = Path(f'/proc/{identity["pid"]}/stat')
    if not path.exists():
        return False
    fields = path.read_text().split()
    return fields[21] == identity['start_ticks'] and fields[2] != 'Z'


def resource_check():
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip(), 'GPU is not independently idle'
    mem = {x.split(':')[0]:int(x.split(':')[1].split()[0])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable'] > 64*1024**3
    assert os.getloadavg()[0] < max(4, .75*len(os.sched_getaffinity(0)))
    disk = os.statvfs(S)
    assert disk.f_bavail*disk.f_frsize > 20*1024**3
    snapshot = dict(utc=time.time(), gpu_apps=[], memory_available_bytes=mem['MemAvailable'],
                    load=os.getloadavg(), CPU_affinity=len(os.sched_getaffinity(0)),
                    free_disk_bytes=disk.f_bavail*disk.f_frsize,
                    io_pressure=Path('/proc/pressure/io').read_text() if Path('/proc/pressure/io').exists() else 'unavailable')
    with (S/'critic-resource-checks.jsonl').open('a') as log:
        log.write(json.dumps(snapshot)+'\n')


def main():
    plan = read(ROOT/'research_notes/p4000_critic_protocol.json')
    state_path = S/'critic-followup-status.json'
    write(state_path, dict(status='queued_after_original_sequence', automatic_next_stage=False))
    original = read(S/'sequence-process.json')
    deadline = time.monotonic()+9000
    while owned_alive(original):
        assert time.monotonic()<deadline, 'Finite waiting deadline exceeded'
        time.sleep(15)
    status = read(S/'sequence-status.json')
    assert status['status']=='PPO_and_sealed_physics_complete_pending_analysis_videos', status
    # Do not examine original sealed physical scores or any new-arm results to
    # choose duration/replication. Resource decision uses only runtime records.
    assert not read(S/'extension-decision.json')['extend_both'], 'Unexpected budget-consuming extension'
    resource_check()
    # User expanded the aggregate wall budget to make both complete paired
    # 32-batch arms possible; do not allocate based on new-arm/held-out scores.
    train_limit = 5500
    unit_limit, final_limit, reserve = 90, 600, 400
    available = remaining()
    paired_seeds = [5201,5202]
    assert available >= 2*train_limit+unit_limit+final_limit+reserve+60, ('Insufficient aggregate budget for both matched32 arms',available)
    write(S/'critic-followup-allocation.json', dict(seeds=paired_seeds, iterations=32,
          max_job_seconds_each=train_limit, unit_max_seconds=unit_limit, final_max_seconds=final_limit,
          remaining_before=available, reserve_seconds=reserve,
          selection_uses='runtime/resource forecast only; no sealed physical scores or new-arm outcomes',
          single_seed_if_insufficient='no automatic downgrade; preserve resource boundary and report'))
    write(state_path, dict(status='checking_CUDA_update_contract', seeds=paired_seeds))
    job('critic-unit-01', ROOT, 'p4000_critic_check.py', unit_limit,
        ['--output',str(S/'runs/critic-unit-01')])
    assert read(S/'runs/critic-unit-01/result.json')['actor_equal']
    outputs = {}
    for seed in paired_seeds:
        name = f'critic-seed{seed}-32'
        out = S/'runs'/name
        resource_check()
        write(state_path, dict(status='critic_comparison_running', seeds=paired_seeds, current_seed=seed))
        job(name, ROOT, 'p4000_critic_ppo.py', train_limit,
            ['--output',str(out),'--seed',str(seed),'--iterations','32','--std','.008',
             '--max-seconds',str(train_limit-60)])
        outputs[seed] = out
    models = dict(P4000=plan['initial']['path'])
    for seed,out in outputs.items():
        models[f'original{seed}_fixed32'] = str(S/f'runs/seed{seed}-32/last.pt')
        models[f'critic{seed}_fixed32'] = str(out/'last.pt')
    write(S/'critic-final-models.json', models)
    write(state_path, dict(status='critic_matched_final_evaluation', seeds=paired_seeds))
    resource_check()
    job('critic-final-01', ROOT, 'p4000_critic_evaluate.py', final_limit,
        ['--output',str(S/'runs/critic-final-01')])
    decisions = sum(read(out/'result.json')['RL_decisions'] for out in outputs.values())
    ledger = read(S/'ledger.json')
    original_decisions = sum(read(S/f'runs/seed{seed}-32/result.json')['RL_decisions'] for seed in [5201,5202])
    assert decisions <= 49152 and original_decisions+decisions <= ledger['RL_decisions_limit']
    ledger.update(critic_followup_RL_decisions=decisions, combined_RL_decisions=original_decisions+decisions)
    write(S/'ledger.json',ledger)
    write(state_path,dict(status='critic_comparison_complete_pending_analysis_and_videos',
          seeds=paired_seeds,new_RL_decisions=decisions,combined_RL_decisions=original_decisions+decisions,
          models=models,automatic_next_training=False))


if __name__=='__main__':
    try:
        main()
    except BaseException:
        import traceback
        write(S/'critic-followup-status.json',dict(status='failed_saved_no_auto_retry',error=traceback.format_exc()))
        raise
