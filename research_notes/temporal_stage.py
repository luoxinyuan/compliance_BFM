"""Bounded stage-local sequential queue; no generic runner or new research tasks."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import traceback

P = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
R = P/'bridge-v2-owned/trees/native-temporal-learning/repo'
O = P/'bridge-v2-owned/runs/native-temporal-20261107'
PY = str(P/'.venv/bin/python')
PLAN = json.loads((R/'research_notes/temporal_protocol.json').read_text())
STATE = O/'stage-status.json'


def save(value):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(STATE)


def job(name, limit, command, gpu=True):
    for attempt in range(24):
        mi = {l.split(':')[0]: int(l.split(':')[1].split()[0])*1024
              for l in Path('/proc/meminfo').read_text().splitlines()}
        query = subprocess.check_output(
            ['nvidia-smi', '-i', '0', '--query-gpu=memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True).strip()
        mem, util = map(int, query.split(','))
        snap = dict(time=time.time(), GPU0=query, RAM_available_bytes=mi['MemAvailable'],
                    affinity=len(os.sched_getaffinity(0)), load=os.getloadavg(),
                    IO_pressure=Path('/proc/pressure/io').read_text())
        (O/(name+'-resources.json')).write_text(json.dumps(snap, indent=2))
        if (not gpu or (mem < 1000 and util < 5)) and mi['MemAvailable'] > 24*1024**3:
            break
        time.sleep(10)
    else:
        raise RuntimeError('No independently safe resource; no CPU/other-card fallback')
    current = sum(f.stat().st_size for f in O.rglob('*') if f.is_file())
    assert current < 11*1024**3, 'Output reserve exhausted; do not launch more jobs'
    subprocess.run([PY, str(R/'research_notes/native_job.py'), '--ledger', str(O/'ledger.json'),
                    '--name', name, '--limit', str(limit), '--', *command],
                   check=True, cwd=O)


def read(path):
    return json.loads(path.read_text())


def extension_decision(groups):
    by = {}
    reports = []
    for g in groups:
        root = O/g['name']
        selected = read(root/'selection.json')
        initial = {r['label'].split('__')[0]: r for r in read(root/'dev-0000/summary.json')}
        rows = {r['label'].split('__')[0]: r for r in read(root/selected['evaluation'])}
        long = rows['long']['error_cm'] <= .9*initial['long']['error_cm']
        long_force = rows['long']['force_rms'] <= initial['long']['force_rms']+.5
        keys = ['walk_continue', 'walk_stop']
        moving = sum(rows[k]['error_cm'] for k in keys) <= .9*sum(initial[k]['error_cm'] for k in keys)
        moving_force = all(rows[k]['force_rms'] <= initial[k]['force_rms']+.5 for k in keys)
        passed = selected['iteration'] > 0 and ((long and long_force) or (moving and moving_force))
        by[(g['arm'], g['seed'])] = passed
        reports.append(dict(group=g['name'], selected=selected, long=long,
                            long_force=long_force, moving=moving,
                            moving_force=moving_force, passes=passed))
    common = any(all(by[(arm, seed)] for seed in PLAN['seeds']) for arm in PLAN['arms'])
    ledger = read(O/'ledger.json')
    spent = sum(j.get('elapsed_s', 0) for j in ledger['jobs'])
    worst = max(j['elapsed_s'] for j in ledger['jobs'] if j['name'].startswith('train-'))
    reserve = 4*1.25*worst + 3500
    resources = spent+reserve <= PLAN['budget']['experiment_job_seconds']
    result = dict(extend_all_four=common and resources, reports=reports,
                  common_evidence=common, resource_reserve_ok=resources,
                  spent_s=spent, additional_reserved_s=reserve,
                  final_seen=False, no_asymmetric_extension=True)
    (O/'extension-decision.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    assert read(O/'conditional-probability.json')['status'] == 'passed'
    assert read(O/'walking-audit-01/result.json')['study']['episodes'] == 3
    assert not any(j['status'] == 'running' for j in read(O/'ledger.json')['jobs'])
    # Counterbalanced order, same paired seed and physical task distribution.
    groups = [dict(name='A-4901', arm='independent', seed=4901),
              dict(name='B-4901', arm='continuous', seed=4901),
              dict(name='B-4902', arm='continuous', seed=4902),
              dict(name='A-4902', arm='independent', seed=4902)]
    commit = subprocess.check_output(['git', '-C', str(R), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(R), 'status', '--porcelain'])
    save(dict(status='first16_training', runtime=commit, groups=groups,
              pid=os.getpid(), started=time.time(), sealed_opened=False))
    for g in groups:
        job('train-'+g['name'], 2800,
            [PY, '-u', str(R/'research_notes/temporal_ppo.py'), '--kind', 'direct',
             '--exploration', 'guided', '--arm', g['arm'], '--stage', 'issue49',
             '--output', str(O/g['name']), '--gpu', '0', '--iterations', '16',
             '--seed', str(g['seed']), '--resume', PLAN['initialization']['path'],
             '--max-seconds', '2650'])
        result = read(O/g['name']/'result.json')
        assert result['status'] == 'completed' and result['new_decisions'] == 12288, result
    decision = extension_decision(groups)
    if decision['extend_all_four']:
        save(dict(status='matched_extension_to32', runtime=commit,
                  extension=decision, sealed_opened=False))
        for g in groups:
            parent = O/g['name']
            new = g['name']+'-extended'
            job('extend-'+g['name'], 2800,
                [PY, '-u', str(R/'research_notes/temporal_ppo.py'), '--kind', 'direct',
                 '--exploration', 'guided', '--arm', g['arm'], '--stage', 'issue49-extension',
                 '--parent', str(parent), '--output', str(O/new), '--gpu', '0',
                 '--iterations', '32', '--seed', str(g['seed']), '--resume', str(parent/'last.pt'),
                 '--max-seconds', '2650'])
            result = read(O/new/'result.json')
            assert result['status'] == 'completed' and result['new_decisions'] == 12288, result
            g['final_root'] = new
    else:
        for g in groups:
            g['final_root'] = g['name']
    (O/'frozen-groups.json').write_text(json.dumps(groups, indent=2))
    save(dict(status='frozen_final', runtime=commit, groups=groups,
              sealed_opened=True, no_further_RL=True))
    job('frozen-final', 3000,
        [PY, '-u', str(R/'research_notes/temporal_final.py'), '--gpu', '0',
         '--output', str(O/'final-01'), '--seed', '4910', '--max-seconds', '2900'])
    job('saved-qpos-videos', 500,
        [PY, '-u', str(R/'research_notes/temporal_render.py'), '--root', str(O), '--repo', str(R)])
    job('public-summary', 240,
        [PY, str(R/'research_notes/temporal_summarize.py'), '--root', str(O)], gpu=False)
    save(dict(status='complete_server_artifacts_ready', runtime=commit,
              groups=groups, ended=time.time(), Mac_sync_verified=False,
              no_more_training=True, scientific_review='pending human-readable review'))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        save(dict(status='failed_preserved_no_automatic_restart',
                  error=traceback.format_exc(), ended=time.time()))
        raise
