"""Freeze development selections and summarize saved traces; no model execution."""
import collections
import datetime
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from m39b_analyze import summarize

R = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m39b-retention-20261104')
P = R.parents[2]

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_once(p, data):
    encoded = json.dumps(data, indent=2, allow_nan=False)
    if p.exists():
        assert read(p) == data, f'Immutable artifact already differs: {p}'
    else:
        with p.open('x') as f:
            f.write(encoded + '\n')

def main():
    start = time.monotonic()
    out = R / 'development-closeout'
    out.mkdir(exist_ok=True)
    plan = read(Path(__file__).with_name('m39b_protocol.json'))
    models = {
        'initial': {'path': str(P / 'bridge-v2-owned/runs/m39a-demand-20261104/B-ppo-03/job/best.pt'),
                    'sha256': '72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a'},
        'raw': {'path': str(R / 'A-ppo-04/job/best.pt'),
                'sha256': 'cce7e0fedcd4bb3c6e821d933c06d29c96ebb12d4473a64cd03f5df072cd3aac'},
        'action': {'path': str(R / 'B-ppo-05/job/best.pt'),
                   'sha256': '49cc22b8df7f08ea4b57125f5366a19e32856f9ec6d38d14f5aea7c4364427fe'},
        'T30': {'path': str(P / 'bridge-v2-owned/runs/stop-sensing-20261103/B1-train-02/job/best.pt'),
                'sha256': 'c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33'},
    }
    for name, ident in models.items():
        assert sha(Path(ident['path'])) == ident['sha256'], name
    write_once(out / 'final-config.json', {'models': models, 'keys': list(plan['sealed'])})
    detail, results, learning = {}, {}, {}
    for name, arm, dev in [('initial', 'A-ppo-04', 0), ('raw', 'A-ppo-04', 64), ('action', 'B-ppo-05', 56)]:
        job = R / arm / 'job'
        folder = job / f'dev-{dev:04d}'
        if dev:
            selection = read(job / 'selection.json')
            assert selection['iteration'] == dev and selection['sha256'] == models[name]['sha256']
        results[name] = read(folder / 'summary.json')
        detail[name] = {}
        for row in results[name]:
            key = row['label'].removesuffix('__mean')
            detail[name][key] = summarize(folder / (row['label'] + '.json.gz'))
        print('SAVED_PHYSICS', name, len(detail[name]), flush=True)
        if not dev:
            continue
        rows = [json.loads(s) for s in (job / 'training.jsonl').read_text().splitlines()]
        counts = {key: dict(sum((collections.Counter(row['coverage'][key]) for row in rows), collections.Counter()))
                  for key in ('decisions_by_phase', 'decisions_by_family', 'completed_episodes_by_family')}
        grads = [g for row in rows for g in row['optimization'].get('gradient_components', [])]
        learning[name] = {
            'selection': selection, 'result': read(job / 'result.json'), 'coverage': counts,
            'weighted_retention_to_ppo_gradient_ratio': {
                'min': min(g['ratio'] for g in grads), 'median': float(np.median([g['ratio'] for g in grads])),
                'max': max(g['ratio'] for g in grads)},
            'development': [{'iteration': int(p.parent.name[4:]), **read(p)} for p in sorted(job.glob('dev-*/score.json'))],
            'rollbacks': [read(p) for p in sorted(job.glob('rollback-*.json'))],
            'batches': [{key: row[key] for key in ('iteration', 'rl_decisions', 'sampling_seconds', 'optimization_seconds',
                        'force_rms', 'cap_fraction', 'reward', 'retention_diagnostics', 'coverage', 'optimization')} for row in rows]
        }
    for name, data in [('development-physics', detail), ('development-results', results), ('learning', learning)]:
        write_once(out / (name + '.json'), data)
    cost = {'kind': 'CPU saved-trajectory analysis, no simulation or policy update',
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'wall_seconds': time.monotonic()-start,
            'source': str(Path(__file__).resolve()), 'source_sha256': sha(Path(__file__))}
    # Append-only, including repeated read-only analysis costs if explicitly run again.
    with (out / 'analysis-cost.jsonl').open('a') as f:
        f.write(json.dumps(cost) + '\n')
    print(json.dumps(cost), flush=True)

if __name__ == '__main__':
    main()
