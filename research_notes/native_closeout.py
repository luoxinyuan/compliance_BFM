"""Read saved experiment artifacts; no model, GPU, physics, or new training."""
import argparse
import json
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
pub = a.root / 'public'


def load(path):
    return json.loads(path.read_text())


def save(name, data):
    (pub / name).write_text(json.dumps(data, indent=2, allow_nan=False))


def cost(r):
    # Exact existing m39a_metrics.cost, copied here to avoid GPU/model imports.
    if not r['condition']['enabled']:
        return .5*r['root_max_displacement'] + .02*r.get('torso_excursion_deg', 0) + 20*bool(r['boundary'])
    heading = r['torso_heading_rms_deg'] if r['condition'].get('geometry') == 'torso_pair' else r['moving_heading_rms_deg']
    return r.get('sum_force_norm_rms', r['force_rms']) + 3*r['error_cm']/100 + r['cap_fraction'] + .04*heading + 20*bool(r['boundary'])


review = {}
for arm in ['ordinary', 'guided', 'coordinate']:
    root = a.root / ('ppo-' + arm)
    initial = {r['label']: r for r in load(root / 'dev-0000/summary.json')}
    rounds = []
    for f in sorted(root.glob('dev-*/summary.json')):
        rows = []
        for r in load(f):
            base = initial[r['label']]
            rows.append(dict(label=r['label'], initial_cost=cost(base), current_cost=cost(r),
                             relative=cost(r)/max(cost(base), .1),
                             root_max_m=r['root_max_displacement'],
                             torso_excursion_deg=r.get('torso_excursion_deg'),
                             error_cm=r['error_cm'], force_rms=r['force_rms']))
        rel = [v['relative'] for v in rows]
        rounds.append(dict(evaluation=f.parent.name, cases=rows,
                           score=.5*max(rel)+sum(rel)/len(rel)))
    review[arm] = dict(selection=load(root / 'selection.json'), rounds=rounds)
save('selection-review.json', review)

ledger = load(a.root / 'ledger.json')
assert all(j['status'] in ['completed', 'failed'] for j in ledger['jobs'])
supplemental = [load(a.root / n) for n in ['native-direction-versions.json', 'update-geometry-diagnostic.json', 'native-coordinate-check.json']]
jobs_s = sum(j['elapsed_s'] for j in ledger['jobs'])
supp_s = sum(j['seconds'] for j in supplemental)
results = {arm: load(a.root / ('ppo-' + arm) / 'result.json') for arm in review}
assert sum(r['decisions'] for r in results.values()) == 73728
assert all(r['status'] == 'completed' for r in results.values())
mac = load(a.root / 'mac-final32-sync-status.json')
assert mac['status'] == 'complete_verified_on_mac'
output_bytes = sum(f.stat().st_size for f in a.root.rglob('*') if f.is_file())
summary = dict(
    status='completed_no_further_training', timestamp=time.time(),
    supervised_experiment_job_seconds=jobs_s,
    supplemental_diagnostic_seconds=supp_s,
    total_experiment_job_seconds=jobs_s+supp_s,
    resource_definition='Each experiment supervisor includes its child workers once. Queue waiting, read-only analysis, docs/git and Mac transfer are not extra GPU experiment jobs. Failed launches, probability, qualification, BC, RL, frozen eval and both video versions are included.',
    output_bytes=output_bytes, output_GiB=output_bytes/1024**3,
    rl_decisions=73728, native_initial_supervised_updates=2000,
    accepted_PPO_Adam_updates={k: r['updates'] for k, r in results.items()},
    frozen_final_episodes=len(load(a.root/'final-01/summaries.json')),
    independent_final_conditions_note='97 executions include duplicated new_long/new_continue and identical best0 actor functions; not97 independent tasks.',
    C0_new_training=0, student_training=0, gpu_devices=['0'],
    limits=ledger['authorized_limits'], Mac_video_revision='fixed-final32',
    Mac_sync=mac, no_heldout_reselection=True,
)
assert summary['total_experiment_job_seconds'] < 24000
assert output_bytes < 15*1024**3
save('closeout.json', summary)
ledger.update(state='completed_no_further_training', supplemental_diagnostics=supplemental,
              completed_job_seconds_plus_supplemental=jobs_s+supp_s)
(a.root/'ledger.json').write_text(json.dumps(ledger, indent=2))
save('resource-ledger.json', ledger)
save('initial-physical-rollout-parity.json', load(a.root/'coordinate-initial-rollout-parity.json'))
old = load(a.root/'postprocessing/status.json')
old.update(status='completed_closed_no_further_training', Mac_sync_verified=True,
           final_video_revision=str(a.root/'videos-final32'),
           closeout_summary=str(pub/'closeout.json'),
           training_status_note='result.json is final; status.json retains last sampler snapshot')
(a.root/'postprocessing/status.json').write_text(json.dumps(old, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['Mac_sync']}, indent=2))
