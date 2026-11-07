"""Small stage summaries; no raw trajectories, observations or model weights."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
out = a.root/'public'
out.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (out/name).write_text(json.dumps(value, indent=2, allow_nan=False))


groups = read(a.root/'frozen-groups.json')
curves, models, samples = [], [], 0
for g in groups:
    roots = [a.root/g['name']]
    if g['final_root'] != g['name']:
        roots.append(a.root/g['final_root'])
    for root in roots:
        result = read(root/'result.json')
        samples += result['collected_new_decisions']
        models.append(dict(group=g['name'], root=str(root), result=result,
                           selection=read(root/'selection.json'),
                           launch=read(root/'launch.json'), resume=read(root/'resume.json')))
        for line in (root/'training.jsonl').read_text().splitlines():
            r = json.loads(line)
            u = r['optimization']
            keys = ['iteration', 'rl_decisions', 'force_rms', 'cap_fraction', 'reward',
                    'actual_raw_mean_change_rms', 'native_span_mean_change_norm',
                    'orthogonal_mean_change_norm', 'raw_mean_norm', 'raw_marginal_std_mean',
                    'sampling_seconds', 'optimization_seconds', 'coverage', 'task_advantage',
                    'reward_terms', 'elapsed_s']
            curves.append(dict(group=g['name'], **{k:r[k] for k in keys},
                               optimization=u))
save('learning-curves.json', curves)
save('training-identities.json', models)
save('extension-decision.json', read(a.root/'extension-decision.json'))
save('conditional-probability.json', read(a.root/'conditional-probability.json'))
audit = read(a.root/'walking-audit-01/summaries.json')
save('native-walk-audit.json', audit)
rows = read(a.root/'final-01/summaries.json')
save('final-physical.json', rows)
save('models.json', read(a.root/'final-01/models.json'))
save('common-prefix.json', read(a.root/'final-01/common-prefix.json'))
keys = ['model', 'case', 'execution', 'error_cm', 'force_rms', 'cap_fraction',
        'root_max_displacement', 'qpd_max_deg', 'boundary']
with (out/'final-matrix.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    writer.writerows({k:r.get(k) for k in keys} for r in rows)
ledger = read(a.root/'ledger.json')
save('resource-ledger.json', ledger)
subprocess.run([sys.executable, str(Path(__file__).with_name('temporal_metrics.py')),
                '--root', str(a.root)], check=True)
save('server-completion.json', dict(status='server_artifacts_complete_needs_review',
                                   total_new_RL_decisions=samples,
                                   completed_experiment_job_seconds=sum(j.get('elapsed_s',0) for j in ledger['jobs']),
                                   current_summary_job_not_double_counted=True,
                                   frozen_execution_cells=len(rows),
                                   scientific_success='requires physical review, not automatic',
                                   Student_training=0, C0_training=0))
print('TEMPORAL_SERVER_SUMMARY', samples, len(rows), flush=True)
