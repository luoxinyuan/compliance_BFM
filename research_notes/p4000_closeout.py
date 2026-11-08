"""Summarize saved Issue52 results; no policy load, inference, or physics."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    evidence = {'methods': {}, 'new_training': False, 'sealed_evaluation_executed': False}
    for method in ['independent', 'joint']:
        folder = args.stage / 'runs' / f'single5201-{method}-32'
        training = [json.loads(row) for row in (folder / 'training.jsonl').read_text().splitlines()]
        entry = dict(result=read(folder / 'result.json'), selection=read(folder / 'selection.json'),
                     launch=read(folder / 'launch.json'), evaluations={}, models={}, training=[])
        for sub in ['dev-0000', 'dev-0008', 'dev-0016', 'dev-0024', 'dev-0032',
                    'sample-0000', 'sample-0016', 'sample-0032']:
            rows = read(folder / sub / 'summary.json')
            entry['evaluations'][sub] = [{key: row[key] for key in [
                'label', 'condition', 'boundary', 'complete', 'root_max_displacement',
                'episode_return', 'all', 'moving', 'holding', 'released', 'phase_metrics'
            ] if key in row} | ({'decelerating': row['decelerating']} if 'decelerating' in row else {}) for row in rows]
        for name in ['initial.pt', 'candidate-0008.pt', 'candidate-0016.pt',
                     'candidate-0024.pt', 'candidate-0032.pt', 'best.pt', 'last.pt']:
            path = folder / name
            entry['models'][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for row in training:
            entry['training'].append({key: row[key] for key in [
                'iteration', 'RL_decisions', 'optimizer_updates', 'critic_optimizer_updates',
                'sampling_seconds', 'optimization_seconds', 'decisions_per_second',
                'physical_steps', 'BFM_control_steps', 'reset_prefill_steps', 'source_steps',
                'reward_mean', 'reward_terms', 'force_RMS', 'cap_fraction', 'groups',
                'exploration', 'mean_update_z_angle', 'optimization', 'elapsed_s']})
        phases = {}
        for row in training:
            for key, val in row['groups'].items():
                if key.startswith('phase:'):
                    phases[key] = phases.get(key, 0) + val['decisions']
        entry['actual_phase_decisions'] = phases
        evidence['methods'][method] = entry
    diagnosis = args.stage / 'runs/mean-diagnosis-02'
    evidence['diagnosis'] = {name: read(diagnosis / (name + '.json')) for name in [
        'diagnosis-result', 'fixed-state-geometry', 'fixed-state-five-substep-actions', 'physical-branches']}
    evidence['ledger_snapshot'] = read(args.stage / 'ledger.json')
    evidence['limits'] = ('Single paired development seed; fixed32 method comparison, not unseen success. '
                          'Fixed-observation BFM replay is not physical rollout. Branches are 2.4s diagnostics, '
                          'not PPO training, recovery labels, or whole-episode improvements.')
    target = args.output / 'p4000_full_analysis.json'
    target.write_text(json.dumps(evidence, indent=2, allow_nan=False))
    public = {key: value for key, value in evidence.items() if key != 'methods'}
    public['methods'] = {}
    for method, entry in evidence['methods'].items():
        item = {key: value for key, value in entry.items() if key not in ['training', 'evaluations']}
        item['evaluations'] = {node: [{key: row[key] for key in [
            'label', 'condition', 'boundary', 'complete', 'root_max_displacement',
            'episode_return', 'all', 'moving', 'decelerating', 'holding', 'released'] if key in row}
            for row in rows] for node, rows in entry['evaluations'].items()}
        item['training'] = [{key: row[key] for key in [
            'iteration', 'RL_decisions', 'optimizer_updates', 'critic_optimizer_updates',
            'sampling_seconds', 'optimization_seconds', 'decisions_per_second',
            'reward_mean', 'mean_update_z_angle', 'elapsed_s']} | {
                'value_EV': row['optimization']['updated_value_EV'],
                'actor_lr_after': row['optimization']['actor_lr_after'],
                'pre_logprob_max_error': row['optimization']['pre_logprob_max_error']}
            for row in entry['training']]
        public['methods'][method] = item
    (args.output / 'p4000_final_evidence.json').write_text(json.dumps(public, separators=(',', ':'), allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for method, entry in evidence['methods'].items():
        for ax, case in zip(axes[0], ['short08', 'normal72', 'stop66']):
            values = []
            for node in [0, 8, 16, 24, 32]:
                rows = entry['evaluations'][f'dev-{node:04d}']
                values.append(next(r['all']['error_cm'] for r in rows if r['label'].split('__')[0] == case))
            ax.plot([0, 8, 16, 24, 32], values, marker='o', label=method)
            ax.set(title=case + ': active 3D error', xlabel='PPO batches', ylabel='cm')
        rows = entry['training']; x = [r['iteration'] for r in rows]
        axes[1, 0].plot(x, [r['reward_mean'] for r in rows], label=method)
        axes[1, 1].plot(x, [r['mean_update_z_angle']['mean'] for r in rows], label=method)
        axes[1, 2].plot(x, [r['optimization']['updated_value_EV'] for r in rows], label=method)
    axes[1, 0].set(title='Sampled rollout reward (changing task mix)', xlabel='PPO batches', ylabel='reward / decision')
    axes[1, 1].set(title='Mean policy change per batch', xlabel='PPO batches', ylabel='latent direction degrees')
    axes[1, 2].set(title='Critic target fit (not actor success)', xlabel='PPO batches', ylabel='explained variance')
    for ax in axes.flat:
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Issue52: same-seed fixed32 development; no sealed evaluation')
    fig.savefig(args.output / 'p4000_development_curves.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'status': 'completed', 'evidence_bytes': target.stat().st_size,
                      'new_physics_steps': 0, 'new_RL_training_decisions': 0}))


if __name__ == '__main__':
    main()
