"""Read only temporal diagnostics; no model, relabeling, or simulation."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def audit(folder, previous_audit, output):
    previous = json.loads(Path(previous_audit).read_text())
    indices = np.asarray(previous['sample_checks']['indices'])
    rows = {}
    with h5py.File(Path(folder) / 'buffer.hdf5', 'r') as f:
        cap = f['qpos'].shape[0]
        def take(key, shift=0):
            return np.stack([f[key][int((t + shift) % cap), int(e)] for t, e in indices])
        full_indices = indices.copy()
        cfg = json.loads((Path(folder) / 'config.json').read_text())
        valid = np.ones(len(indices), dtype=bool)
        for shift in range(-2, 3):
            for key in ('terminated', 'truncated'):
                valid &= ~take(key, shift).reshape(len(indices), -1).any(axis=1)
            valid &= ((indices[:, 0] + shift) % cap != cfg['_idx'])
        indices = indices[valid]
        assert len(indices) > 0
        obs = take('observation-state')
        privileged = take('observation-privileged_state')
        last_action = take('observation-last_action')
        default = np.array([-.1, 0, 0, .3, -.2, 0] * 2 + [0] * 17)
        def error(a, b):
            diff = a - b
            return {'rms': float(np.sqrt(np.mean(diff ** 2))),
                    'max_abs': float(np.max(np.abs(diff)))}
        for shift in (-2, -1, 0, 1, 2):
            q, v = take('qpos', shift), take('qvel', shift)
            w, x, y, z = q[:, 3:7].T
            gravity = -np.column_stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))
            rows[str(shift)] = {
                'state_position': error(obs[:, :29], q[:, 7:] - default),
                'state_velocity': error(obs[:, 29:58], v[:, 6:]),
                'state_gravity': error(obs[:, 58:61], gravity),
                'state_angular_velocity': error(obs[:, 61:64], v[:, 3:6] * .25),
                'privileged_root_height': error(privileged[:, :1], q[:, 2:3]),
                'last_action_vs_action': error(last_action, take('action', shift)),
                'last_action_vs_five_times_action': error(last_action, 5 * take('action', shift)),
            }
        reset_window = sum(bool(take(k, s).any()) for k in ('terminated', 'truncated') for s in range(-2, 3))
    result = {'rows': len(indices), 'original_rows': len(full_indices),
              'excluded_near_reset_or_cursor': int((~valid).sum()),
              'indices': indices.tolist(), 'seed': 20261030, 'model_forwards': 0,
              'simulation_steps': 0, 'shift_definition': 'Compare observation[t] to physical/action field[t+shift]',
              'window_columns_with_any_reset_flag': reset_window, 'shifts': rows,
              'note': 'Diagnostics only; does not change original data or sampling/relabeling alignment.'}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder'); p.add_argument('previous_audit'); p.add_argument('output')
    a = p.parse_args()
    audit(a.folder, a.previous_audit, a.output)
