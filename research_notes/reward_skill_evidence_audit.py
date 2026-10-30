"""Audit saved actions/contact records; never load a model or advance physics."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(episode, contract):
    rows, controls = episode['physics'], episode['control']
    actions = np.asarray([c['action'] for c in controls])
    targets = np.asarray([c['q_PD'] for c in controls])
    predicted = np.asarray(contract['q_PD_offset']) + np.clip(
        actions * 5, -contract['clip'], contract['clip']) * np.asarray(contract['q_PD_scale']) / 5
    error = float(abs(predicted - targets).max())
    assert error < 2e-6, error  # float32 target production versus float64 audit
    increments = np.max(abs(np.diff(targets, axis=0)), axis=1)
    stored = np.asarray([c['q_PD_step_max_rad'] for c in controls[1:]])
    assert float(abs(increments - stored).max()) < 2e-6
    out = {'q_PD_mapping_max_error_rad': error, 'phase_peaks': {}}
    for phase, lo, hi in [('enter', 0, .75), ('execute', .75, 3.25),
                           ('exit', 3.25, 4), ('settle', 4, 6)]:
        indices = [i for i, c in enumerate(controls) if i and lo-1e-8 <= c['relative_time'] < hi-1e-8]
        i = max(indices, key=lambda j: controls[j]['q_PD_step_max_rad'])
        a, b = controls[i-1], controls[i]
        j = int(np.argmax(abs(targets[i]-targets[i-1])))
        out['phase_peaks'][phase] = {
            'relative_time': b['relative_time'], 'joint': contract['dof_names'][j],
            'q_PD_step_deg': float(np.degrees(b['q_PD_step_max_rad'])),
            'alpha': b['alpha'], 'raw_action_pair': [a['action'][j], b['action'][j]]}
    contact = np.asarray([r['support'] for r in rows], bool)
    forces = np.asarray([r['foot_normal_force_n'] for r in rows])
    out['contact_geometry_counts'] = {str(pattern): int((contact == pattern).all(axis=1).sum())
        for pattern in [(True, True), (True, False), (False, True), (False, False)]}
    out['geometry_vs_force_above_1N_disagreement_frames'] = int((contact != (forces > 1)).any(axis=1).sum())
    # Consecutive absence of geometric ground contact followed by contact again.
    # Count evidence of support change, without claiming each is a clean gait step.
    out['foot_recontact_events'] = []
    for side in range(2):
        was = True
        events = []
        for i, active in enumerate(contact[:, side]):
            if active and not was:
                events.append({'time': rows[i]['force_time']-3,
                    'normal_force_n': float(forces[i, side]),
                    'next_50ms_normal_force_peak_n': float(forces[i:i+11, side].max()),
                    'foot_xyz_cm': (np.asarray(rows[i]['feet_xyz'][side])*100).tolist()})
            was = active
        out['foot_recontact_events'].append(events)
    out['final_root_velocity'] = rows[-1]['qvel'][:6]
    out['last_1s_q_PD_step_max_deg'] = float(np.degrees(max(c['q_PD_step_max_rad']
        for c in controls if c['relative_time'] >= 5-1e-8)))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('outputs', type=Path);p.add_argument('destination', type=Path)
    a = p.parse_args();read = lambda f: json.loads((a.outputs/f).read_text())
    result = {name: audit(read(name+'.json'), read('action-contract.json'))
        for name in ['standing', 'raisearms-l-m', 'move-ego-0-0.3']}
    a.destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))
