"""Post-hoc constraint audit from saved traces only. No model, fit or rollout.

This describes the original results; it does not replace sealed pass criteria.
"""
import gzip
import json
from pathlib import Path
import numpy as np

RUN = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m1-state-response-20261031A')


def angle_degrees(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def main():
    out = RUN / 'causal_final-attempt-01/outputs'
    contract = json.loads((out / 'action-contract.json').read_text())
    rows = []
    for p in sorted(out.glob('*.json.gz')):
        e = json.loads(gzip.decompress(p.read_bytes()))
        if 'physics' not in e or e['boundary']:
            continue
        cs, ps = e['control'], e['physics']
        q = np.array([c['q_PD'] for c in cs])
        ct = np.array([c['time'] for c in cs])
        dq = np.abs(np.diff(q, axis=0))
        dq[ct[1:] < e['control_ready_s'] - 1e-8] = -1
        i, j = np.unravel_index(dq.argmax(), dq.shape)
        before, after = cs[i], cs[i + 1]
        # qpos/qvel are post-step states; evaluate recovery using their actual time.
        tt = np.array([r['time'] for r in ps])
        release = e['condition']['release']
        recovery = (tt >= release - 1e-8) & (tt < release + 2 + 1e-8)
        tail = (tt >= release + 1.5 - 1e-8) & (tt <= release + 2 + 1e-8)
        speed = np.linalg.norm(np.array([r['qvel'][:3] for r in ps]), axis=-1)
        last = int(np.argmin(np.abs(tt - (release + 2))))
        assert abs(tt[last] - (release + 2)) < 1e-7
        peak_physics = min(4 * (i + 1), len(ps) - 1)
        support = np.array([r['foot_normal_force_n'] for r in ps]) > 1e-6
        feet = np.array([r['feet_xyz'] for r in ps])
        r0 = np.flatnonzero(recovery)[0]
        peak = dict(time=after['time'], joint=contract['dof_names'][j],
                    q_PD_step_deg=float(np.degrees(dq[i, j])),
                    executed_z_step_deg=angle_degrees(before['executed_z'][0], after['executed_z'][0]),
                    same_joint_actual_20ms_motion_deg=float(np.degrees(after['pre_qpos'][7+j] - before['pre_qpos'][7+j])),
                    same_joint_standing_action_qPD_step_deg=float(np.degrees(after['standing_q_PD'][j] - before['standing_q_PD'][j])),
                    first_physics_torque_nm=ps[peak_physics]['torque'][j],
                    native_coords_before=before['native_coords'], native_coords_after=after['native_coords'])
        rows.append(dict(label=e['label'], contact_enabled=e['condition']['enabled'], command_peak=peak,
                         release_plus_2_speed_mps=float(speed[last]),
                         recovery_last_half_second_speed_rms_mps=float(np.sqrt(np.mean(speed[tail]**2))),
                         recovery_last_half_second_speed_minmax_mps=[float(speed[tail].min()), float(speed[tail].max())],
                         recovery_both_feet_load_fraction=float(support[recovery].all(-1).mean()),
                         recovery_any_foot_load_fraction=float(support[recovery].any(-1).mean()),
                         recovery_foot_xy_motion_max_cm=float(np.linalg.norm(feet[recovery, :, :2]-feet[r0, :, :2], axis=-1).max()*100)))
    result = dict(interpretation='Post-hoc descriptive audit, not new pass thresholds or causal action decomposition. Constant-z jumps establish native state-feedback sensitivity; they do not explain every Student jump. Two-second speed alone does not establish divergent instability. Some unloading is permitted for whole-body responses, but neither foot displacement nor temporary single support qualifies controlled stepping.', rows=rows)
    (RUN / 'analysis/constraint-audit.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for r in rows:
        if '__alignment' in r['label'] or '__distill' in r['label'] or 'raised__no_contact' in r['label']:
            print(json.dumps(r))


if __name__ == '__main__':
    main()
