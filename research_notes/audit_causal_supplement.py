"""Post-run, CPU-only matched-duration and bounded-fit audit. No new model calls."""
import json, sys
from pathlib import Path
import numpy as np
import torch


def audit(out):
    source = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-accumulated-20261030A/attempt-02/outputs')
    episodes = {'C': json.loads((source/'accumulated.json').read_text()),
                'D': json.loads((out/'standing_accumulated.json').read_text())}
    c, d = episodes['C'], episodes['D']
    assert c['same_initial_sha256'] == d['same_initial_sha256']
    assert all(x['time'] == y['time'] and x['anchor'] == y['anchor'] for x, y in zip(c['physics'], d['physics']))
    result = {'same_initial': True, 'same_partner_on_common_duration': True,
              'audited_physical_status': 'physical-stopped' if d['stop_boundary'] else 'completed',
              'reachability_status': 'completed', 'matched': {}}
    for phase, lo, hi in [('ramp', 3., 7.), ('observed_hold_only', 7., 8.), ('all_observed_contact', 3., 8.)]:
        ids = [i for i, r in enumerate(d['physics']) if r['active'] and lo <= r['force_time'] < hi]
        result['matched'][phase] = {}
        for name, ep in episodes.items():
            rows = [ep['physics'][i] for i in ids]
            e = np.array([np.array(r['anchor'])-r['wrist'] for r in rows])
            force = np.array([r['force'] for r in rows])
            last = rows[-1]; initial = ep['initial']
            delta = lambda key: (100*(np.array(last[key])-initial[key])).tolist()
            result['matched'][phase][name] = dict(n=len(rows), duration_s=len(rows)/200,
                time_last=last['time'], error_RMS_cm=float(100*np.sqrt(np.mean(np.sum(e*e, axis=1)))),
                error_XYZ_RMS_cm=(100*np.sqrt(np.mean(e*e, axis=0))).tolist(),
                force_RMS_N=float(np.sqrt(np.mean(np.sum(force*force, axis=1)))),
                force_XYZ_mean_N=force.mean(0).tolist(), force_XYZ_RMS_N=np.sqrt(np.mean(force*force, axis=0)).tolist(),
                force_peak_N=float(np.linalg.norm(force, axis=1).max()),
                force_cap_fraction=float(np.mean(np.linalg.norm(force, axis=1)>=4.9999)),
                wrist_delta_cm=delta('wrist'), shoulder_delta_cm=delta('shoulder_xyz'), root_delta_cm=delta('root'),
                wrist_relative_root_delta_cm=(np.array(delta('wrist'))-delta('root')).tolist(),
                waist_delta_deg=np.degrees(np.array(last['qpos'])[19:22]-np.array(initial['qpos'])[19:22]).tolist(),
                root_height_min_cm=100*min(r['height'] for r in rows), tilt_max_deg=max(r['tilt_deg'] for r in rows),
                both_feet_loaded_fraction=float(np.mean([min(r['foot_normal_force_n'])>1e-6 for r in rows])),
                joint_excess_deg=float(np.degrees(max(r['joint_limit_excess_max_rad'] for r in rows))),
                torque_saturation_fraction=float(np.mean([r['torque_saturation'] for r in rows])))
    contract = json.loads((out/'action-contract.json').read_text())
    names = contract['dof_names']
    last = d['control'][-1]
    action = np.array(last['action']); correction = np.array(last['correction_q_PD'])
    result['D_boundary'] = dict(**d['stop_boundary'], last_successful_command_time=last['tether_time'],
        last_successful_action_box_headroom_normalized=(contract['clip']/5-abs(action)).tolist(),
        right_arm_names=names[22:], last_applied_correction_deg=np.degrees(correction).tolist(),
        last_velocity_rad_s=last['accumulation']['velocity'],
        qualification='The failing next command was rejected before row logging. Exact failed joint and needed retraction were not recorded; no post-hoc claim of which joint caused the conflict.')
    fits = json.loads((out/'latent-reachability.json').read_text())
    vectors = torch.load(out/'temporary-latent-solutions.pt', map_location='cpu', weights_only=True)
    error = (vectors['predicted_q']-vectors['target_q']).numpy()
    trace = json.loads((out/'latent-optimization-trace.json').read_text())
    result['latent_details'] = []
    for i, row in enumerate(fits['final']):
        rms = np.degrees(np.sqrt(np.mean(error[i]**2, axis=0)))
        largest = np.argsort(rms)[-4:][::-1]
        result['latent_details'].append(dict(**row,
            top_joint_RMS_deg=[dict(joint=names[j], rms_deg=float(rms[j])) for j in largest],
            initial_arm_RMS_deg=fits['baseline'][i]['arm_q_RMS_deg'],
            initial_nonarm_RMS_deg=fits['baseline'][i]['nonarm_q_RMS_deg'],
            first_gradient_norm=fits['first_gradient'][i],
            final_gradient_norm=trace[-1]['gradient_norm_per_case'][i],
            last_49_update_loss_reduction_pct=100*(1-trace[-1]['loss_per_case'][i]/max(trace[-2]['loss_per_case'][i],1e-30))))
    result['limitation'] = 'D failed before full hold/release; compare only paired observed windows. Fixed-state finite fits do not prove closed-loop stability or global unreachability. Gate-one is diagnostic only.'
    (out/'causal-supplement.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='latent_details'}, indent=2))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
