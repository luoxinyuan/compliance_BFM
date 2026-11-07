"""Read-only saved-physics analysis; no model inference or new simulations."""
import argparse
import gzip
import json
import time
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
start = time.monotonic()
rows = []
for name in ['full-contact', 'half-free', 'half-contact']:
    path = a.root/'walking-audit-01'/(name+'.json.gz')
    ep = json.loads(gzip.decompress(path.read_bytes()))
    phases = [('enter', 4.9, 6.4), ('sustain', 6.4, 12.9),
              ('native_return', 12.9, 14.4), ('contact_hold', 14.4, 14.9),
              ('released_recovery', 14.9, 18.91)]
    for label, t0, t1 in phases:
        rs = [r for r in ep['physics'] if t0 <= r['force_time'] < t1]
        if not rs:
            continue
        t = np.array([r['force_time'] for r in rs])
        root = np.array([r['root'] for r in rs])
        wrist = np.array([r['wrist'] for r in rs])
        shoulder = np.array([r['shoulder_xyz'] for r in rs])
        feet = np.array([r['feet_xyz'] for r in rs])
        loads = np.array([r['foot_normal_force_n'] for r in rs])
        R = np.array([r['pelvis_matrix'] for r in rs])
        yaw = np.rad2deg(np.unwrap(np.arctan2(R[:,1,0], R[:,0,0])))
        duration = max(float(t[-1]-t[0]), .005)
        active = [r for r in rs if r['active']]
        value = dict(native=name, phase=label, seconds=len(rs)*.005,
                     root_displacement_xyz_cm=(100*(root[-1]-root[0])).tolist(),
                     root_mean_velocity_xyz_mps=((root[-1]-root[0])/duration).tolist(),
                     root_horizontal_path_cm=float(100*np.linalg.norm(np.diff(root[:,:2],axis=0),axis=-1).sum()),
                     pelvis_yaw_start_end_deg=[float(yaw[0]), float(yaw[-1])],
                     wrist_relative_shoulder_change_cm=(100*((wrist-shoulder)[-1]-(wrist-shoulder)[0])).tolist(),
                     ankle_link_path_cm=(100*np.linalg.norm(np.diff(feet,axis=0),axis=-1).sum(0)).tolist(),
                     foot_unloaded_time_s=((loads<5).sum(0)*.005).tolist(),
                     slip_contact_max_mps=max(r['contact_tangent_speed_max_mps'] for r in rs),
                     peak_tilt_deg=max(r['tilt_deg'] for r in rs))
        if active:
            F = np.array([r['force'] for r in active])
            error = np.array([np.array(r['anchor'])-r['wrist'] for r in active])
            value.update(force_xyz_mean_n=F.mean(0).tolist(),
                         force_xyz_rms_n=np.sqrt((F*F).mean(0)).tolist(),
                         force_rms_n=float(np.sqrt((F*F).sum(-1).mean())),
                         error_signed_xyz_mean_cm=(100*error.mean(0)).tolist(),
                         error_rms_cm=float(100*np.sqrt((error*error).sum(-1).mean())),
                         cap_fraction=float((np.linalg.norm(F,axis=-1)>9.999).mean()))
        rows.append(value)
result = dict(scope='saved actual physics; native timed qualification, not learned feedback',
              partner='72cm/8s with real tether, mean requested .09m/s; actual speed profile continuous',
              full_free_historical='Issue48 full-rate free has a different duration: not a causal full-vs-half free comparison',
              phases=rows, read_only_analysis_seconds=time.monotonic()-start,
              RL_decisions=0, new_physics_steps=0)
(a.root/'walk-phase-analysis.json').write_text(json.dumps(result, indent=2))
for row in rows:
    if row['phase'] == 'sustain':
        print(row['native'], row['root_mean_velocity_xyz_mps'],
              row.get('force_xyz_mean_n'), row.get('error_rms_cm'),
              row['pelvis_yaw_start_end_deg'])
