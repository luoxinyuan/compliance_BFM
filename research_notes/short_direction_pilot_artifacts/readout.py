"""Read deterministic development trajectories, without touching simulator or learner."""
import argparse
import csv
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


def summarize(path):
    with gzip.open(path, 'rb') as stream:
        rows = pickle.load(stream)
    _, node, source, branch = path.name.removesuffix('.gz').split('-')
    t = np.asarray([r['force_time'] for r in rows])
    motion = np.asarray([np.linalg.norm(r['anchor_velocity']) > 1e-7 for r in rows])
    active = np.asarray([r['active'] for r in rows], dtype=bool)
    force, wrist, anchor, shoulder, root = (
        np.asarray([r[key] for r in rows]) for key in ('force','wrist','anchor','shoulder_xyz','root'))
    last = np.flatnonzero(active)
    masks = {'full': np.ones(len(t),dtype=bool)}
    if len(last):
        masks.update(moving=active & motion, hold=active & ~motion, release=t>t[last[-1]])
    for phase,mask in masks.items():
        if not mask.any():continue
        indices=np.flatnonzero(mask)
        lo,hi=indices[0],indices[-1]
        arm=(wrist[hi]-shoulder[hi])-(wrist[lo]-shoulder[lo])
        loaded=[np.linalg.norm(c['tangent_velocity']) for i in indices for c in rows[i].get('ctft_material_contacts',[]) if c['normal_n']>=20]
        connected=active & mask
        yield dict(node=node,source=source,branch=branch,phase=phase,ticks=len(indices),
            connected_ticks=int(connected.sum()),force_rms_N=np.mean((force[mask]**2).sum(axis=1))**.5,
            connected_error_rms_cm=100*np.mean(((wrist[connected]-anchor[connected])**2).sum(axis=1))**.5 if connected.any() else '',
            arm_x_cm=100*arm[0],arm_y_cm=100*arm[1],arm_z_cm=100*arm[2],
            root_path_cm=100*np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1)[mask[1:] & mask[:-1]].sum(),
            material_speed_p95_mps_at_20N=np.percentile(loaded,95) if loaded else '')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    rows=[r for path in sorted(args.run.glob('dev-*-*-*.gz')) for r in summarize(path)]
    if rows:
        with (args.run/'stage-readout.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
