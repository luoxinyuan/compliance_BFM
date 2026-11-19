"""Replay-free paired onset readout from deterministic 5ms records."""
import csv
import gzip
import pickle
import sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
writer = csv.writer(sys.stdout)
writer.writerow(['node','source','branch','seconds','torso_forward_cm','right_arm_forward_cm',
                 'left_arm_forward_cm','right_arm_in_cm','left_arm_in_cm','wrist_forward_cm','anchor_forward_cm'])

def positions(row):
    p = row['body_positions']
    torso_rotation = np.asarray(row['torso_matrix'])
    arms = [torso_rotation.T @ (np.asarray(p[s+'_wrist_roll_link']) - p[s+'_shoulder_pitch_link'])
            for s in ('right','left')]
    return np.array([p['torso_link'][0], arms[0][0], arms[1][0], arms[0][1], -arms[1][1],
                     row['wrist'][0],row['anchor'][0]])

for node in (0,12571,16384):
    for source in (9,10):
        for branch in ('front','none'):
            with gzip.open(run/f'dev-{node}-{source}-{branch}.gz','rb') as stream:
                rows=pickle.load(stream)
            times=np.array([r['force_time'] for r in rows])
            for delay in (.25,.5,1.,2.1):
                i=np.abs(times-times[0]-delay).argmin()
                assert abs(times[i]-times[0]-delay)<1e-7
                writer.writerow([node,source,branch,delay,*((positions(rows[i])-positions(rows[0]))*100)])
