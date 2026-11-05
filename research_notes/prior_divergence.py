"""Descriptive closed-loop divergence; not same-state intervention evidence."""
import json,gzip,sys
from pathlib import Path
import numpy as np
r=Path(sys.argv[1]);a=json.load(gzip.open(r/'fit-A/A_walk.json.gz','rt'));b=json.load(gzip.open(r/'fit-B/B_walk.json.gz','rt'));rows=[]
for t in [4.,4.2,4.3,4.5,5.,6.,7.]:
 ca=min(a['control'],key=lambda c:abs(c['time']-t));cb=min(b['control'],key=lambda c:abs(c['time']-t));pa=min(a['physics'],key=lambda c:abs(c['time']-t));pb=min(b['physics'],key=lambda c:abs(c['time']-t));za=np.array(ca['executed_z']);zb=np.array(cb['executed_z'])
 rows.append(dict(time=t,z_deg=float(np.rad2deg(np.arccos(np.clip((za*zb).sum()/256,-1,1)))),qpd_rms_deg=float(np.sqrt(np.mean(np.rad2deg(np.array(ca['q_PD'])-cb['q_PD'])**2))),root_separation_cm=float(100*np.linalg.norm(np.array(pa['root'])-pb['root'])),scope='Different evolving trajectories; descriptive, not same-state causal action comparison'))
(r/'first-initialization-divergence.json').write_text(json.dumps(rows,indent=2))
