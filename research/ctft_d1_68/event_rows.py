"""Historical stochastic training context; never a frozen-U counterfactual."""
from pathlib import Path
from collections import Counter
import gzip,json,pickle,sys,time,math
import numpy as np
import torch
s=Path(sys.argv[1]);out=Path(sys.argv[2]); t=time.monotonic()
d=s/'runs/formal-U-v1a-01'; ep=json.loads((d/'train-only-check-131072.json').read_bytes())['source_episodes']
offset=0; picks={}; samples=[]
for e in ep:
    spec=e['spec'];n=e['controls'];begin=offset;offset+=n
    if begin<8192 or not spec['mode'] or not e['complete']:continue
    b=spec['branch']; p=spec['partner']; release=p['distance_m']/p['plateau_speed_mps']+p['accel_s']+p['hold_s']
    candidates={'no_force':[('never_force',.5)],'stop':[('connected_hold',release-p['hold_s']/2),('actual_release',release+.5)],'continue':[('loaded',.5)]}[b]
    for label,when in candidates:
        key=(spec['source']['case'],label)
        if key in picks:continue
        pos=begin+round(when/.02)
        assert begin<=pos<offset
        picks[key]=(pos,spec,when)
for blk in sorted({(p[0]//1024+1)*1024 for p in picks.values()}):
    v=torch.load(d/f'actual-rows-{blk:06d}.pt',map_location='cpu',weights_only=False)
    with gzip.open(d/f'physics-{blk}.pkl.gz','rb') as f:ph=pickle.load(f)
    for (case,phase),(pos,spec,when) in picks.items():
        i=pos-(blk-1024)
        if not 0<=i<1024:continue
        part={k:float(x[i].item()) for k,x in v['ctft_reward_components'].items()}
        # Include all 4 physical ticks of this actually executed command.
        for j,r in enumerate(ph[4*i:4*i+4]):
            samples.append(dict(case=case,parent=spec['parent'],phase=phase,domain='historical_U_stochastic',row=pos,tick=j,force_time=float(r['force_time']),active=bool(r['active']),applied_force=r['force'],foot_normal_N=r['foot_normal_force_n'],q=r['qpos'],dq=r['qvel'],root=r['root'],wrist=r['wrist'],action=v['action'][i].flatten().tolist(),native_aux=part['native_aux'],contact_aux=part['contact_aux']))
def conv(x):return np.asarray(x).tolist()
(out/'training-events.json').write_text(json.dumps(samples,default=conv,allow_nan=False))
(out/'training-events-cost.json').write_text(json.dumps(dict(wall_s=time.monotonic()-t,physics_ticks=0,n_events=len(picks),n_rows=len(samples),phase_counts=dict(Counter(x[1] for x in picks)))))
