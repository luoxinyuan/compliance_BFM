from pathlib import Path
import numpy as np,gzip,pickle,csv
s=Path.cwd()
with gzip.open('arm_curves.csv.gz','rt') as f: curves={(r['arm'],r['series'],round(float(r['t_s'])/.02)):float(r['wrist_shoulder_forward_m']) for r in csv.DictReader(f)}
rows=[]
for p in sorted((s/'soft').glob('*-continue.gz')):
    with gzip.open(p,'rb') as f:x=pickle.load(f)
    spec=x['spec']['partner'];ang=np.deg2rad(spec['direction_deg']);h=np.array([np.cos(ang),np.sin(ang)]);l=np.array([-h[1],h[0]])
    get=lambda k:np.asarray([r[k] for r in x['physics']]);force=get('force');t=get('force_time')-x['start'];fw=force[:,:2]@h
    yes=(np.linalg.norm(force,axis=1)>=1)&(fw>0)&(fw>=abs(force[:,:2]@l))&(fw>=abs(force[:,2]))&(t<spec['distance_m']/spec['plateau_speed_mps'])
    candidates=np.flatnonzero(np.convolve(yes.astype(int),np.ones(100,int),'valid')==100)
    lag=(get('anchor')[:,:2]-get('wrist')[:,:2])@h
    row=dict(case=p.stem[:2],lead_2s_cm=-100*lag[399],onset=None,net_forward_reach_change_cm=None)
    if len(candidates):
        i=int(candidates[0]);row['onset']=i*.005
        a=max(1,(i+3)//4);b=(i+100)//4
        def d(k):return curves['soft',p.stem,k]-curves['soft',p.stem[:2]+'-no_force',k]
        row['net_forward_reach_change_cm']=100*(d(b)-d(a))
    rows.append(row)
with open('forward-windows.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
