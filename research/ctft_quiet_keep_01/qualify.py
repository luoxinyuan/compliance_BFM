"""Read the final 1 s of each train-only native source, never holdout paths."""
import gzip, hashlib, json, pickle
from pathlib import Path
import numpy as np
import torch
stage=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/bfm_ctft_v1')
records=json.loads((stage/'runs/s0-source-admission-01/result.json').read_text())['train']
out=[]
for r in records:
    p=stage/'runs/s0-native-sources-03'/r['evidence_file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256']
    with gzip.open(p,'rb') as f:x=pickle.load(f)
    rows=x['physics'][-200:]; ctr=x['controls']; ref=ctr[0]['previous_z']
    a=lambda k:np.asarray([v[k] for v in rows])
    yaw=np.unwrap(np.arctan2(a('pelvis_matrix')[:,1,0],a('pelvis_matrix')[:,0,0]))
    item=dict(case=r['case'],ancestor=r['ancestor'],complete=x['boundary'] is None,standing_z=all(torch.equal(c['z'].reshape(-1),ref.reshape(-1)) for c in ctr[-50:]),ticks=len(rows),speed95=float(np.percentile(np.linalg.norm(a('qvel')[:,:2],axis=1),95)),rootpath=float(np.linalg.norm(np.diff(a('root')[:,:2],axis=0),axis=1).sum()),wristpath=float(np.linalg.norm(np.diff(a('wrist'),axis=0),axis=1).sum()),yaw=float(np.rad2deg(np.ptp(yaw))),max_force=float(abs(a('force')).max()),min_peak_foot_N=float(a('foot_normal_force_n').max(axis=1).min()))
    item['prequalified']=bool(item['complete'] and item['standing_z'] and item['ticks']==200 and item['speed95']<=.05 and item['rootpath']<=.03 and item['wristpath']<=.04 and item['yaw']<=3 and item['max_force']==0 and item['min_peak_foot_N']>=30)
    out.append(item)
    print(item['case'],item['complete'],item['prequalified'],flush=True)
    del x,rows,ctr
Path('train-source-audit.json').write_text(json.dumps(out,indent=2))
