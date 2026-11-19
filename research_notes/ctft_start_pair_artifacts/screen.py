import gzip,json,pickle,hashlib,time
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
root=p.parent/'bfm_ctft_v1/runs/s0-native-sources-03'
cases=json.loads((root/'source-summary.json').read_text())['sources'];out=[]
for case in cases:
 f=root/case['evidence_file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==case['evidence_sha256']
 with gzip.open(f) as h:x=pickle.load(h)
 rows=x['physics'];a=lambda k:np.asarray([r[k] for r in rows])
 force=a('force');rootxy=a('root')[:,:2];wrist=a('wrist');v=np.linalg.norm(a('qvel')[:,:2],axis=1);R=a('pelvis_matrix');yaw=np.unwrap(np.arctan2(R[:,1,0],R[:,0,0]));load=a('foot_normal_force_n').max(1)
 stages=[];eligible=[]
 for i in range(500,min(len(x['states']),len(rows)//4+1)):
  start=4*i-200;k=4*i
  sl=slice(start,k)
  vals=[float(np.percentile(v[sl],95)),float(np.linalg.norm(np.diff(rootxy[sl],axis=0),axis=1).sum()),float(np.linalg.norm(np.diff(wrist[sl],axis=0),axis=1).sum()),float(np.rad2deg(np.ptp(yaw[sl]))),float(abs(force[sl]).max()),float(load[sl].min())]
  good=all(a<=b for a,b in zip(vals[:5],[.05,.03,.04,3.,0.])) and vals[5]>=.05*x['states'][i]['online_support']['weight_n']
  stages.append(vals);eligible+=([i] if good else [])
 out.append(dict(case=case['case'],ancestor=case['ancestor'],evaluated=len(stages),eligible=len(eligible),first=eligible[0] if eligible else None,first_metrics=stages[eligible[0]-500] if eligible else None,mins=np.min(stages,0).tolist() if stages else None,hard=case['boundary_kind']))
 print(out[-1]['case'],len(stages),len(eligible),flush=True);(p/'origins.json').write_text(json.dumps(out,indent=1))
