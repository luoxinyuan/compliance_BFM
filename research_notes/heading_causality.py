from pathlib import Path
import json,gzip,numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/heading-transfer-20261103/B-final-06/job')
def read(s):
 with gzip.open(R/s,'rt') as f:return json.load(f)
out={}
for name in ['T30','HeadingPPO']:
 a,b=[read(name+'_prefix_'+s+'.json.gz') for s in ['left','right']];report={}
 for start,end in [(8.6,9.6),(9.6,12.05),(12.05,14.05),(14.05,20.05)]:
  states=[]
  for ep in [a,b]:
   rows=[r for r in ep['physics'] if start<=r['time']<end and (not r['active'] if start>=ep['condition']['release'] else True)];F=np.array([r['force'] for r in rows]);root=np.array([r['root'] for r in rows]);w=np.array([r['wrist'] for r in rows]);Rmat=np.array([r['pelvis_matrix'] for r in rows]);yaw=np.unwrap(np.arctan2(Rmat[:,1,0],Rmat[:,0,0]));states.append(dict(force_xyz_mean=F.mean(0).tolist(),root_delta_cm=(100*(root[-1]-root[0])).tolist(),wrist_delta_cm=(100*(w[-1]-w[0])).tolist(),wrist_root_delta_cm=(100*((w[-1]-root[-1])-(w[0]-root[0]))).tolist(),yaw_delta_deg=float(np.degrees(yaw[-1]-yaw[0]))))
  ca=[r for r in a['control'] if start<=r['time']<end];cb=[r for r in b['control'] if start<=r['time']<end];assert len(ca)==len(cb)
  za=np.array([r['executed_z'][0] for r in ca]);zb=np.array([r['executed_z'][0] for r in cb]);angle=np.degrees(np.arccos(np.clip((za*zb).sum(1)/256,-1,1)));qa=np.array([r['q_PD'] for r in ca]);qb=np.array([r['q_PD'] for r in cb]);report[str((start,end))]=dict(left=states[0],right=states[1],latent_angle_mean_deg=float(angle.mean()),latent_angle_peak_deg=float(angle.max()),paired_qpd_rms_deg=float(np.degrees(np.sqrt(np.mean((qa-qb)**2)))),scope='Different evolved physical states after real branch; not same-state action causal attribution')
 out[name]=report
(R/'causal-response-v2.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
