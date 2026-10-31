"""Read-only deployment/trajectory audit; neither BFM inference nor physics steps."""
import json
from pathlib import Path
import numpy as np
from ccl02a_teacher import audit_episode
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m1-state-response-20261031A'
def main():
 ledger=json.loads((R/'compute-ledger.json').read_text());initial=json.loads((P/'bridge-v2-owned/runs/issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json').read_text());audits=[]
 for attempt in ledger['attempts']:
  out=Path(attempt['path'])/'outputs';contract_path=out/'action-contract.json'
  if not contract_path.exists():continue
  contract=json.loads(contract_path.read_text())
  for path in sorted(out.glob('*__*.json')):
   e=json.loads(path.read_text())
   if not isinstance(e,dict) or not e.get('physics'):continue
   rows=e['physics'];cs=e['control'];tt=np.array([r['force_time'] for r in rows]);ct=e['condition'];on=ct['onset'];rel=ct['release'];q=np.array([c['q_PD'] for c in cs]);z=np.array([c['executed_z'][0] for c in cs]);force=np.array([r['force'] for r in rows]);fn=np.linalg.norm(force,axis=-1)
   # On an unsafe interrupt the final control tick can be partial, but every captured physics sample remains auditable.
   parity=audit_episode(e,contract,initial)
   assert fn.max()<=5.000001
   i,j=np.unravel_index(np.abs(np.diff(q,axis=0)).argmax(),(len(q)-1,29))
   item=dict(source=str(path),boundary=e['boundary'],parity=parity,latent_norm_max_error=float(abs(np.linalg.norm(z,axis=-1)-16).max()),largest_command_step=dict(time=cs[i+1]['time'],joint=contract['dof_names'][j],degrees=float(np.degrees(abs(q[i+1,j]-q[i,j]))),latent_step_deg=cs[i+1]['z_step_deg']),phases={})
   for name,lo,hi in [('entry',3,3.75),('settle',3.75,on),('contact_window',on,rel),('recovery',rel,rel+2)]:
    ix=np.flatnonzero((tt>=lo-1e-8)&(tt<hi-1e-8))
    if not len(ix):continue
    rr=[rows[k] for k in ix];w=np.array([r['wrist'] for r in rr]);feet=np.array([r['feet_xyz'] for r in rr]);vel=np.array([r['qvel'] for r in rr]);qp=np.array([r['qpos'] for r in rr]);start=rows[ix[0]-1] if ix[0]>0 else e['preparation_initial'];point0=np.array(start['wrist']);root0=np.array(start['root'])
    item['phases'][name]=dict(wrist_range_cm=((w.max(0)-w.min(0))*100).tolist(),wrist_delta_cm=((w[-1]-point0)*100).tolist(),root_delta_cm=((qp[-1,:3]-root0)*100).tolist(),actual_right_arm_change_deg=np.degrees(qp[-1,29:36]-np.array(start['qpos'])[29:36]).tolist(),actual_waist_change_deg=np.degrees(qp[-1,19:22]-np.array(start['qpos'])[19:22]).tolist(),foot_displacement_xy_max_cm=float(np.linalg.norm(feet[:,:,:2]-np.array(start['feet_xyz'])[None,:,:2],axis=-1).max()*100),root_speed_peak_mps=float(np.linalg.norm(vel[:,:3],axis=-1).max()))
   audits.append(item)
 dst=R/'analysis';dst.mkdir(exist_ok=True);(dst/'deployment-audit.json').write_text(json.dumps(audits,indent=2,allow_nan=False));print(json.dumps([dict(source=a['source'],parity=a['parity'],norm_error=a['latent_norm_max_error'],largest_command_step=a['largest_command_step']) for a in audits],indent=2))
if __name__=='__main__':main()
