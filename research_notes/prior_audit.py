"""Read-only reward decomposition of already recorded actual control/physics."""
import json,gzip,math,hashlib
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
def audit(root,out):
 rows=[]
 for path in sorted(root.glob('physical-*/*.json.gz')):
  if path.name.endswith('-source-prefix.json.gz') or path.name=='timelines.json.gz':continue
  ep=json.load(gzip.open(path,'rt'))
  if not isinstance(ep,dict) or 'rl_stats' not in ep:continue
  s=ep['condition'];st=ep['rl_stats'];ph=ep['physics'];cs=ep['control'];offset=0;segs={};maxerr=0
  for r in st:
   ps=ph[offset:offset+r['physics_steps']];offset+=r['physics_steps'];t=ps[0]['force_time'];stage='pre' if t<s['onset'] else 'recover' if t>=s['release'] else 'move' if t<s['onset']+s['ramp'] else 'hold';dur=.02*r['control_steps'];terms=r['terms'];expected=-dur*sum(terms.values());maxerr=max(maxerr,abs(expected-r['reward']))
   if r['boundary']:continue
   seg=segs.setdefault(stage,dict(seconds=0,return_value=0,cost_integrals={}))
   seg['seconds']+=dur;seg['return_value']+=r['reward']
   for k,v in terms.items():seg['cost_integrals'][k]=seg['cost_integrals'].get(k,0)+dur*v
  active=[r for r in ph if r['active']];pre=min(ph,key=lambda r:abs(r['time']-s['onset']));end=min(ph,key=lambda r:abs(r['time']-s['release']));fe=np.asarray([r['feet_xyz'] for r in ph]);root=np.asarray([r['root'] for r in ph]);q=np.asarray([r['qpos'][7:] for r in ph]);qd=np.asarray([r['qvel'][6:] for r in ph]);
  rows.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),label=ep['label'],condition=s,phases=segs,reward_sum=sum(r['reward'] for r in st),reward_integral_max_error=maxerr,foot_xy_path_m=np.linalg.norm(np.diff(fe[:,:,:2],axis=0),axis=-1).sum(0).tolist(),root_xy_path_m=float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1).sum()),root_contact_end_delta_cm=(100*(np.asarray(end['root'])-pre['root'])).tolist(),joint_travel_rad=float(abs(np.diff(q,axis=0)).sum()),joint_squared_speed_integral=float((qd*qd).sum()*.005),max_tilt=max(r['tilt_deg'] for r in ph),qpd_max_deg=float(np.degrees(max(c['q_PD_step_max_rad'] for c in cs))),actual_q_max_deg=float(np.degrees(abs(np.diff(q,axis=0))).max()),boundary=ep['boundary']))
 out.write_text(json.dumps(rows,indent=2));print('AUDIT',len(rows),max(r['reward_integral_max_error'] for r in rows))
if __name__=='__main__':
 import sys;audit(Path(sys.argv[1]),Path(sys.argv[2]))
