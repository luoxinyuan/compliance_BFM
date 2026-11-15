"""Issue66 H0 fixed result contract; no actor, scalar-Q, or deployment truth."""
import json,hashlib,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];S=ROOT.parent;P=ROOT.parents[2]
BFM='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
SCALES=np.array([.1]*9+[10.]*3+[.5]*6+[.35,.45,15.,1.,1.,.02],dtype=np.float32)
FIELDS=['wrist_delta_xyz','root_delta_xyz','wrist_relative_root_delta_xyz','mean_applied_force_xyz','final_wrist_velocity_xyz','final_root_velocity_xyz','pelvis_yaw_delta_rad','loaded_slip_peak_mps','tilt_peak_deg','supported_fraction','saturated_fraction','joint_excess_rad']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)));os.replace(tmp,p)
def result(ph):
 assert len(ph)==400
 a,b=ph[0],ph[-1];w=np.array(b['wrist'])-a['wrist'];r=np.array(b['root'])-a['root']
 R=np.array(a['pelvis_matrix']);world_to_start=R.T
 yaw=np.unwrap([np.arctan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0]) for p in ph])
 f=np.array([p['force'] for p in ph]);ac=[p for p in ph if p['active']]
 v=np.array(b['contacts'][0]['velocity']);vr=np.array(b['qvel'][:3])
 y=np.r_[world_to_start@w,world_to_start@r,world_to_start@(w-r),world_to_start@f.mean(0),world_to_start@v,world_to_start@vr,yaw[-1]-yaw[0],max(p['stage_loaded_slip_mps'] for p in ph),max(p['tilt_deg'] for p in ph),np.mean([any(p['support']) for p in ph]),np.mean([np.linalg.norm(p['force'])>=9.999 for p in ac]) if ac else 0.,max(p['joint_limit_excess_max_rad'] for p in ph)]
 assert len(y)==24 and np.isfinite(y).all()
 return y.astype(np.float32)
def target(follow,local_direction):
 y=np.zeros(24,dtype=np.float32);y[:3]=np.array(local_direction)*(.10 if follow else 0.);y[6:9]=y[:3];y[21]=1.;return y
def error_audit(ph,K):
 er=ef=0.
 for p in ph:
  for c in p['contacts']:
   raw=K*(np.array(c['anchor'])-c['point'])+8*(np.array(c['anchor_velocity'])-c['velocity']) if c['active'] else np.zeros(3)
   f=raw*min(1.,10/max(np.linalg.norm(raw),1e-30));er=max(er,float(np.max(abs(raw-c['raw_force']))));ef=max(ef,float(np.max(abs(f-c['force']))))
 assert er<=1e-12 and ef<=1e-12,(er,ef)
 return dict(raw_linf=er,applied_linf=ef,K=K,D=8,cap=10)
