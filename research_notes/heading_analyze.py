"""Actual-force and active-window metrics; released anchor error excluded from success."""
from pathlib import Path
import json,gzip,math,numpy as np

def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def metrics(ep):
 ph=ep['physics'];ctl=ep.get('control',[]);s=ep['condition'];out={}
 def stage(r):
  t=r.get('force_time',r['time'])
  # Rows are logged after the 5ms physics step. The last pre-contact row can
  # have time==onset while its applied force_time was still before onset.
  # Never concatenate that row with release and count the whole walk as recovery.
  if not r['active']:return 'free_recovery' if t>=s['release']-1e-8 else 'pre'
  if 'stop_time' in s:return 'move' if t<s['stop_time'] else 'decelerate' if t<s['stop_time']+s['deceleration'] else 'hold'
  return 'move' if t<s['onset']+s['ramp'] else 'hold'
 for name in ['contact','move','decelerate','hold','free_recovery','pre']:
  rs=[r for r in ph if (r['active'] if name=='contact' else stage(r)==name)]
  if not rs:continue
  E=np.array([np.array(r['anchor'])-r['wrist'] for r in rs]);F=np.array([r['force'] for r in rs]);root=np.array([r['root'] for r in rs]);w=np.array([r['wrist'] for r in rs]);qv=np.array([r['qvel'] for r in rs]);ts=np.array([r['time'] for r in rs]);R=np.array([r['pelvis_matrix'] for r in rs]);yaw=np.unwrap(np.arctan2(R[:,1,0],R[:,0,0]));vp=np.array([r['anchor_velocity'] for r in rs]);moving=np.linalg.norm(vp[:,:2],axis=1)>.02
  h=np.arctan2(vp[:,1],vp[:,0]);he=np.arctan2(np.sin(yaw-h),np.cos(yaw-h));c=[r for r in ctl if ts[0]-.02<=r['time']<=ts[-1]];jumps=np.degrees([r['q_PD_step_max_rad'] for r in c]);vel=np.diff(w,axis=0)/np.diff(ts)[:,None] if len(rs)>1 else np.zeros((1,3))
  out[name]=dict(samples=len(rs),seconds=len(rs)/200,force_rms=float(np.sqrt((F*F).sum(1).mean())),force_xyz_mean=F.mean(0).tolist(),force_xyz_rms=np.sqrt((F*F).mean(0)).tolist(),cap_fraction=float((np.linalg.norm(F,axis=1)>=4.999).mean()),root_delta_cm=(100*(root[-1]-root[0])).tolist(),root_path_cm=float(100*np.linalg.norm(np.diff(root,axis=0),axis=1).sum()),wrist_delta_cm=(100*(w[-1]-w[0])).tolist(),wrist_relative_root_delta_cm=(100*((w[-1]-root[-1])-(w[0]-root[0]))).tolist(),root_speed_rms=float(np.sqrt((qv[:,:3]**2).sum(1).mean())),wrist_speed_rms=float(np.sqrt((vel*vel).sum(1).mean())),root_height_min=float(root[:,2].min()),tilt_max_deg=max(r['tilt_deg'] for r in rs),heading_delta_deg=float(np.degrees(yaw[-1]-yaw[0])),heading_end_deg=float(np.degrees(yaw[-1])),moving_heading_rms_deg=float(np.degrees(np.sqrt(np.mean(he[moving]**2)))) if moving.any() and name not in ['free_recovery','pre'] else None,unsupported_fraction=float(np.mean([not any(r['support']) for r in rs])),slip_p95_mps=float(np.quantile([r['contact_tangent_speed_max_mps'] for r in rs],.95)),torque_saturation_mean=float(np.mean([r['torque_saturation'] for r in rs])),qpd_jump_max_deg=float(max(jumps)) if len(jumps) else None,qpd_jump_p95_deg=float(np.quantile(jumps,.95)) if len(jumps) else None)
  if name not in ['free_recovery','pre']:out[name].update(error_cm=float(100*np.sqrt((E*E).sum(1).mean())),signed_xyz_cm=(100*E.mean(0)).tolist())
 return dict(label=ep['label'],condition=s,boundary=ep['boundary'],phases=out)
if __name__=='__main__':
 import sys
 root=Path(sys.argv[1]);result={}
 for f in root.rglob('*.json.gz'):
  if 'queries' in f.name or 'source' in f.name or 'accepted' in f.name or 'data' in f.name:continue
  ep=read(f)
  if isinstance(ep,dict) and 'physics' in ep:result[str(f.relative_to(root))]=metrics(ep)
 (root/'physical-metrics-v2.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('episodes',len(result))
