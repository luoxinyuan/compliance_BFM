"""Saved real physical trajectories only: XYZ load, overshoot, motion and continuity."""
import argparse,gzip,json
from pathlib import Path
import numpy as np

def phase(rows):
 if not rows:return None
 f=np.array([r['force'] for r in rows]);err=np.array([np.array(r['anchor'])-r['wrist'] for r in rows]);root=np.array([r['root'] for r in rows]);w=np.array([r['wrist'] for r in rows]);contact=np.array([r['active'] for r in rows],bool)
 result=dict(samples=len(rows),force_xyz_rms=np.sqrt(np.mean(f*f,0)).tolist(),force_xyz_mean=f.mean(0).tolist(),root_delta=(root[-1]-root[0]).tolist(),wrist_delta=(w[-1]-w[0]).tolist(),wrist_relative_root_delta=((w[-1]-root[-1])-(w[0]-root[0])).tolist(),root_height_min=float(root[:,2].min()),tilt_max_deg=max(r['tilt_deg'] for r in rows),contact_samples=int(contact.sum()))
 if contact.any():
  fc=f[contact];ec=err[contact];result.update(force_rms=float(np.sqrt(np.mean(np.sum(fc*fc,1)))),cap_fraction=float(np.mean(np.linalg.norm(fc,axis=1)>=4.999)),error_cm=float(100*np.sqrt(np.mean(np.sum(ec*ec,1)))),error_xyz_rms_cm=(100*np.sqrt(np.mean(ec*ec,0))).tolist(),signed_anchor_minus_wrist_cm=(100*ec.mean(0)).tolist(),force_peak=float(np.linalg.norm(fc,axis=1).max()))
 for key in ['left_foot_force_n','right_foot_force_n','contact_tangent_speed_max_mps','joint_limit_violation_rad']:
  if key in rows[0]:result[key+'_max']=max(r[key] for r in rows)
 return result

def analyze(p):
 with gzip.open(p,'rt') as f:e=json.load(f)
 rows=e['physics'];c=e['control'];result=dict(path=str(p),label=e['label'],kind=e.get('kind'),condition=e['condition'],boundary=e['boundary'],phases={k:phase([r for r in rows if lo<=r['time']<hi]) for k,(lo,hi) in e['phases'].items()},all=phase(rows),active=phase([r for r in rows if r['active']]))
 if c:result['control']=dict(qpd_step_max_deg=float(np.rad2deg(max(r['q_PD_step_max_rad'] for r in c))),saturation_mean=float(np.mean([r['action_saturation_fraction'] for r in c])),reward_terms_integral={k:.02*sum(r['reward_terms'][k] for r in c) for k in c[0]['reward_terms']})
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 files=sorted(a.root.glob('*/job/qualification/*.json.gz'))+sorted(a.root.glob('*/job/dev-*/*.json.gz'))+sorted(a.root.glob('*/job/evaluation/*.json.gz'))
 results=[]
 for file in files:
  try:results.append(analyze(file))
  except (EOFError,json.JSONDecodeError):continue # actively-written gzip is not a completed artifact
 a.output.write_text(json.dumps(results,indent=2,allow_nan=False));print('completed_trajectories',len(results))
