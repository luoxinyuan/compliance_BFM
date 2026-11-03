"""Read-only old-run geometry/reward audit. No dynamics, models or GPU."""
from pathlib import Path
import gzip,json,math,numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/heading-transfer-20261103/B-final-06/job';OUT=P/'bridge-v2-owned/runs/contact-geometry-20261103';result={}
for name in ['T30_new_left','T30_new_right','T30_prefix_left','T30_prefix_right','HeadingPPO_new_left','HeadingPPO_new_right']:
 with gzip.open(R/(name+'.json.gz'),'rt') as f:ep=json.load(f)
 rows=ep['physics'];a=[r for r in rows if r['active']];F=np.array([r['force'] for r in a]);w=np.array([r['wrist'] for r in a]);com=np.array([r['com_xyz'] for r in a]);moment=np.cross(w-com,F);dt=.005;velocity=np.gradient(w,dt,axis=0);power=(velocity*F).sum(1);root=np.array([r['root'] for r in a]);angles={}
 for key in ['pelvis_matrix','torso_matrix']:
  m=np.array([r[key] for r in a]);yaw=np.unwrap(np.arctan2(m[:,1,0],m[:,0,0]));angles[key]=dict(delta_deg=float(np.degrees(yaw[-1]-yaw[0])),range_deg=np.degrees([yaw.min(),yaw.max()]).tolist())
 spec=ep['condition'];u=np.clip((spec.get('change_time',spec['onset'])-spec['onset'])/spec['ramp'],0,1);sc=spec['distance']*(3*u*u-2*u*u*u);theta=spec['curvature']*spec['distance'] if 'change_time' not in spec else spec['curvature']*sc+spec['after_curvature']*(spec['distance']-sc)
 controls=ep['control'];terms={k:float(np.mean([r['reward_terms'][k] for r in controls])) for k in controls[0]['reward_terms']}
 result[name]=dict(path_terminal_tangent_deg=math.degrees(theta),force_xyz_mean=F.mean(0).tolist(),force_xyz_rms=np.sqrt((F*F).mean(0)).tolist(),yaw_moment_reference='saved whole-body subtree COM; force applied pre-step, positions saved post-step, ≤5ms stagger',yaw_moment_mean=float(moment[:,2].mean()),yaw_moment_rms=float(np.sqrt((moment[:,2]**2).mean())),yaw_moment_positive_fraction=float((moment[:,2]>0).mean()),work_j_finite_difference_estimate=float(power.sum()*dt),root_delta=(root[-1]-root[0]).tolist(),wrist_delta=(w[-1]-w[0]).tolist(),angles=angles,reward_terms_all_control=terms,small_angle_heading_cost_at_terminal_max=.1*(1-math.cos(theta)))
(OUT/'old-geometry-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
