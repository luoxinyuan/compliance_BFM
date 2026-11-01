"""One deterministic kinematic cycle extraction per already selected reference.
No new motion/latent search. Root XY and global yaw are intentionally excluded
from closure cost because locomotion translates and turns; body velocities remain.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m9-contact-locomotion-20261101A')
def build():
 rows={}
 for name in ['straight','left','right']:
  r=json.loads((P/'tracking-attempt-01/outputs'/f'{name}-reference.json').read_text());q=np.array(r['qpos']);v=np.array(r['qvel_world_root_angular']);z=np.array(r['z']);R=Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix();bodyv=np.einsum('nji,nj->ni',R,v[:,:3]);bodyw=np.einsum('nji,nj->ni',R,v[:,3:6]);gravity=R[:,2,:]
  h=np.c_[q[:,7:]/.3,v[:,6:]/3,q[:,2]/.1,gravity/.2,bodyv/.4,bodyw]
  pairs=[(float(np.square(h[j]-h[i]).mean()),i,j) for i in range(len(h)-40) for j in range(i+40,min(i+91,len(h)))];cost,i,j=min(pairs)
  rows[name]=dict(start_index=i,end_index=j,period=(j-i)*.02,closure_scaled_rms=cost**.5,closure_joint_max_deg=float(np.degrees(abs(q[j,7:]-q[i,7:])).max()),closure_z_angle_deg=float(np.degrees(np.arccos(np.clip(z[j]@z[i]/256,-1,1)))),z=z[i:j].tolist(),q=q[i:j].tolist(),phase_offset=0,source=r['source'])
 # Align left/right cycles to straight's leg phase, not to task IDs or observed
 # rollout success. Single deterministic circular offset minimizing leg angles.
 base=np.array(rows['straight']['q']);phase=np.arange(64)/64
 for name in ['left','right']:
  other=np.array(rows[name]['q']);b=base[(phase*len(base)).astype(int),7:19];scores=[]
  for shift in range(len(other)):
   o=other[((phase*len(other)).astype(int)+shift)%len(other),7:19];scores.append(float(np.square(b-o).mean()))
  rows[name]['phase_offset']=int(np.argmin(scores));rows[name]['leg_phase_alignment_rms_rad']=min(scores)**.5
 report=dict(method='Single kinematic closure pair within .8-1.8 seconds per fixed6s reference, original velocities/latents; phase aligned from leg pose; no joint smoothing or BFM edit',cycles=rows)
 (P/'native-cycles.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a not in ['q','z']} for k,v in rows.items()},indent=2))
if __name__=='__main__':build()
