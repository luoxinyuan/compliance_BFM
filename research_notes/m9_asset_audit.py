"""One bounded existing-motion scan; geometry is not execution evidence."""
import json,joblib,numpy as np
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m9-contact-locomotion-20261101A'
a=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/data/lafan_29dof.pkl');motions=joblib.load(a);families={k:[] for k in ['straight','left','right']};inventory=[]
for name,v in motions.items():
 p=np.asarray(v['root_trans_offset']);q=np.asarray(v['root_rot']);fps=v['fps'];yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)));tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,0]**2+q[:,1]**2),-1,1)));vel=np.gradient(p,1/fps,axis=0);speed=np.linalg.norm(vel[:,:2],axis=1)
 inventory.append(dict(name=name,frames=len(p),fps=fps,speed_mean=float(speed.mean()),yaw_range_deg=float(np.degrees(yaw.max()-yaw.min()))))
 if not name.startswith('walk'):continue
 length=int(6*fps)
 for i in range(int(fps),len(p)-length-int(fps),int(fps)):
  j=i+length;sp=float(np.mean(speed[i:j]));dy=float(np.degrees(yaw[j]-yaw[i]));alignment=np.mean((vel[i:j,0]*np.cos(yaw[i:j])+vel[i:j,1]*np.sin(yaw[i:j]))/(speed[i:j]+1e-6));root=float(p[i:j,2].min());ti=float(tilt[i:j].max())
  if not (.15<=sp<=.85 and root>.65 and ti<20 and alignment>.6):continue
  family='straight' if abs(dy)<10 else 'left' if 20<=dy<=90 else 'right' if -90<=dy<=-20 else None
  if family is None:continue
  row=dict(motion=name,start=i,end=j,fps=fps,mean_speed=sp,yaw_change_deg=dy,root_min=root,tilt_max=ti,heading_velocity_alignment=float(alignment),score=abs(sp-.35)+.002*abs(dy-(0 if family=='straight' else 40 if family=='left' else -40)))
  families[family].append(row)
for f in families:families[f].sort(key=lambda x:x['score'])
x=dict(asset=str(a),asset_sha256='f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11',inventory=inventory,selected={k:v[0] if v else None for k,v in families.items()},candidates={k:v[:3] for k,v in families.items()},scope='6s windows/1s stride, existing walk motions only, geometric selection fixed before BFM execution. No live qpos reset or latent sampling.')
(RUN/'motion-inventory.json').write_text(json.dumps(x,indent=2));print(json.dumps(x['selected'],indent=2))
