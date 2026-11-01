"""Bounded existing-motion geometry inventory, not latent search or simulation."""
import json,joblib,hashlib
from pathlib import Path
import numpy as np,mujoco
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');ROOT=P/'bridge-v2-owned/trees/m7-continuous/repo';RUN=P/'bridge-v2-owned/runs/m8-contact-body-20261101A'
asset=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/data/lafan_29dof.pkl');motions=joblib.load(asset)
m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
bid=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
feet=[bid(s+'_ankle_roll_link') for s in ['left','right']];torso=bid('torso_link');names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,m.njnt)]
ki=[names.index(s+'_knee_joint') for s in ['left','right']];hip=[names.index(s+'_hip_pitch_joint') for s in ['left','right']]
rows=[];stats=[]
for name,v in motions.items():
 n=len(v['dof']);fps=v['fps'];candidates=[]
 for i in range(10,n-10,10):
  q=np.r_[v['root_trans_offset'][i],v['root_rot'][i][[3,0,1,2]],v['dof'][i]]
  d.qpos[:]=q;mujoco.mj_forward(m,d)
  tilt=np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3,3)[2,2],-1,1)));knee=v['dof'][i,ki]
  if not (.45<q[2]<.71 and max(knee)<1.8 and min(knee)>.35 and tilt<15):continue
  footz=d.xpos[feet,2];flat=[np.degrees(np.arccos(np.clip(d.xmat[b].reshape(3,3)[2,2],-1,1))) for b in feet]
  speed=np.linalg.norm(v['root_trans_offset'][i+5]-v['root_trans_offset'][i-5])*fps/10
  if abs(footz[0]-footz[1])>.04 or max(flat)>20:continue
  if speed>.2 or np.linalg.norm(d.xpos[feet[0],:2]-d.xpos[feet[1],:2])>.38:continue
  score=abs(q[2]-.65)+.3*abs(knee[0]-knee[1])+.03*speed+.002*tilt+.001*sum(flat)
  candidates.append(dict(motion=name,frame=i,fps=fps,score=float(score),root_height=float(q[2]),root_speed=float(speed),tilt=float(tilt),knee=knee.tolist(),hip=v['dof'][i,hip].tolist(),feet=d.xpos[feet].tolist(),foot_tilt_deg=flat,torso=d.xpos[torso].tolist(),qpos=q.tolist()))
 candidates.sort(key=lambda x:x['score']);stats.append(dict(motion=name,frames=n,eligible=len(candidates),best=candidates[:1]));rows+=candidates[:1]
rows.sort(key=lambda x:x['score']);result=dict(asset=str(asset),asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),frames_examined='stride10, all40existingmotions; geometry only; no BFM search',stats=stats,ranked=rows[:8],selected=rows[:2],method='Choose up to two natural bilateral flexed low-root frames by fixed geometry score; no direct live qpos reset. Original goal_inference used later to enter from standing.')
RUN.mkdir(exist_ok=True);(RUN/'asset-audit-static.json').write_text(json.dumps(result,indent=2));print(json.dumps([ {k:v for k,v in r.items() if k!='qpos'} for r in result['ranked']],indent=2))
