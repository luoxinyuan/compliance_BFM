import sys,json,time,hashlib,math
from pathlib import Path
import numpy as np,torch,mujoco
start=time.monotonic();S=Path(sys.argv[1]);R=S/'runtime';sys.path.insert(0,str(R));run=S/'runs/physical-01';out=S/'analysis02';rows=json.loads((run/'results.json').read_text())
m=mujoco.MjModel.from_xml_path(str(R/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
shoulder=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_shoulder_pitch_link');wrist=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');torso=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'torso_link');assert min(shoulder,wrist,torso)>0
extra=[];pair={}
for row in rows:
 if row['rep']:continue
 x=torch.load(run/row['artifact'],map_location='cpu',weights_only=False);cut=x['start']+3-1e-8;ph=[p for p in x['physics'] if p['force_time']<cut];rel=[];relT=[];shoulders=[]
 for p in ph[::4]:
  d.qpos[:]=p['qpos'];d.qvel[:]=p['qvel'];mujoco.mj_forward(m,d);R0=d.xmat[torso].reshape(3,3);rel.append(R0.T@(d.xpos[wrist]-d.xpos[shoulder]));relT.append(R0.T@(d.xpos[wrist]-d.xpos[torso]));shoulders.append(d.xpos[shoulder].copy())
 rel=np.asarray(rel);relT=np.asarray(relT);shoulders=np.asarray(shoulders)
 extra.append(dict(state=row['state_id'],controller=row['controller'],wrist_relative_shoulder_delta_body_m=(rel[-1]-rel[0]).tolist(),wrist_relative_torso_delta_body_m=(relT[-1]-relT[0]).tolist(),wrist_relative_shoulder_path_m=float(np.linalg.norm(np.diff(rel,axis=0),axis=1).sum()),shoulder_world_dx_m=float(shoulders[-1,0]-shoulders[0,0]),FK_sample_dt_s=.02,physical_steps=0))
 if row['state_id'].startswith('new_stop-'):pair[(row['controller'],row['state_id'].endswith('-continue'))]=x
prefix=[]
for kind in ['BC500','CS','CP']:
 a,b=pair[kind,False],pair[kind,True];first=None;gd=None
 for x,y in zip(a['decisions'],b['decisions']):
  if first is None and not torch.equal(x['raw'],y['raw']):first=x['time']
  if gd is None and x['guidance']!=y['guidance']:gd=x['time']
 prefix.append(dict(controller=kind,actual_partner_branch_s=a['spec']['stop_time'],first_raw_difference_s=first,first_guidance_difference_s=gd,pre_branch_exact=True))
(out/'kinematics.json').write_text(json.dumps(extra,indent=2));(out/'branch-timing.json').write_text(json.dumps(prefix,indent=2));(out/'extra-cost.json').write_text(json.dumps(dict(job_s=time.monotonic()-start,physics_steps=0,kind='saved qpos FK and saved command comparison'),indent=2));print(prefix)
