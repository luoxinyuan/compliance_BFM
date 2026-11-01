from pathlib import Path
import json,gzip,sys
import numpy as np,mujoco
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs';ROOT=P/'bridge-v2-owned/trees/m10-task-appropriate/repo';OUT=R/'m10-task-appropriate-20261101'
m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
names=['pelvis','torso_link','left_wrist_roll_link','right_wrist_roll_link'];ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in names]
def metrics(path):
 e=json.loads(gzip.decompress(path.read_bytes()));rs=e['physics'];cs=e['control'];spec=e.get('condition',{});on=spec.get('onset',4.5);release=spec.get('release',10.5);active=[r for r in rs if r.get('active')];summary={}
 if active:
  F=np.array([r['force'] for r in active]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);summary.update(error_cm=float(np.sqrt((E**2).sum(1).mean())*100),error_xyz_cm=(np.sqrt((E**2).mean(0))*100).tolist(),force_rms=float(np.sqrt((F**2).sum(1).mean())),force_xyz_rms=np.sqrt((F**2).mean(0)).tolist(),cap_fraction=float((np.linalg.norm(F,axis=1)>4.999).mean()))
 fk=[]
 for r in [e['initial']]+rs[::4]:
  d.qpos[:]=r['qpos'];mujoco.mj_forward(m,d);A=d.xmat[ids[0]].reshape(3,3);B=d.xmat[ids[1]].reshape(3,3);pitch=lambda x:float(np.rad2deg(np.arctan2(-x[2,0],x[2,2])))
  fk.append(dict(t=r['time'],pelvis_pitch=pitch(A),torso_pitch=pitch(B),waist_pitch=pitch(A.T@B),left=(A.T@(d.xpos[ids[2]]-d.xpos[ids[0]])).tolist(),right=(A.T@(d.xpos[ids[3]]-d.xpos[ids[0]])).tolist()))
 for phase,lo,hi in [('entry',on,on+2),('contact',on,release),('recover',release,rs[-1]['time']+.01)]:
  ss=[r for r in fk if lo<=r['t']<hi]
  if not ss:continue
  stats={n:float(np.mean([r[n] for r in ss])) for n in ['pelvis_pitch','torso_pitch','waist_pitch']}
  phys=[r for r in rs if lo<=r['time']<hi]
  for arm in ['left','right']:
   key=arm+'_wrist_roll_link';xyz=np.array([r['body_positions'][key] for r in phys]);ready=np.array(e['initial']['body_positions'][key]);stats[arm+'_world_xyz_delta_min_cm']=(100*(xyz.min(0)-ready)).tolist();stats[arm+'_world_xyz_delta_max_cm']=(100*(xyz.max(0)-ready)).tolist()
  stats['root_xyz_delta_range_m']=np.ptp(np.array([r['root'] for r in phys]),axis=0).tolist()
  stats['shoulder_xyz_range_cm']=(100*np.ptp(np.array([r['body_positions']['right_shoulder_pitch_link'] for r in phys]),axis=0)).tolist()
  for arm in ['left','right']:
   xyz=np.array([r[arm] for r in ss]);stats[arm+'_rel_z_change_min_cm']=float((xyz[:,2].min()-fk[0][arm][2])*100);stats[arm+'_rel_z_change_max_cm']=float((xyz[:,2].max()-fk[0][arm][2])*100)
  summary[phase]=stats
 q=np.array([r['q_PD'] for r in cs]);root=np.array([r['root'] for r in rs]);summary.update(boundary=e.get('boundary'),qpd_step_max_deg=float(np.rad2deg(abs(np.diff(q,axis=0)).max())),root_end_delta=(root[-1]-root[0]).tolist(),root_min=float(root[:,2].min()),tilt_max=max(r['tilt_deg'] for r in rs),joint_limit_max=max(r.get('joint_limit_excess_max_rad',0) for r in rs),torque_saturation=max(r.get('torque_saturation',0) for r in rs),single_support_fraction=float(np.mean([sum(r['support'])==1 for r in rs])),source=str(path))
 return summary
paths=list(OUT.glob('*-attempt-*/outputs/*__*.json.gz'))
paths+=list((R/'m9-contact-locomotion-20261101A/expert-attempt-01/outputs').glob('cycle-*.json.gz'))
result={str(p.relative_to(R)):metrics(p) for p in paths};(OUT/'live-metrics.json').write_text(json.dumps(result,indent=2))
for k,v in result.items():print(Path(k).name, {a:v.get(a) for a in ['error_cm','force_rms','cap_fraction','boundary','root_end_delta','qpd_step_max_deg']},v.get('contact'))
