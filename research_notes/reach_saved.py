import json,gzip,sys
from pathlib import Path
import numpy as np,mujoco
class Writer:
 @staticmethod
 def write(p,x):p.write_text(json.dumps(x,indent=2))
runner=Writer()
W=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(W/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
c=json.loads((out.parent/'physical-01/joint-contract.json').read_text());names=c['names'];qa=np.array(c['qpos_addresses']);limits=np.array(c['limits']);arm=c['arm_indices'];bodies=c['bodies'];timelines={};summaries=[]
def analyze(ep,label):
  p=ep['physics'];s=ep['condition'];on=s['onset'];rr=[]
  for r in p[::4]:
   d.qpos[:]=r['qpos'];d.qvel[:]=r['qvel'];mujoco.mj_forward(m,d);pos={k:d.xpos[b].copy() for k,b in bodies.items()};T=d.xmat[bodies['torso']].reshape(3,3);q=d.qpos[qa];margin=np.minimum(q-limits[:,0],limits[:,1]-q)
   rr.append(dict(time=r['time'],wrist=pos['wrist'].tolist(),shoulder=pos['shoulder'].tolist(),elbow=pos['elbow'].tolist(),root=r['root'],wrist_shoulder=(pos['wrist']-pos['shoulder']).tolist(),wrist_torso=(T.T@(pos['wrist']-pos['torso'])).tolist(),arm_q_deg=np.degrees(q[arm]).tolist(),arm_margin_deg=np.degrees(margin[arm]).tolist(),shoulder_wrist_distance=float(np.linalg.norm(pos['wrist']-pos['shoulder'])),arm_chain_length=float(np.linalg.norm(pos['elbow']-pos['shoulder'])+np.linalg.norm(pos['wrist']-pos['elbow'])),feet=r['feet_xyz'],support=r['foot_normal_force_n'],torso_yaw_deg=float(np.degrees(np.arctan2(T[1,0],T[0,0]))),force=r['force'],anchor=r['anchor'],anchor_velocity=r['anchor_velocity'],active=r['active']))
  r0=min(rr,key=lambda r:abs(r['time']-on));active=[r for r in p if r['active']];onrow=next((r for r in p if r['active']),p[0]);end=min(rr,key=lambda r:abs(r['time']-(on+s['ramp'])));root0=np.array(r0['root']);feet0=np.array(r0['feet']);w0=np.array(r0['wrist']);ws0=np.array(r0['wrist_shoulder'])
  events={}
  for name,pred in [('root_2cm',lambda r:np.linalg.norm(np.array(r['root'])[:2]-root0[:2])>.02),('root_5cm',lambda r:np.linalg.norm(np.array(r['root'])[:2]-root0[:2])>.05),('wrist_relative_shoulder_2cm',lambda r:np.linalg.norm(np.array(r['wrist_shoulder'])-ws0)>.02),('foot_moved_2cm',lambda r:np.max(np.linalg.norm(np.array(r['feet'])-feet0,axis=-1))>.02),('single_foot_below5N',lambda r:min(r['support'])<5)]:
   events[name]=next((r['time'] for i,r in enumerate(rr[:-5]) if r['time']>=on and all(pred(v) for v in rr[i:i+5])),None)
  snapshots=[]
  for dt in [0,.25,.5,1,2,3,4,s['ramp']]:
   r=min(rr,key=lambda r:abs(r['time']-on-dt));snapshots.append(dict(dt=dt,time=r['time'],wrist_delta_cm=(100*(np.array(r['wrist'])-w0)).tolist(),shoulder_delta_cm=(100*(np.array(r['shoulder'])-r0['shoulder'])).tolist(),relative_wrist_delta_cm=(100*(np.array(r['wrist_shoulder'])-ws0)).tolist(),root_delta_cm=(100*(np.array(r['root'])-root0)).tolist(),demand_cm=(100*(np.array(r['anchor'])-r0['anchor'])).tolist(),force=r['force'],arm_q_delta_deg=(np.array(r['arm_q_deg'])-r0['arm_q_deg']).tolist(),arm_margin_deg=r['arm_margin_deg']))
  row=dict(label=label,condition=s,onset_error_cm=(100*(np.array(onrow['anchor'])-onrow['wrist'])).tolist(),onset_force=onrow['force'],error_cm=100*np.sqrt(np.mean([r['error']**2 for r in active])) if active else None,force_rms=np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in active])) if active else 0,force_xyz_rms=np.sqrt(np.mean(np.array([r['force'] for r in active])**2,axis=0)).tolist() if active else [0,0,0],cap_fraction=np.mean([np.linalg.norm(r['force'])>=4.999 for r in active]) if active else 0,events=events,snapshots=snapshots,initial_arm=r0,root_at_motion_end_cm=(100*(np.array(end['root'])-root0)).tolist(),relative_wrist_end_cm=(100*(np.array(end['wrist_shoulder'])-ws0)).tolist(),root_recovery_delta_cm=(100*(np.array(rr[-1]['root'])-min(rr,key=lambda r:abs(r['time']-s['release']))['root'])).tolist(),max_tilt=max(r['tilt_deg'] for r in p),boundary=ep.get('boundary'),qpd_max_step_deg=float(np.degrees(max([r.get('q_PD_step_max_rad',0) for r in ep['control']]))),actual_q_max_step_deg=float(np.degrees(abs(np.diff(np.array([r['qpos'][7:] for r in p]),axis=0))).max()),joint_limit_max=max(r['joint_limit_excess_max_rad'] for r in p),torque_saturation_peak=max(r['torque_saturation'] for r in p))
  timelines[label]=rr;summaries.append(row);runner.write(out/'analysis.json',summaries)
  with gzip.open(out/'timelines.json.gz','wt') as f:json.dump(timelines,f)
  return row
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/t30-transfer-20261103/A-prepare-01/job')
for name in ['source_local','S0_local']:
 p=P/(name+'.json.gz');e=json.load(gzip.open(p,'rt'));analyze(e,'saved_'+name)
print([(r['label'],r['error_cm'],r['root_at_motion_end_cm']) for r in summaries])
