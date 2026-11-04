"""Frozen reach-to-walk audit; IK is offline goal construction, never joint control."""
import json,time,copy,gzip
from pathlib import Path
import transfer_student_validated as runner

def study(env,model,contract,out):
 import numpy as np,torch,mujoco
 from scipy.optimize import least_squares
 from clean_policy import Expert
 from geometry_policy import expand
 from transition_env import Task,PLAN,install_frame
 from locomotion_rollout import spherical
 from wholebody_physics import goal_observation
 from m11_runtime_speed import install
 install_frame()
 import geometry_env,reach_contact
 geometry_env.install_contact=reach_contact.install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 cfg=json.loads(Path(__file__).with_name('reach_protocol.json').read_text());task=Task(env,model,contract,0);nets={}
 for k,v in cfg['models'].items():
  assert runner.sha(v['path'])==v['sha256'];ck=torch.load(v['path'],map_location=env.standing.device,weights_only=True);net=Expert(1088,'direct').to(env.standing.device);expand(net,ck);nets[k]=net.eval().requires_grad_(False)
 runner.write(out/'identities.json',cfg['models']);m=env.sim.model;d=mujoco.MjData(m);names=contract['dof_names'];jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];qa=m.jnt_qposadr[jids];limits=m.jnt_range[jids];arm=[i for i,n in enumerate(names) if n.startswith('right_') and any(x in n for x in ['shoulder','elbow','wrist'])]
 bodies={k:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for k,n in dict(wrist='right_wrist_yaw_link',shoulder='right_shoulder_pitch_link',elbow='right_elbow_link',torso='torso_link').items()};assert min(bodies.values())>0
 runner.write(out/'joint-contract.json',dict(names=names,arm_indices=arm,qpos_addresses=qa.tolist(),limits=limits.tolist(),bodies=bodies))
 summaries=[];timelines={}
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
 old=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m41-startup-causal-20261104/physical-01')
 for key in ['I','T30','A32','B32']:
  ep=json.load(gzip.open(old/(key+'.json.gz'),'rt'));analyze(ep,'saved_'+key)
 # Bounded offline arm-only IK: actual joint ranges, no locking in physics or controller.
 task.reset(copy.deepcopy(PLAN['dev']['stand_walk']),record=True,label='goal_construction');q0=env.sim.data.qpos.copy();d.qpos[:]=q0;mujoco.mj_forward(m,d);w=d.xpos[bodies['wrist']].copy();start=q0[qa[arm]].copy();goals={};ik=[]
 z0=model.goal_inference(goal_observation(env,q0,np.zeros(m.nv)))
 for dist in [.05,.15,.30,.80]:
  target=w+np.array([dist,0,0])
  def objective(x):
   d.qpos[:]=q0;d.qpos[qa[arm]]=x;mujoco.mj_forward(m,d);return np.r_[d.xpos[bodies['wrist']]-target,.002*(x-start)]
  fit=least_squares(objective,start,bounds=(limits[arm,0]+1e-6,limits[arm,1]-1e-6),max_nfev=100);objective(fit.x);pose=d.qpos.copy();zg=model.goal_inference(goal_observation(env,pose,np.zeros(m.nv)));za=model.project_z(env.standing+zg-z0)
  ik.append(dict(distance=dist,error_cm=float(100*np.linalg.norm(d.xpos[bodies['wrist']]-target)),target=target.tolist(),wrist=d.xpos[bodies['wrist']].tolist(),qpos=pose.tolist(),arm_change_deg=np.degrees(fit.x-start).tolist(),native_z=zg.tolist(),anchored_z=za.tolist(),scope='offline geometry only; no live qpos or action override'))
  if dist in [.05,.15]:goals[dist]=(zg,za)
 runner.write(out/'offline-ik.json',ik)
 def run(label,who,dist,ramp,anchor_mode='onset_actual',forceoff=False,mask=False,goal=None):
  if time.monotonic()-runner.START>runner.ARGS.max_seconds-60:raise RuntimeError('bounded runtime')
  spec=copy.deepcopy(PLAN['dev']['stand_walk']);spec.update(distance=dist,ramp=ramp,release=4.25+ramp+2,end=4.25+ramp+6,anchor_mode=anchor_mode,physical_force_off=forceoff)
  task.reset(spec,record=True,label=label)
  while not task.done:
   x=task.features()
   if mask and env.tether.active:
    # Mask exact privileged goal-error/relative-velocity channels, NOT force or proprio.
    x=x.clone();x[:,465:471]=0;x[:,1068:1086]=0
   if goal is not None:
    t=float(env.tether.time);a,_=reach_contact.smooth(t,4.25,ramp)
    if t>=spec['release']:f,_=reach_contact.smooth(t,spec['release'],1.5);a*=1-f
    raw=spherical(env.standing,goal,a)/16
   else:raw=env.standing/16 if who=='Standing' else nets[who].distribution(x).mean
   task.step(raw)
  result=task.save(out);ep=json.load(gzip.open(out/(label+'.json.gz'),'rt'));row=analyze(ep,label);row['alignment']=getattr(task,'alignment',None);runner.write(out/'analysis.json',summaries)
  print('REACH',label,round(row['error_cm'],3),round(row['force_rms'],3),row['root_at_motion_end_cm'],result['boundary'],flush=True)
  return result
 # Corrected sensitivity: mask only privileged geometric error and relative velocity
 # after onset; preserve existing constant heading preference and identical prefix.
 for who in ['I','T30']:run(who+'_mask_error_velocity_80',who,.80,7.8,mask=True)
 for who in ['I','T30']:run(who+'_goalcontrol_05',who,.05,3.)
 for dist in [.05,.15]:
  for k,name in enumerate(['native','anchored']):run(f'goal_free_{name}_{round(dist*100):02d}','Standing',dist,3.,forceoff=True,goal=goals[dist][k])
 return dict(status='completed',episodes=8,rl_training_decisions=0,supervised_updates=0,elapsed_s=time.monotonic()-runner.START)

runner.study=study
if __name__=='__main__':runner.main()
