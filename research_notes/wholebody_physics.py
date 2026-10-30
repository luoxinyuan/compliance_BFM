"""Finite four-way reach feasibility, original BFM/PD; private geometric goal queries only."""
import copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.optimize import least_squares
from ccl02a_teacher import arm_ik,ReleaseFade,audit_episode
from wholebody_geometry import analyze
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUNS=P/'bridge-v2-owned/runs'
REF=RUNS/'issue21-direction-20261030T020000Z/attempt-02/outputs'
ZERO=RUNS/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json'
INITIAL=RUNS/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json'
def save(out,name,x):(out/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def read(p):return json.loads(p.read_text())
def smooth(u):
 u=float(np.clip(u,0,1));return u*u*(3-2*u)

def goal_observation(env,qpos,qvel):
 """Original backward-map inputs from private MjData; no stepping or live mutation."""
 from humanoidverse.utils.helpers import get_reference_proprioception
 from humanoidverse.utils.torch_utils import quat_mul,my_quat_rotate
 from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max,compute_humanoid_observations_max_with_contact
 m=env.sim.model;d=mujoco.MjData(m);d.qpos[:]=qpos;d.qvel[:]=qvel;mujoco.mj_forward(m,d)
 dev=env.standing.device;t=lambda a:torch.as_tensor(np.array(a),device=dev,dtype=torch.float32).unsqueeze(0)
 ids=env.sim.body_id;pos=t(d.xpos[ids]);rot=t(d.xquat[ids])[:,:,[1,2,3,0]];vel=np.empty((len(ids),6))
 for i,b in enumerate(ids):mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,int(b),vel[i],0)
 v=t(vel[:,3:]);av=t(d.cvel[ids,:3]);base=env.base
 if base.num_extend_bodies:
  parent=base.extend_body_parent_ids;pr=rot[:,parent];offset=my_quat_rotate(pr.reshape(-1,4),base.extend_body_pos_in_parent.reshape(-1,3)).reshape(1,-1,3)
  pos=torch.cat([pos,pos[:,parent]+offset],1);rot=torch.cat([rot,quat_mul(pr,base.extend_body_rot_in_parent_xyzw,w_last=True)],1)
  v=torch.cat([v,v[:,parent]+torch.cross(av[:,parent],offset,dim=-1)],1);av=torch.cat([av,av[:,parent]],1)
 args=[pos,rot,v,av,True,bool(base.config.obs.get('root_height_obs',True))]
 if base.use_contact_in_obs_max:obs=compute_humanoid_observations_max_with_contact(*args,contact_binary=base.foot_contact_detect(pos,v))
 else:obs=compute_humanoid_observations_max(*args)
 # Native MuJoCo free-joint angular velocity is body-local; helper expects world angular velocity.
 world_ang=d.xmat[1].reshape(3,3)@d.qvel[3:6]
 proprio=get_reference_proprioception(base,t(d.qpos[7:]),t(d.qvel[6:]),rot[:,0],t(world_ang))
 return {'state':torch.cat(list(proprio.values()),-1),'privileged_state':torch.cat(list(obs.values()),-1)}

def build_goals(env,model,geometry,out):
 m=env.sim.model;d=mujoco.MjData(m);q=np.asarray(geometry['qpos']);body=env.tether.body
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in env.sim.dof_names]
 indices=[12,13,14]+list(range(22,29));adr=m.jnt_qposadr[np.array(ids)[indices]];limits=m.jnt_range[np.array(ids)[indices]];start=q[adr].copy();x=start.copy()
 zero_vel=np.zeros(m.nv);z0=model.goal_inference(goal_observation(env,q,zero_vel));goals=[];rows=[]
 for i in range(41):
  target=np.asarray(geometry['wrist'])+np.array([geometry['delta_x_m']*smooth(i/40),0,0])
  def objective(a):
   d.qpos[:]=q;d.qpos[adr]=a;mujoco.mj_forward(m,d)
   return np.r_[d.xpos[body]-target,.005*(a-start)]
  if i:fit=least_squares(objective,x,bounds=(limits[:,0],limits[:,1]),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9);x=fit.x
  objective(x);pose=d.qpos.copy();goal=model.goal_inference(goal_observation(env,pose,zero_vel))
  # Anchoring cancels the static-pose vs historical standing-latent offset at onset.
  z=model.project_z(env.standing+goal-z0);goals.append(z)
  rows.append(dict(t_from_onset=i/10,target=target.tolist(),qpos=pose.tolist(),wrist=d.xpos[body].tolist(),shoulder=d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_shoulder_pitch_link')].tolist(),com=d.subtree_com[1].tolist(),ik_error_m=float(np.linalg.norm(d.xpos[body]-target)),z=z.tolist(),z_angle_deg=float(torch.rad2deg(torch.acos(((z*env.standing).sum()/256).clamp(-1,1))))))
 save(out,'goal-path.json',dict(method='native goal_inference, project_z(standing+goal(pose)-goal(initial)), no weight updates',waypoints=rows))
 return goals

def run_study(env,model,contract,out):
 root=Path(__file__).resolve().parents[1];protocol=read(root/'research_notes/wholebody_protocol.json');save(out,'protocol.json',protocol)
 geometry=analyze(root);save(out,'geometry.json',geometry)
 cfg=env.cfg;m=env.sim.model;dev=env.standing.device;original_act=model.act
 names=contract['dof_names'];arm=contract['arm_indices'];waist=[i for i,n in enumerate(names) if n.startswith('waist')];assert waist==[12,13,14]
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids]
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
 def qtarget(a):return offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 zero=read(ZERO)['physics'];initial0=read(INITIAL);common=read(REF/'common-state.json')
 mode='baseline';record=False;control=[];delta=torch.zeros(1,29,device=dev);fade=ReleaseFade();command_z=env.standing;last=None;chosen_indices=arm
 goals=None;baseline=None;results=[]
 def act(obs,z,mean=True):
  nonlocal last
  nominal=original_act(obs,env.standing,mean=mean);actual_z=command_z
  if mode=='goal_bfm' and float(env.tether.time)>=8:
   f=max(0.,1.-(float(env.tether.time)-8)/.2);actual_z=model.project_z(env.standing+f*(goals[-1]-env.standing)) if f else env.standing
  base=original_act(obs,actual_z,mean=mean) if mode=='goal_bfm' else nominal
  applied=delta
  if mode!='baseline' and fade.release_start is not None:
   applied=torch.zeros_like(base);applied[:,chosen_indices]=torch.as_tensor(fade.value(float(env.tether.time)),device=dev,dtype=base.dtype)/scale[:,chosen_indices]
  action=base+applied if mode!='baseline' else base;q=qtarget(action);q0=qtarget(nominal)
  if record:
   dq=(q-q0)[0].cpu().numpy();ql=torch.tensor(limits,device=dev)
   row=dict(tether_time=float(env.tether.time),action=action[0].tolist(),nominal=nominal[0].tolist(),q_PD=q[0].tolist(),nominal_q_PD=q0[0].tolist(),delta_q_PD=dq.tolist(),executed_z=actual_z.tolist(),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),action_saturation=float(((action*5).abs()>=contract['clip']).float().mean()),q_PD_limit_excess_max_rad=float(((ql[:,0]-q).clamp_min(0)+(q-ql[:,1]).clamp_min(0)).max()),q_PD_step_max_rad=0. if last is None else float(abs(q[0].cpu().numpy()-last['q_PD']).max()),z_angle_deg=float(torch.rad2deg(torch.acos(((actual_z*env.standing).sum()/256).clamp(-1,1)))))
   control.append(row);last=row
  return action
 model.act=act
 def snapshot():
  spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,env.sim.data,state,spec)
  tensors=lambda obj:{k:v.tolist() for k,v in vars(obj).items() if torch.is_tensor(v)}
  return dict(mj_integration=state.tolist(),base_tensors=tensors(env.base),sim_tensors=tensors(env.sim),wrapped_tensors=tensors(env.wrapped),bfm_obs={k:v.tolist() for k,v in env.obs.items()},previous_z=env.previous_z.tolist(),previous_action=env.previous_action.tolist(),force_history=env.history.tolist(),steps=env.steps,common_step_counter=env.base.common_step_counter,history={k:v.tolist() for k,v in env.base.history_handler.history.items()},env_bookkeeping=dict(done=bool(env.done),failed=bool(env.failed),last_qpos=env.last_qpos.tolist(),last_point=env.last_point.tolist(),last_support=env.last_support),tether={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(env.tether).items() if k in ['origin','delta','time','force','anchor','active','body','jac']})
 print('MODEL_READY_WHOLEBODY',flush=True)
 for label in protocol['labels']:
  record=False;mode='baseline';command_z=env.standing;cfg.enabled=False;cfg.activation_time=3.;cfg.ramp_duration=4.;cfg.release_time=8.
  env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
  for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
   v=getattr(env.base,name);v.copy_(torch.as_tensor(initial0['base_tensors'][name],device=v.device,dtype=v.dtype))
  env.base.common_step_counter=initial0['common_step_counter'];mujoco.mj_setState(m,env.sim.data,np.array(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,env.sim.data);mujoco.mj_setState(m,env.sim.data,np.array(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
  for _ in range(30):
   _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
  diff={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']};save(out,label+'-prefix-parity.json',diff);assert max(diff.values())==0
  env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype));state=snapshot();assert state==common,'Full prefix state mismatch'
  save(out,label+'-initial-state.json',state)
  assert np.max(abs(env.last_point-np.asarray(geometry['wrist'])))==0
  if goals is None:
   reconstructed=goal_observation(env,env.sim.data.qpos,env.sim.data.qvel)
   parity={k:float((v-env.obs[k]).abs().max()) for k,v in reconstructed.items()};save(out,'backward-input-parity.json',parity)
   assert max(parity.values())<1e-4,'Private geometric backward observation does not match native interface'
   goals=build_goals(env,model,geometry,out)
  initial=copy.deepcopy(env.trace[-1]);env.tether.origin=env.last_point.copy();env.tether.anchor=env.last_point.copy();env.tether.delta=np.array([geometry['delta_x_m'],0,0]);cfg.enabled=True
  mode=label;record=True;control=[];last=None;delta=torch.zeros(1,29,device=dev);fade=ReleaseFade();chosen_indices=waist+arm if label=='waist_dls' else arm;fade.last=np.zeros(len(chosen_indices));start=len(env.trace);decisions=[];boundary=None
  for high in range(60):
   assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
   now=float(env.tether.time);active=bool(env.tether.active);command_z=goals[min(40,max(0,round((now-3)*10)))] if label=='goal_bfm' and now<8 else env.standing
   if label=='goal_bfm' and now>=8:command_z=model.project_z(env.standing+max(0.,1-(now-8)/.2)*(goals[-1]-env.standing))
   nominal=original_act(env.obs,command_z,mean=True);desired=np.zeros(len(chosen_indices))
   if label!='baseline' and active:
    point,_=env.tether.point_state();error=env.tether.anchor-point;desired=arm_ik(env.tether.jac[:,vadr[chosen_indices]],error)
    q0=qtarget(nominal).cpu().numpy().reshape(-1);desired=np.clip(q0[chosen_indices]+desired,limits[chosen_indices,0],limits[chosen_indices,1])-q0[chosen_indices]
   correction=fade.update(now,active,desired);delta=torch.zeros_like(nominal)
   if label!='baseline':delta[:,chosen_indices]=torch.tensor(correction,device=dev,dtype=nominal.dtype)/scale[:,chosen_indices]
   decisions.append(dict(time=now,active=active,correction=correction.tolist(),z=command_z.tolist()))
   _,_,done,info=env.step(torch.zeros_like(env.standing),baseline=True)
   rows=env.trace[-20:]
   for r in rows:
    if r['failed'] or r['height']<.65 or r['tilt_deg']>20 or r['undesired_ground_contacts'] or r['joint_limit_excess_max_rad']>.02:
     boundary=dict(time=r['time'],failed=r['failed'],height=r['height'],tilt_deg=r['tilt_deg'],undesired_ground_contacts=r['undesired_ground_contacts'],joint_limit_excess_max_rad=r['joint_limit_excess_max_rad']);break
   if done or boundary:break
  record=False;ep=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=control,decisions=decisions,failed=bool(env.failed),stop_boundary=boundary,same_initial_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
  save(out,label+'.json',ep)
  if label=='baseline':baseline=ep
  assert [r['anchor'] for r in ep['physics']]==[r['anchor'] for r in baseline['physics'][:len(ep['physics'])]]
  audit=audit_episode(ep,contract,state);save(out,label+'-pd-audit.json',audit)
  results.append(dict(label=label,physics_samples=len(ep['physics']),complete=len(ep['physics'])==1200 and not ep['failed'] and boundary is None,audit=audit,boundary=boundary));save(out,'physical-result.json',results);print('BRANCH_COMPLETE',json.dumps(results[-1]),flush=True)
 model.act=original_act
 return dict(episodes=results,policy_updates=0,branch_sim_seconds=sum(r['physics_samples'] for r in results)/200,prefix_sim_seconds=12.)
