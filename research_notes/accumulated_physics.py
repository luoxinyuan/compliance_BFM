"""Single teacher-method intervention on the exact archived B goal path; no training."""
import copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from ccl02a_teacher import arm_ik,audit_episode
from wholebody_physics import save,read,REF,ZERO,INITIAL
from accumulated_precheck import build_and_check,SOURCE
from accumulated_controller import AccumulatedCorrection,executable_bounds

def run_study(env,model,contract,out):
 root=Path(__file__).resolve().parents[1];protocol=read(root/'research_notes/accumulated_protocol.json');save(out,'protocol.json',protocol)
 geometry=read(SOURCE/'geometry.json')
 cfg=env.cfg;m=env.sim.model;dev=env.standing.device;original_act=model.act
 names=contract['dof_names'];arm=contract['arm_indices'];waist=[i for i,n in enumerate(names) if n.startswith('waist')];assert waist==[12,13,14]
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids]
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
 def qtarget(a):return offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 zero=read(ZERO)['physics'];initial0=read(INITIAL);common=read(REF/'common-state.json')
 mode='baseline';record=False;control=[];command_z=env.standing;last=None;chosen_indices=arm
 goals,precheck=build_and_check(env,model,contract,common,out,protocol)
 settings=protocol['accumulated'];controller=None;old_b=read(SOURCE/'anchored.json');baseline=old_b;results=[]
 if not precheck['physical_allowed']:return dict(physical_executed=False,precheck=precheck,episodes=[])
 def act(obs,z,mean=True):
  nonlocal last
  nominal=original_act(obs,env.standing,mean=mean);actual_z=command_z
  if mode in ('accumulated',) and float(env.tether.time)>=8:
   f=max(0.,1.-(float(env.tether.time)-8)/.2);actual_z=model.project_z(env.standing+f*(goals[-1]-env.standing)) if f else env.standing
  base=original_act(obs,actual_z,mean=mean) if mode in ('accumulated',) else nominal
  applied=torch.zeros_like(base);correction=np.zeros(7);accumulation=None
  if mode=='accumulated':
   base_q=qtarget(base)[0].cpu().numpy();margin=settings['online_joint_margin_rad']
   lower,upper=executable_bounds(base_q[arm],limits[arm],scale[0,arm].cpu().numpy(),offset[0,arm].cpu().numpy(),contract['clip'],margin)
   correction,accumulation=controller.tick(float(env.tether.time),bool(env.tether.active),lower,upper)
   applied[:,arm]=torch.as_tensor(correction,device=dev,dtype=base.dtype)/scale[:,arm]
  action=base+applied if mode!='baseline' else base;q=qtarget(action);q0=qtarget(nominal)
  if record:
   dq=(q-q0)[0].cpu().numpy();ql=torch.tensor(limits,device=dev)
   row=dict(bfm_obs={k:v.tolist() for k,v in obs.items()},tether_time=float(env.tether.time),action=action[0].tolist(),nominal=nominal[0].tolist(),q_PD=q[0].tolist(),nominal_q_PD=q0[0].tolist(),delta_q_PD=dq.tolist(),executed_z=actual_z.tolist(),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),action_saturation=float(((action*5).abs()>=contract['clip']).float().mean()),q_PD_limit_excess_max_rad=float(((ql[:,0]-q).clamp_min(0)+(q-ql[:,1]).clamp_min(0)).max()),q_PD_step_max_rad=0. if last is None else float(abs(q[0].cpu().numpy()-last['q_PD']).max()),z_angle_deg=float(torch.rad2deg(torch.acos(((actual_z*env.standing).sum()/256).clamp(-1,1)))))
   row.update(goal_action=base[0].tolist(),goal_q_PD=qtarget(base)[0].tolist(),correction_q_PD=correction.tolist(),accumulation=accumulation)
   if not torch.isfinite(action).all() or row['action_saturation'] or row['q_PD_limit_excess_max_rad']>1e-6:
    raise ValueError('Nonfinite or clipped or out-of-range executed command')
   ci=len(control)
   assert row['executed_z']==old_b['control'][ci]['executed_z'],'Archived B latent path changed'
   assert abs(row['tether_time']-old_b['control'][ci]['tether_time'])<1e-10
   control.append(row);last=row
  return action
 model.act=act
 def snapshot():
  spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,env.sim.data,state,spec)
  tensors=lambda obj:{k:v.tolist() for k,v in vars(obj).items() if torch.is_tensor(v)}
  return dict(mj_integration=state.tolist(),base_tensors=tensors(env.base),sim_tensors=tensors(env.sim),wrapped_tensors=tensors(env.wrapped),bfm_obs={k:v.tolist() for k,v in env.obs.items()},previous_z=env.previous_z.tolist(),previous_action=env.previous_action.tolist(),force_history=env.history.tolist(),steps=env.steps,common_step_counter=env.base.common_step_counter,history={k:v.tolist() for k,v in env.base.history_handler.history.items()},env_bookkeeping=dict(done=bool(env.done),failed=bool(env.failed),last_qpos=env.last_qpos.tolist(),last_point=env.last_point.tolist(),last_support=env.last_support),tether={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(env.tether).items() if k in ['origin','delta','time','force','anchor','active','body','jac']})
 print('MODEL_READY_ACCUMULATED',flush=True)
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
  initial=copy.deepcopy(env.trace[-1]);env.tether.origin=env.last_point.copy();env.tether.anchor=env.last_point.copy();env.tether.delta=np.array([geometry['delta_x_m'],0,0]);cfg.enabled=True
  mode=label;record=True;control=[];last=None;chosen_indices=arm;start=len(env.trace);decisions=[];boundary=None
  controller=AccumulatedCorrection(settings['per_joint_caps_rad'],settings['norm_cap_rad'],settings['per_joint_velocity_rad_s'],settings['norm_velocity_rad_s'],settings['release_duration_s'])
  for high in range(60):
   assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
   now=float(env.tether.time);active=bool(env.tether.active);command_z=goals[min(40,max(0,round((now-3)*10)))] if label in ('accumulated',) and now<8 else env.standing
   if label in ('accumulated',) and now>=8:command_z=model.project_z(env.standing+max(0.,1-(now-8)/.2)*(goals[-1]-env.standing))
   nominal=original_act(env.obs,command_z,mean=True);desired=np.zeros(len(chosen_indices))
   if label!='baseline' and active:
    point,_=env.tether.point_state();error=env.tether.anchor-point;desired=arm_ik(env.tether.jac[:,vadr[chosen_indices]],error)
    q0=qtarget(nominal).cpu().numpy().reshape(-1);desired=np.clip(q0[chosen_indices]+desired,limits[chosen_indices,0],limits[chosen_indices,1])-q0[chosen_indices]
   q0=qtarget(nominal).cpu().numpy().reshape(-1);margin=settings['online_joint_margin_rad']
   lower,upper=executable_bounds(q0[arm],limits[arm],scale[0,arm].cpu().numpy(),offset[0,arm].cpu().numpy(),contract['clip'],margin)
   controller.propose(desired,active,lower,upper)
   decisions.append(dict(time=now,active=active,correction=controller.applied.tolist(),increment=desired.tolist(),proposal=copy.deepcopy(controller.last_proposal),z=command_z.tolist()))
   try:_,_,done,info=env.step(torch.zeros_like(env.standing),baseline=True)
   except ValueError as exc:
    boundary=dict(time=float(env.tether.time),kind='command_envelope',reason=str(exc));break
   rows=env.trace[-20:]
   for r in rows:
    if r['failed'] or r['height']<.65 or r['tilt_deg']>20 or r['undesired_ground_contacts'] or r['joint_limit_excess_max_rad']>.02:
     boundary=dict(time=r['time'],failed=r['failed'],height=r['height'],tilt_deg=r['tilt_deg'],undesired_ground_contacts=r['undesired_ground_contacts'],joint_limit_excess_max_rad=r['joint_limit_excess_max_rad']);break
   if done or boundary:break
  record=False;ep=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=control,decisions=decisions,failed=bool(env.failed),stop_boundary=boundary,same_initial_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
  save(out,label+'.json',ep)
  assert [r['anchor'] for r in ep['physics']]==[r['anchor'] for r in baseline['physics'][:len(ep['physics'])]]
  audit=audit_episode(ep,contract,state);save(out,label+'-pd-audit.json',audit)
  results.append(dict(label=label,physics_samples=len(ep['physics']),complete=len(ep['physics'])==1200 and not ep['failed'] and boundary is None,audit=audit,boundary=boundary));save(out,'physical-result.json',results);print('BRANCH_COMPLETE',json.dumps(results[-1]),flush=True)
 model.act=original_act
 return dict(physical_executed=True,precheck=precheck,episodes=results,policy_updates=0,branch_sim_seconds=sum(r['physics_samples'] for r in results)/200,prefix_sim_seconds=3.)
