"""M6 causal task expert and deployable LPP rollouts; original BFM and bounded correction."""
import copy,gzip,hashlib,json,math,os
from pathlib import Path
import numpy as np
import torch,mujoco
from locomotion_rollout import UnsafePhysics,snapshot,spherical
from contact_projector import ContactGate,check_deadline
from contact_gate import smooth,ContactGate as NativeGate
from alpha_student import student_input
from alpha_memory import ForceMemory
from ccl02a_teacher import arm_ik,ReleaseFade
from frozen_bfm_placement import tensor_digest
from m1_stage import cached_point,physical_state_digest,save
from m1_learning import completed,load_net
from m1_interface import endpoints,policy_latent,base_latent
from m1_geometry import geometry
from m2_rule import infer_coordinate
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs'
RUN=R/'m6-multibehavior-20261031A'
M3=R/'m3-continuous-body-20261031A'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_gz(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def handoff_progress(t,ready):return smooth((t-ready)/.75)

def run_study(env,model,contract,out):
 protocol=read(Path(__file__).with_name('m6_protocol.json'));save(out,'protocol.json',protocol);stage=os.environ['M6_STAGE']
 if stage.startswith('train'):
  from m6_learning import train
  return train(env,model,contract,out,int(stage[-1]))
 dev=env.standing.device;m=env.sim.model;orig=model.act;ends=endpoints(env);ready=3.
 from m6_learning import load_net,done,policy_latent
 nets={}
 if stage not in ['qualification','projected']:
  rnd=2 if stage=='final' else int(stage[-1])-1
  nets={k:load_net(done('train'+str(rnd))/(k+'-final.pt'),dev) for k in ['alignment','distill','single_negative']}
 if stage=='final':
  assert {k:sha(done('train2')/(k+'-final.pt')) for k in nets}==protocol['final_checkpoint_sha256']
 anchor_reference=read_gz(R/'m5-contact-lpp-20261031A/qualification-attempt-01/outputs/free__teacher.json.gz')['physics']
 if stage=='final':nets['m5']=load_net(R/'m5-contact-lpp-20261031A/train2-attempt-01/outputs/distill-final.pt',dev)
 identities={k:tensor_digest(v) for k,v in nets.items()};pose=env.standing;goal=ends[2]
 save(out,'references.json',dict(endpoints=[z.tolist() for z in ends],teacher='causal native force gate + original bounded arm feedback',input='force/proprio/action history only',checkpoint_hashes=identities))
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qtarget=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids];arm=np.arange(22,29);sc=np.asarray(contract['q_PD_scale'])[0]
 zero=read(R/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics'];initial0=read(R/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json');common=read(R/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
 jobs=protocol['schedule'][stage]
 replay=None
 if stage=='final':
  from m6_rule import calibration,command as force_rule_command
  rule_template,rule_force,rule_info=calibration(dev);save(out,'force-rule-calibration.json',rule_info)

 native_cfg=read(Path(__file__).with_name('contact_gate_protocol.json'))['gate']
 mode='prefix';controller='native_hold';enabled=False;cs=[];tick=0;last_q=None;last_z=None;anchor=None;contact_initial=None;contact_state=None;ready_state=None;ready_initial=None;geo=None
 memory=ForceMemory(320);gate=ContactGate();native=NativeGate(native_cfg);rule_context=None;fade=ReleaseFade();corr=np.zeros(7)
 def align_anchor(t):
  nonlocal anchor,contact_initial,contact_state,geo
  if anchor is not None or t<onset-1e-8:return
  point,_=env.tether.point_state();anchor=point.copy()
  if anchor_reference is not None:
   from m6_reference import wrist_at_time
   anchor=np.asarray(wrist_at_time(anchor_reference,onset)).copy()
  env.tether.origin=anchor.copy();env.tether.anchor=anchor.copy();env.tether.delta=delta.copy()
  contact_initial=copy.deepcopy(env.trace[-1]);contact_initial['wrist']=point.tolist();contact_state=snapshot(env);geo=geometry(m,env.sim.data.qpos.copy(),delta[0])
 def act(obs,z,mean=True):
  nonlocal tick,last_q,last_z,ready_state,ready_initial,rule_context,corr
  t=float(env.tether.time);force=env.tether.force.copy();nominal=orig(obs,env.standing,mean=True)
  if mode=='prefix':return nominal
  if tick%5==0:gate.update(t,force)
  lg=gate.value(t);deployable_h=student_input(model,obs,[memory.append(force.tolist())],env.keys)
  student_key=controller.replace('_no_residual','').replace('_no_force','');student_coords=None;rule_record=native.step(force.tolist());raw=None;projection_record=None
  oracle_command=spherical(pose,goal,rule_record['alpha'])
  if ready_state is None:
   ready_state=snapshot(env);ready_initial=copy.deepcopy(env.trace[-1]);ready_initial['wrist']=cached_point(env).tolist()
  if student_key in nets:
   net_h=deployable_h.clone()
   if controller.endswith('_no_force'):net_h[:,465:]=0.
   student_raw=nets[student_key](net_h);
   if controller.endswith('_no_residual'):student_raw=student_raw.clone();student_raw[:,1:]=0.
   raw=student_raw[0].tolist();student_coords=[float(student_raw[0,0].clamp(0,1))];command=policy_latent(model,ends,student_raw,lg)
  elif controller=='force_rule':command=force_rule_command(model,oracle_command,deployable_h,lg,rule_template,rule_force)
  elif controller in ['teacher','rule','projected']:command=oracle_command
  else:assert controller=='native_hold';command=pose
  base=orig(obs,command,mean=True);base_q=qtarget(base);action=base;oracle={}
  # Original queryable feedback, privileged labels only. No extra actuator for a Student.
  if controller in ['teacher','projected'] or (student_key in nets and stage!='final'):
   oracle_base=base if controller in ['teacher','projected'] else orig(obs,oracle_command,mean=True);oq=qtarget(oracle_base)[0].cpu().numpy();point=cached_point(env)
   if tick%5==0:
    desired=np.zeros(7)
    if env.tether.active:
     jac=np.zeros((3,m.nv));mujoco.mj_jac(m,env.sim.data,jac,None,point,env.tether.body);desired=arm_ik(jac[:,vadr[arm]],env.tether.anchor-point);desired=np.clip(oq[arm]+desired,limits[arm,0],limits[arm,1])-oq[arm]
    corr=fade.update(t,bool(env.tether.active),desired)
   if fade.release_start is not None:corr=fade.value(t)
   oracle_valid=bool(abs(corr).max()<=.040001 and np.linalg.norm(corr)<=.080001)
   if controller in ['teacher','projected'] and not oracle_valid:raise ValueError('Teacher bound conflicts with native target; do not expand bounds')
   da=torch.zeros_like(oracle_base);da[0,22:]=torch.tensor(corr/sc[arm],device=dev,dtype=base.dtype);oa=oracle_base+da
   oracle=dict(oracle_query_valid=oracle_valid,oracle_z=oracle_command.tolist(),oracle_base_action=oracle_base[0].tolist(),oracle_base_q_PD=oq.tolist(),oracle_action=oa[0].tolist(),oracle_q_PD=qtarget(oa)[0].tolist(),oracle_correction=corr.tolist())
   if controller=='teacher':action=oa;command=oracle_command
   elif controller=='projected':
    from m3_learning import project
    if lg>0 and abs(corr).max()>1e-10:
     command,pe=project(model,obs,oracle_command,lg,qtarget(oa),qtarget)
    else:command=oracle_command;pe=qtarget(orig(obs,command,mean=True))-qtarget(oa)
    action=orig(obs,command,mean=True);projection_record=dict(arm_rms_deg=float(torch.rad2deg(pe[:,22:].square().mean().sqrt())),nonarm_rms_deg=float(torch.rad2deg(pe[:,:22].square().mean().sqrt())))
  q=qtarget(action);q0=qtarget(nominal)
  assert torch.isfinite(action).all() and abs(float(command.norm())-16)<1e-4
  cs.append(dict(**oracle,projection=projection_record,time=t,tether_time=t,alpha=rule_record['alpha'],gate=rule_record,local_gate=lg,native_coords=student_coords,student_raw=raw,student_input=deployable_h[0].tolist(),rule_context=rule_context,rule_record=rule_record,executed_z=command.tolist(),measured_force_xyz=force.tolist(),bfm_obs={k:v.tolist() for k,v in obs.items()},base_action=base[0].tolist(),base_q_PD=base_q[0].tolist(),teacher_correction=corr.tolist() if controller=='teacher' else [0.]*7,pre_anchor=env.tether.anchor.tolist(),pre_wrist=cached_point(env).tolist(),active=bool(env.tether.active),action=action[0].tolist(),standing_action=nominal[0].tolist(),nominal=nominal[0].tolist(),q_PD=q[0].tolist(),standing_q_PD=q0[0].tolist(),nominal_q_PD=q0[0].tolist(),delta_q_PD=(q-q0)[0].tolist(),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),action_saturation_fraction=float(((action*5).abs()>=contract['clip']).float().mean()),q_PD_step_max_rad=float(abs(q-last_q).max()),z_step_deg=float(torch.rad2deg(torch.acos(((command*last_z).sum()/(command.norm()*last_z.norm())).clamp(-1,1))))))
  last_q=q.clone();last_z=command.clone();tick+=1;return action
 model.act=act;results=dict(status='running',episodes=[],student_optimizer_updates=0);paired={}
 try:
  for condition_name,controller in jobs:
   check_deadline();spec=protocol['conditions'][condition_name];onset=spec['onset'];release=spec['release'];end=spec['end'];delta=np.asarray(spec['delta_xyz']);enabled=spec['enabled'];configuration='standing';label=condition_name+'__'+controller;mode='prefix';env.qualification_active=False;env.interruption_window=None
   for cfg in (env.cfg,env.tether.cfg):cfg.enabled=False;cfg.activation_time=onset;cfg.ramp_duration=spec['ramp'];cfg.release_time=release
   env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
   for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
    v=getattr(env.base,name);v.copy_(torch.as_tensor(initial0['base_tensors'][name],device=v.device,dtype=v.dtype))
   env.base.common_step_counter=initial0['common_step_counter'];mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,env.sim.data);mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
   for _ in range(30):_,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
   parity={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']};assert max(parity.values())==0;save(out,label+'-prefix.json',parity)
   env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype));assert snapshot(env)==common
   cs=[];tick=0;gate=ContactGate();memory=ForceMemory(320);native=NativeGate(native_cfg);rule_context=None;fade=ReleaseFade();corr=np.zeros(7);last_q=qtarget(env.previous_action).clone();last_z=env.standing.clone()
   preparation_initial=copy.deepcopy(env.trace[-1]);start=len(env.trace);anchor=None;contact_initial=None;contact_state=None;geo=None;boundary=None;ready_state=None;ready_initial=None
   for cfg in (env.cfg,env.tether.cfg):cfg.enabled=enabled;cfg.activation_time=onset;cfg.ramp_duration=spec['ramp'];cfg.release_time=release
   env.tether.delta=np.zeros(3);env.m1_anchor_hook=align_anchor;mode='record';env.qualification_active=True;steps=math.ceil((end-3)*50-1e-8)
   for i in range(steps):
    check_deadline()
    try:
     a=model.act(env.obs,env.standing,mean=True);env.obs,_,terminated,truncated,_=env.wrapped.step(a,to_numpy=False);env.steps+=1;env.previous_action=a.clone();env.history=np.roll(env.history,-1,axis=0);env.history[-1]=env.tether.force
     if env.failed or bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any()):boundary=dict(kind='native_failure',time=float(env.tether.time));break
    except UnsafePhysics as exc:boundary=dict(kind='unsafe_physics',physics=exc.row);break
    except ValueError as exc:
     if str(exc)!='Teacher bound conflicts with native target; do not expand bounds':raise
     boundary=dict(kind='teacher_query_bound',time=float(env.tether.time),message=str(exc));break
   mode='prefix';env.qualification_active=False
   condition=dict(split=stage,name=condition_name,onset=onset,ramp=spec['ramp'],duration=release-onset,release=release,end=end,enabled=enabled,delta_xyz=delta.tolist())
   ep=dict(requested_end_s=end,actual_end_s=float(env.tether.time),control_ready_s=ready,task_onset_handoff=False,ready_initial=ready_initial,ready_physical_state_sha256=physical_state_digest(ready_state) if ready_state else None,label=label,controller=controller,configuration=configuration,condition=condition,initial=contact_initial or preparation_initial,preparation_initial=preparation_initial,physics=copy.deepcopy(env.trace[start:]),control=cs,decisions=[],teacher_decisions=[],boundary=boundary,geometry=geo,goal_candidate=dict(source='Original standing-nativegoal endpoint, force-causal progress; no goal search',u=0,z=goal.tolist()),physical_state_sha256=physical_state_digest(contact_state) if contact_state else None,contact_state_sha256=hashlib.sha256(json.dumps(contact_state,sort_keys=True).encode()).hexdigest() if contact_state else None)
   save(out,label+'.json',ep);save(out,label+'-contact-state.json',contact_state)
   entry=dict(label=label,seconds=len(ep['physics'])/200,complete=len(ep['physics'])==steps*4 and boundary is None,boundary=boundary);results['episodes'].append(entry);save(out,'physical-result.json',results);print('M6_PHYSICS',entry,flush=True)
   if not entry['complete']:results['status']='capability-stopped' if boundary and boundary['kind']=='teacher_query_bound' else 'physical-stopped';return results
   if configuration not in paired:paired[configuration]=ep['ready_physical_state_sha256']
   else:assert ep['ready_physical_state_sha256']==paired[configuration]
  results['status']='completed';assert {k:tensor_digest(v) for k,v in nets.items()}==identities;save(out,'physical-result.json',results)
  return results
 except Exception as exc:
  save(out,'failure-partial.json',dict(label=locals().get('label'),time=float(env.tether.time),exception_type=type(exc).__name__,message=str(exc),physical_trace=env.trace,control=cs));raise
 finally:model.act=orig;env.qualification_active=False;env.m1_anchor_hook=None
