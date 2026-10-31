"""M1 natural configuration qualification. No Student or BFM training in this stage."""
import copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch,mujoco
from scipy.optimize import least_squares
from locomotion_rollout import UnsafePhysics,snapshot,spherical
from contact_gate import smooth
from contact_projector import ContactGate,deploy_input,constrained,check_deadline
from m0_lpp_stage import load_net
from alpha_student import load_student,student_input
from alpha_memory import ForceMemory
from frozen_bfm_placement import tensor_digest
from wholebody_physics import goal_observation
from m1_geometry import geometry
from ccl02a_teacher import arm_ik,ReleaseFade
from accumulated_controller import AccumulatedCorrection,executable_bounds
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs';RUN=R/'m1-state-response-20261031A'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(out,name,x):(out/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def physical_state_digest(state):
 # Reward accumulator is not consumed by BFM, PD, physics or Student. Preserve full raw snapshot separately.
 state=copy.deepcopy(state);state['base_tensors'].pop('rew_buf')
 return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
def cached_point(env):
 # Original wrapper has already forwarded the physical state. A measurement must not re-solve warmstarts.
 d=env.sim.data;return (d.xpos[env.tether.body]+d.xmat[env.tether.body].reshape(3,3)@np.asarray(env.cfg.local_point)).copy()
def enter_progress(t):return smooth((t-3.)/.75)
def command_handoff(start,target,t,onset):return spherical(start,target,smooth((t-onset)/.75))
def make_goal(env,model,configuration,point,delta,forward):
 if configuration=='standing':return forward,{'source':'existing verified forward nativegoal','new_optimization':False}
 m=env.sim.model;d=mujoco.MjData(m);q=env.sim.data.qpos.copy();ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in env.sim.dof_names];arm=np.asarray(ids)[22:29];adr=m.jnt_qposadr[arm];lo,hi=m.jnt_range[arm].T;target=point+delta
 def residual(x):
  d.qpos[:]=q;d.qpos[adr]=x;mujoco.mj_forward(m,d);return d.xpos[env.tether.body]+d.xmat[env.tether.body].reshape(3,3)@np.asarray(env.cfg.local_point)-target
 fit=least_squares(residual,np.clip(q[adr],lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=150);error=float(np.linalg.norm(residual(fit.x)));pose=d.qpos.copy()
 z=model.goal_inference(goal_observation(env,pose,np.zeros(m.nv)))
 return z,dict(source='ONE fixed nativegoal from original-range arm geometric feasible example',new_optimization=True,max_nfev=150,nfev=fit.nfev,geometric_error_m=error,target=target.tolist(),target_pose=pose.tolist(),z=z.tolist(),no_physical_IK=True)

def run_study(env,model,contract,out):
 protocol=read(Path(__file__).with_name('m1_protocol.json'));save(out,'protocol.json',protocol);stage=os.environ['M1_STAGE']
 if stage=='representation':
  from m1_representability import run
  return run(env,model,contract,out)
 assert stage in ['qualification','teachers','repair_teacher']
 dev=env.standing.device;m=env.sim.model;orig=model.act;spec=protocol['contact'];onset=spec['onset'];release=spec['release'];end=spec['end'];delta=np.asarray(spec['delta_xyz'])
 sp=read(Path(__file__).with_name('skill_compat_protocol.json'));s1path=R/sp['S1']['path'];a2path=R/sp['A2']['path'];assert sha(s1path)==sp['S1']['sha256'] and sha(a2path)==sp['A2']['sha256']
 s1=load_net(s1path,dev);a2,_=load_student(a2path,dev);digests=[tensor_digest(s1),tensor_digest(a2)]
 raisedpath=R/'issue21-reward-stage-20261030A/inference-attempt-01/outputs/raisearms-l-m-latent.json';raiseddata=read(raisedpath);assert sha(raisedpath)=='3a542d7e836ea557a90c14d5d953537a05f415c8a8401050a65e26c35b2d760c'
 raised=torch.tensor(raiseddata['z'],device=dev,dtype=torch.float32).reshape(1,256)
 forward=torch.tensor(read(R/'issue21-goalcalibration-20261030A/attempt-01/outputs/same-state-forward.json')['states'][-1]['per_mode']['native']['z'],device=dev,dtype=torch.float32)
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qtarget=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 zero=read(R/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics'];initial0=read(R/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json');common=read(R/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
 jobs=[(cfg,'pose_hold',False) for cfg in protocol['configurations']]+[(cfg,ctl,True) for cfg in protocol['configurations'] for ctl in protocol['qualification']['controllers'][cfg]]
 if stage=='teachers':jobs=[(cfg,ctl,True) for cfg in protocol['configurations'] for ctl in ['teacher_pose','teacher_goal']]
 if stage=='repair_teacher':jobs=[(cfg,'accumulated_goal',True) for cfg in protocol['configurations']]
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids];arm=np.arange(22,29);sc=np.asarray(contract['q_PD_scale'])[0];off=np.asarray(contract['q_PD_offset'])[0]
 fade=ReleaseFade();accum=AccumulatedCorrection([.2]*7);corr=np.zeros(7);teacher_decisions=[]
 mode='prefix';pose=env.standing;configuration='standing';controller='standing';enabled=False;target_z=forward
 cs=[];ds=[];gate=ContactGate();memory=ForceMemory(320);raw=torch.zeros_like(env.standing);tick=0;last_q=None;last_z=None;contact_initial=None;contact_state=None;anchor=None;pose_info=None;goal_info=None
 def act(obs,z,mean=True):
  nonlocal raw,tick,last_q,last_z,anchor,contact_initial,contact_state,pose_info,target_z,goal_info,corr
  t=float(env.tether.time);force=env.tether.force.copy();alpha=0.;lg=0.;nominal=orig(obs,env.standing,mean=True)
  if mode=='prefix':return nominal
  # Natural state entry and all controller histories are physically warmed; no sensor reset at handoff.
  if tick%5==0:
   gate.update(t,force);raw=s1(deploy_input(model,obs,env.history,env.keys)).detach()
   ds.append(dict(time=t,tether_time=t,force_history=env.history.tolist(),force_xyz_n=force.tolist(),bfm_obs={k:v.tolist() for k,v in obs.items()},raw=raw.tolist(),active=bool(env.tether.active)))
  alpha=float(a2(student_input(model,obs,[memory.append(force.tolist())],env.keys)).item());lg=gate.value(t)
  if t<onset-1e-8:command=spherical(env.standing,pose,enter_progress(t))
  else:
   if anchor is None:
    point,_=env.tether.point_state();anchor=point.copy();env.tether.origin=anchor.copy();env.tether.anchor=anchor.copy();env.tether.delta=delta.copy()
    contact_initial=copy.deepcopy(env.trace[-1]);contact_initial['wrist']=point.tolist();contact_state=snapshot(env)
    pose_info=geometry(m,env.sim.data.qpos.copy(),delta[0])
    # Static FK only: build exactly one fixed candidate from actual natural state, never retune it.
    target_z,goal_info=make_goal(env,model,configuration,anchor,delta,forward)
   if controller=='standing':desired=env.standing
   elif controller=='S1':desired=constrained(model,env.standing,raw,lg)[0]
   elif controller=='A2':desired=spherical(env.standing,forward,alpha)
   elif controller in ['pose_hold','teacher_pose']:desired=pose
   elif controller in ['goal','teacher_goal','accumulated_goal']:desired=spherical(pose,target_z,smooth((t-onset)/spec['ramp']))
   else:raise AssertionError(controller)
   command=command_handoff(pose,desired,t,onset)
   if controller in ['pose_hold','goal','teacher_pose','teacher_goal','accumulated_goal'] and t>=release-1e-8:command=spherical(command,env.standing,smooth((t-release)/.75))
  base=orig(obs,command,mean=True);base_q=qtarget(base);qbase=base_q[0].cpu().numpy();q0=qtarget(nominal);action=base
  expert=controller.startswith('teacher_') or controller=='accumulated_goal'
  if expert:
   point=cached_point(env);error=env.tether.anchor-point;active=bool(env.tether.active);bounds=None
   if controller=='accumulated_goal':bounds=executable_bounds(qbase[arm],limits[arm],sc[arm],off[arm],contract['clip'])
   if tick%5==0:
    desired=np.zeros(7)
    if active:
     jac=np.zeros((3,m.nv));mujoco.mj_jac(m,env.sim.data,jac,None,point,env.tether.body);desired=arm_ik(jac[:,vadr[arm]],error)
     if controller!='accumulated_goal':desired=np.clip(qbase[arm]+desired,limits[arm,0],limits[arm,1])-qbase[arm]
    if controller=='accumulated_goal':accum.propose(desired,active,*bounds)
    else:corr=fade.update(t,active,desired)
    teacher_decisions.append(dict(time=t,active=active,error_xyz_m=error.tolist(),desired_increment=desired.tolist(),base_q_PD=qbase.tolist()))
   if controller=='accumulated_goal':corr,accum_info=accum.tick(t,active,*bounds)
   elif fade.release_start is not None:corr=fade.value(t)
   if controller!='accumulated_goal':assert abs(corr).max()<=.040001 and np.linalg.norm(corr)<=.080001
   da=torch.zeros_like(base);da[0,22:]=torch.tensor(corr/sc[arm],device=dev,dtype=base.dtype);action=base+da
  q=qtarget(action)
  assert torch.isfinite(action).all() and abs(float(command.norm())-16)<1e-4
  cs.append(dict(time=t,tether_time=t,alpha=alpha,gate=None,local_gate=lg,executed_z=command.tolist(),measured_force_xyz=force.tolist(),bfm_obs={k:v.tolist() for k,v in obs.items()},
    base_action=base[0].tolist(),base_q_PD=base_q[0].tolist(),teacher_correction=corr.tolist(),pre_anchor=env.tether.anchor.tolist(),pre_wrist=cached_point(env).tolist(),active=bool(env.tether.active),action=action[0].tolist(),standing_action=nominal[0].tolist(),nominal=nominal[0].tolist(),q_PD=q[0].tolist(),standing_q_PD=q0[0].tolist(),nominal_q_PD=q0[0].tolist(),delta_q_PD=(q-q0)[0].tolist(),
    pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),action_saturation_fraction=float(((action*5).abs()>=contract['clip']).float().mean()),q_PD_step_max_rad=float(abs(q-last_q).max()),z_step_deg=float(torch.rad2deg(torch.acos(((command*last_z).sum()/256).clamp(-1,1))))))
  last_q=q.clone();last_z=command.clone();tick+=1;return action
 model.act=act;results=dict(status='running',episodes=[],student_optimizer_updates=0);pair_start={};pair_anchor={}
 try:
  for configuration,controller,enabled in jobs:
   check_deadline();label=configuration+'__'+controller+('' if enabled else '__free');mode='prefix';env.qualification_active=False;env.interruption_window=None
   for cfg in (env.cfg,env.tether.cfg):cfg.enabled=False;cfg.activation_time=onset;cfg.ramp_duration=spec['ramp'];cfg.release_time=release
   env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
   for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
    v=getattr(env.base,name);v.copy_(torch.as_tensor(initial0['base_tensors'][name],device=v.device,dtype=v.dtype))
   env.base.common_step_counter=initial0['common_step_counter'];mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,env.sim.data);mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
   for _ in range(30):_,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
   parity={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']};assert max(parity.values())==0;save(out,label+'-prefix.json',parity)
   env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype));assert snapshot(env)==common
   pose=env.standing if configuration=='standing' else raised;cs=[];ds=[];tick=0;gate=ContactGate();memory=ForceMemory(320);raw=torch.zeros_like(env.standing);last_q=qtarget(env.previous_action).clone();last_z=env.standing.clone()
   fade=ReleaseFade();accum=AccumulatedCorrection([.2]*7);corr=np.zeros(7);teacher_decisions=[]
   preparation_initial=copy.deepcopy(env.trace[-1]);start=len(env.trace);anchor=None;contact_initial=None;contact_state=None;pose_info=None;goal_info=None;boundary=None
   for cfg in (env.cfg,env.tether.cfg):cfg.enabled=enabled;cfg.activation_time=onset;cfg.ramp_duration=spec['ramp'];cfg.release_time=release
   env.tether.delta=np.zeros(3);mode='record';env.qualification_active=True;steps=round((end-3)*50)
   for i in range(steps):
    check_deadline()
    try:
     a=model.act(env.obs,env.standing,mean=True);env.obs,_,terminated,truncated,_=env.wrapped.step(a,to_numpy=False);env.steps+=1;env.previous_action=a.clone();env.history=np.roll(env.history,-1,axis=0);env.history[-1]=env.tether.force
     if env.failed or bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any()):boundary=dict(kind='native_failure',time=float(env.tether.time));break
    except UnsafePhysics as exc:boundary=dict(kind='unsafe_physics',physics=exc.row);break
    except ValueError as exc:
     if str(exc) not in ['Original BFM command outside reserved joint range','Joint envelope and correction velocity constraints conflict']:raise
     boundary=dict(kind='control_envelope',time=float(env.tether.time),message=str(exc));break
   mode='prefix';env.qualification_active=False
   condition=dict(name=configuration,onset=onset,ramp=spec['ramp'],duration=release-onset,release=release,end=end,enabled=enabled,delta_xyz=delta.tolist())
   ep=dict(label=label,controller=controller,configuration=configuration,condition=condition,initial=contact_initial or preparation_initial,preparation_initial=preparation_initial,physics=copy.deepcopy(env.trace[start:]),control=cs,decisions=ds,teacher_decisions=teacher_decisions,boundary=boundary,geometry=pose_info,goal_candidate=goal_info,contact_state_sha256=hashlib.sha256(json.dumps(contact_state,sort_keys=True).encode()).hexdigest() if contact_state else None,physical_state_sha256=physical_state_digest(contact_state) if contact_state else None)
   save(out,label+'.json',ep);save(out,label+'-contact-state.json',contact_state)
   entry=dict(label=label,seconds=len(ep['physics'])/200,complete=len(ep['physics'])==steps*4 and boundary is None,boundary=boundary);results['episodes'].append(entry);save(out,'physical-result.json',results);print('M1_PHYSICS',entry,flush=True)
   if not entry['complete']:results['status']='capability-stopped' if boundary and boundary['kind']=='control_envelope' else 'physical-stopped';return results
   if configuration not in pair_start:pair_start[configuration]=ep['physical_state_sha256'];pair_anchor[configuration]=anchor.tolist()
   else:assert ep['physical_state_sha256']==pair_start[configuration] and anchor.tolist()==pair_anchor[configuration]
  if stage!='qualification':
   qa=[x for x in read(RUN/'compute-ledger.json')['attempts'] if x['stage']=='qualification' and x.get('result_status')=='completed'];assert len(qa)==1
   for cfg in protocol['configurations']:
    source=read(Path(qa[0]['path'])/'outputs'/(cfg+'__pose_hold__free.json'));assert pair_start[cfg]==source['physical_state_sha256']
  results['status']='completed';assert [tensor_digest(s1),tensor_digest(a2)]==digests;save(out,'physical-result.json',results);return results
 finally:model.act=orig;env.qualification_active=False
