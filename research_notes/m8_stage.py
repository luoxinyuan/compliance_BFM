"""M8 physical qualification and multi-contact LPP. Frozen native BFM/PD only."""
import copy,json,math,os,hashlib
from pathlib import Path
import numpy as np,torch,mujoco
from locomotion_rollout import UnsafePhysics,snapshot,spherical
from contact_projector import check_deadline
from contact_gate import smooth
from m1_stage import save,cached_point,physical_state_digest
from wholebody_physics import goal_observation
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs';RUN=R/'m8-contact-body-20261101A'
read=lambda p:json.loads(Path(p).read_text())

def reset_common(env,model,orig):
 m=env.sim.model;dev=env.standing.device
 zero=read(R/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics'];initial=read(R/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json');common=read(R/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
 model.act=orig;env.qualification_active=False;env.interruption_window=None;env.m1_anchor_hook=None;env.cfg.body='right_wrist_yaw_link';env.cfg.local_point=[0.,0.,0.];env.cfg.enabled=False
 env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
 for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
  v=getattr(env.base,name);v.copy_(torch.as_tensor(initial['base_tensors'][name],device=dev,dtype=v.dtype))
 env.base.common_step_counter=initial['common_step_counter'];mujoco.mj_setState(m,env.sim.data,np.asarray(initial['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,env.sim.data);mujoco.mj_setState(m,env.sim.data,np.asarray(initial['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
 for _ in range(30):
  _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
 parity={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']};assert max(parity.values())==0,parity
 env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype));assert snapshot(env)==common
 return parity

def advance(env,a):
 env.obs,_,terminated,truncated,_=env.wrapped.step(a,to_numpy=False);env.steps+=1;env.previous_action=a.clone();env.history=np.roll(env.history,-1,axis=0);env.history[-1]=env.tether.force
 return env.failed or bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any())

def qualify(env,model,contract,out):
 plan=read(Path(__file__).with_name('m8_protocol.json'))['qualification'];audit=read(RUN/plan['asset_audit']);orig=model.act;m=env.sim.model;dev=env.standing.device
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 goals=[]
 for i,row in enumerate(audit['selected']):
  z=model.goal_inference(goal_observation(env,np.asarray(row['qpos']),np.zeros(m.nv)))
  goals.append(z);save(out,f'candidate-{i}.json',dict(**row,z=z.tolist(),norm=float(z.norm()),angle_deg=float(torch.rad2deg(torch.acos(((z*env.standing).sum()/256).clamp(-1,1)))),asset_sha256=audit['asset_sha256'],source='Existing natural motion frame, original goal_inference with zero velocity; no live pose reset'))
 result=dict(status='completed',episodes=[],student_optimizer_updates=0)
 def alpha(t):return smooth((t-plan['entry_start'])/plan['entry_duration']) if t<plan['hold_until'] else 1.-smooth((t-plan['hold_until'])/plan['return_duration'])
 for idx in [-1]+list(range(len(goals))):
  label='standing' if idx<0 else f'candidate-{idx}';parity=reset_common(env,model,orig);save(out,label+'-prefix.json',parity);goal=env.standing if idx<0 else goals[idx];initial=copy.deepcopy(env.trace[-1]);cs=[];last=qof(env.previous_action).clone();start=len(env.trace);boundary=None
  # Same-state command continuity check. This is not a dynamic proof.
  qs=[qof(orig(env.obs,spherical(env.standing,goal,alpha(float(t))),mean=True)) for t in np.arange(3,plan['end'],.02)]
  step=max(float(torch.rad2deg((b-a).abs().max())) for a,b in zip(qs,qs[1:]));save(out,label+'-command-screen.json',dict(max_qpd_step_deg=step))
  if step>plan['screen_qpd_step_deg']:boundary=dict(kind='command_screen',qpd_step_deg=step)
  else:
   env.qualification_active=True
   for _ in range(round((plan['end']-3)*50)):
    check_deadline();t=float(env.tether.time);z=spherical(env.standing,goal,alpha(t));a=orig(env.obs,z,mean=True);q=qof(a)
    cs.append(dict(time=t,alpha=alpha(t),executed_z=z.tolist(),q_PD=q[0].tolist(),action=a[0].tolist(),q_PD_step_max_rad=float((q-last).abs().max()),action_saturation_fraction=float(((a*5).abs()>=contract['clip']).float().mean()),bfm_obs={k:v.tolist() for k,v in env.obs.items()},pre_qpos=env.sim.data.qpos.tolist()))
    last=q.clone()
    try:
     if advance(env,a):boundary=dict(kind='native_termination',time=float(env.tether.time));break
    except UnsafePhysics as e:boundary=dict(kind='qualification_physics',physics=e.row);break
  env.qualification_active=False;ep=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=cs,boundary=boundary,condition=dict(enabled=False,onset=3.8,release=9.,end=17.),goal_index=idx);save(out,label+'.json',ep)
  result['episodes'].append(dict(label=label,complete=boundary is None,boundary=boundary,seconds=len(ep['physics'])/200));save(out,'qualification-summary.json',result);print('M8_QUALIFY',label,result['episodes'][-1]['complete'],flush=True)
 return result

def run_study(env,model,contract,out):
 stage=os.environ['M8_STAGE']
 if stage=='qualification':return qualify(env,model,contract,out)
 if stage.startswith('train'):
  from m8_learning import train
  return train(env,model,contract,out,int(stage[-1]))
 return contact_study(env,model,contract,out)

def completed(stage):
 a=[a for a in read(RUN/'compute-ledger.json')['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(a)==1;return Path(a[0]['path'])/'outputs'

def contact_study(env,model,contract,out):
 from alpha_student import student_input
 from alpha_memory import ForceMemory
 from contact_projector import ContactGate
 from m1_interface import endpoints
 from m7_learning import load_net as old_load,policy_latent as old_latent
 from ccl02a_teacher import arm_ik,ReleaseFade
 from m8_controller import BodyFeedback
 from frozen_bfm_placement import tensor_digest
 stage=os.environ['M8_STAGE'];plan=read(Path(__file__).with_name('m8_protocol.json'));dev=env.standing.device;m=env.sim.model;orig=model.act;ends=endpoints(env);body_goal=torch.tensor(read(completed('qualification')/'candidate-1.json')['z'],device=dev)
 olddir=R/'m7-continuous-20261101A/train2-attempt-02/outputs';old=old_load(olddir/'distill-final.pt',dev);oldteacher=old_load(olddir/'alignment-final.pt',dev);nets={}
 if stage.startswith('collect') or stage=='final':
  from m8_learning import load_net,policy_latent
  rnd=2 if stage=='final' else int(stage[-1])-1
  for method in ['distill','alignment']:nets[method]=load_net(completed('train'+str(rnd))/(method+'-final.pt'),dev)
  if stage=='final':assert {k:hashlib.sha256((completed('train2')/(k+'-final.pt')).read_bytes()).hexdigest() for k in nets}==plan['final_checkpoint_sha256']
 identities={k:tensor_digest(n) for k,n in dict(old=old,teacher=oldteacher,**nets).items()}
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids];sc=np.asarray(contract['q_PD_scale'])[0]
 import gzip
 free=json.loads(gzip.decompress((completed('qualification')/'standing.json.gz').read_bytes()))['physics'];freewrist=json.loads(gzip.decompress((R/'m5-contact-lpp-20261031A/qualification-attempt-01/outputs/free__teacher.json.gz').read_bytes()))['physics']
 results=dict(status='completed',episodes=[],student_optimizer_updates=0)
 for name,ctl in plan['schedule'][stage]:
  spec=plan['conditions'][name];label=name+'__'+ctl;parity=reset_common(env,model,orig);save(out,label+'-prefix.json',parity)
  body=spec['body'];local=np.asarray(spec['local_point']);isbody=body=='torso_link';env.cfg.body=body;env.cfg.local_point=local.tolist();env.tether.cfg.body=body;env.tether.cfg.local_point=local.tolist();env.tether.body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body);assert env.tether.body>0
  env.last_point=cached_point(env);initial=copy.deepcopy(env.trace[-1]);initial['wrist']=env.last_point.tolist();ready_sha=physical_state_digest(snapshot(env));start=len(env.trace);cs=[];boundary=None;lastq=qof(env.previous_action).clone();lastz=env.standing.clone();anchor=None
  for cfg in [env.cfg,env.tether.cfg]:cfg.enabled=spec['enabled'];cfg.activation_time=spec['onset'];cfg.ramp_duration=spec['ramp'];cfg.release_time=spec['release']
  memory=ForceMemory(320);gate=ContactGate();feedback=BodyFeedback(plan['body_feedback']);fade=ReleaseFade();corr=np.zeros(7);tick=0
  def hook(t):
   nonlocal anchor
   if anchor is None and t>=spec['onset']-1e-8:
    if isbody:
     rr=min(free,key=lambda r:abs(r['time']-spec['onset']));assert abs(rr['time']-spec['onset'])<.00251
     anchor=np.asarray(rr['body_positions']['torso_link'])+np.asarray(rr['torso_matrix'])@local
    else:
     from m6_reference import wrist_at_time
     anchor=np.asarray(wrist_at_time(freewrist,spec['onset']))
    env.tether.origin=anchor.copy();env.tether.anchor=anchor.copy();env.tether.delta=np.asarray(spec['delta_xyz'])
   if anchor is not None and spec.get('transition'):
    from m7_partner import install
    install(env.tether,spec,anchor)
  env.m1_anchor_hook=hook;env.qualification_active=True
  try:
   for _ in range(round((spec['end']-3)*50)):
    check_deadline();t=float(env.tether.time);f=env.tether.force.copy()
    if tick%5==0:gate.update(t,f)
    lg=gate.value(t);h0=student_input(model,env.obs,[memory.append(f.tolist())],env.keys);encoding=torch.tensor([[float(not isbody),float(isbody),*local]],device=dev,dtype=h0.dtype);h=torch.cat([h0,encoding],-1)
    ba=feedback.step(f);body_z=spherical(env.standing,body_goal,ba);oldraw=oldteacher(h0);oldbase=spherical(env.standing,ends[2],float(oldraw[0,0].clamp(0,1)));oldz=old_latent(model,ends,oldraw,lg)
    expertz=body_z if isbody else oldz;expertbase=body_z if isbody else oldbase;oa=orig(env.obs,expertz,mean=True);oq=qof(oa)
    if not isbody:
     if tick%5==0:
      desired=np.zeros(7)
      if env.tether.active:
       jac=np.zeros((3,m.nv));point=cached_point(env);mujoco.mj_jac(m,env.sim.data,jac,None,point,env.tether.body);desired=arm_ik(jac[:,vadr[22:]],env.tether.anchor-point);desired=np.clip(oq[0,22:].cpu().numpy()+desired,limits[22:,0],limits[22:,1])-oq[0,22:].cpu().numpy()
      corr=fade.update(t,bool(env.tether.active),desired)
     if fade.release_start is not None:corr=fade.value(t)
     assert np.max(abs(corr))<=.040001 and np.linalg.norm(corr)<=.080001
     oa=oa.clone();oa[0,22:]+=torch.tensor(corr/sc[22:],device=dev,dtype=oa.dtype)
    raw=None;pe=None
    if ctl in nets:
     raw=nets[ctl](h);z=policy_latent(model,[ends[0],ends[2],body_goal],raw,lg);a=orig(env.obs,z,mean=True)
    elif ctl=='teacher':z=expertz;a=oa
    elif ctl=='projected':
     from m8_learning import project
     z,pe=project(model,env.obs,expertbase,lg,qof(oa),qof,isbody);a=orig(env.obs,z,mean=True)
    elif ctl=='rule':
     alpha=float(np.clip(-f[2]/5.,0,1)) if isbody else float(np.clip(f[0]/5.,0,1));z=spherical(env.standing,body_goal if isbody else ends[2],alpha);a=orig(env.obs,z,mean=True)
    elif ctl=='m7':z=old_latent(model,ends,old(h0),lg);a=orig(env.obs,z,mean=True)
    else:assert ctl=='standing';z=env.standing;a=orig(env.obs,z,mean=True)
    q=qof(a);nom=orig(env.obs,env.standing,mean=True);assert torch.isfinite(a).all() and abs(float(z.norm())-16)<1e-4
    cs.append(dict(time=t,executed_z=z.tolist(),q_PD=q[0].tolist(),action=a[0].tolist(),standing_action=nom[0].tolist(),standing_q_PD=qof(nom)[0].tolist(),q_PD_step_max_rad=float((q-lastq).abs().max()),z_step_deg=float(torch.rad2deg(torch.acos(((z*lastz).sum()/256).clamp(-1,1)))),action_saturation_fraction=float(((a*5).abs()>=contract['clip']).float().mean()),bfm_obs={k:v.tolist() for k,v in env.obs.items()},student_input=h[0].tolist(),student_raw=raw[0].tolist() if raw is not None else None,measured_force_xyz=f.tolist(),local_gate=lg,oracle_z=expertbase.tolist(),oracle_feedback_z=expertz.tolist(),oracle_action=oa[0].tolist(),oracle_q_PD=qof(oa)[0].tolist(),oracle_base_q_PD=oq[0].tolist(),oracle_query_valid=True,body_progress=ba,teacher_correction=corr.tolist(),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),projection_rms_deg=None if pe is None else float(torch.rad2deg(pe.square().mean().sqrt()))))
    lastq=q.clone();lastz=z.clone();tick+=1
    try:
     if advance(env,a):boundary=dict(kind='native_termination',time=float(env.tether.time));break
    except UnsafePhysics as e:boundary=dict(kind='unsafe_physics',physics=e.row);break
  except Exception as exc:
   save(out,label+'-partial.json',dict(label=label,initial=initial,condition=spec,physics=copy.deepcopy(env.trace[start:]),control=cs,boundary=dict(kind=type(exc).__name__,message=str(exc))));raise
  finally:env.qualification_active=False;env.m1_anchor_hook=None
  ep=dict(label=label,controller=ctl,condition=spec,initial=initial,ready_physical_state_sha256=ready_sha,physics=copy.deepcopy(env.trace[start:]),control=cs,boundary=boundary,point_field='wrist means actual selected contact point; body_positions contains real wrists separately')
  save(out,label+'.json',ep);row=dict(label=label,complete=boundary is None,seconds=len(ep['physics'])/200,boundary=boundary);results['episodes'].append(row);save(out,'physical-result.json',results);print('M8_PHYSICS',label,row['complete'],flush=True)
  if boundary:results['status']='physical-stopped';break
 assert {k:tensor_digest(n) for k,n in dict(old=old,teacher=oldteacher,**nets).items()}==identities
 return results
