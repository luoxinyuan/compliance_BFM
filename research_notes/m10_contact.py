"""Finite contact-driven native walking feedback pilot, no optimization/weights.

The partner specifies a real curved tether trajectory. The controller sees only
measured force, pelvis orientation, and its own command/phase memory. Native
reference exhaustion is recorded explicitly, not interpreted as learned intent.
"""
import copy,json,math,types,gzip,os
from pathlib import Path
import numpy as np,torch
from m9_stage import RUN,P,completed
from m9c_stage import RUN as NEW,completed as new_completed
from m8_stage import reset_common,advance
from m9_history import ContactHistory
from m1_stage import save
from locomotion_rollout import spherical,UnsafePhysics
from contact_gate import smooth
from contact_projector import check_deadline

class NativeFeedback:
 def __init__(self):self.filtered=np.zeros(3);self.alpha=0.;self.phase=0.;self.turn=0.
 def step(self,force_world,R,dt=.02):
  self.filtered+=dt/(.25+dt)*(np.asarray(force_world)-self.filtered);f=R.T@self.filtered
  exhausted=self.phase>=6.;target=0. if exhausted else float(np.clip((f[0]-.25)/2.,0,1))
  self.alpha=float(np.clip(target,self.alpha-.9*dt,self.alpha+.6*dt))
  turn=float(np.clip(f[1]/max(abs(f[0]),1.),-1,1));self.turn=float(np.clip(turn,self.turn-dt,self.turn+dt))
  if self.alpha>.05:self.phase=min(6.,self.phase+dt)
  return dict(alpha=self.alpha,phase=self.phase,turn=self.turn,force_body_filtered=f.tolist(),reference_exhausted=exhausted)

def install_partner(tether,spec,anchor):
 from humanoidverse.force_compliance.interaction import spring_force
 def measure(self):
  u=float(np.clip((self.time-spec['onset'])/spec['ramp'],0,1));s=spec['distance']*smooth(u);speed=spec['distance']*6*u*(1-u)/spec['ramp'] if 0<u<1 else 0.;k=spec['curvature']
  if spec.get('sustained'):
   from m9_speedcheck import sustained_profile
   s,speed=sustained_profile(self.time,spec['onset'],**spec['sustained'])
  displacement=np.array([s,0,0]) if abs(k)<1e-9 else np.array([np.sin(k*s)/k,(1-np.cos(k*s))/k,0]);velocity=speed*np.array([np.cos(k*s),np.sin(k*s),0])
  if 'delta_xyz' in spec:displacement=smooth(u)*np.array(spec['delta_xyz']);velocity=6*u*(1-u)/spec['ramp']*np.array(spec['delta_xyz']) if 0<u<1 else np.zeros(3)
  connected=spec['enabled']
  if spec.get('transition'):
   from m7_partner import segment
   base,delta,start,ramp,connected=segment(spec,self.time);u=float(np.clip((self.time-start)/ramp,0,1));displacement=np.array([base+delta*smooth(u),0,0]);velocity=np.array([delta*6*u*(1-u)/ramp,0,0]) if 0<u<1 else np.zeros(3)
  self.anchor=anchor+displacement;self.partner_velocity=velocity;self.active=bool(connected and spec['onset']<=self.time<spec['release']);point,velocity=self.point_state()
  self.force=spring_force(self.anchor,point,self.partner_velocity,velocity,self.cfg.stiffness,self.cfg.damping,self.cfg.max_force) if self.active else np.zeros(3)
  return point,self.force.copy()
 tether.measure=types.MethodType(measure,tether)

def run(env,model,contract,out,cycles=None,composition=False):
 whole=json.loads(Path(__file__).with_name('m10_protocol.json').read_text());stage=os.environ['M9_STAGE'];learning=True;plan=whole['execution'][stage];orig=model.act;dev=env.standing.device
 from m9_learning import old_context,inputs,policy,load_net
 from m8_learning import policy_latent as old_latent
 from contact_projector import ContactGate
 from alpha_student import student_input
 from m9b_learning import memory_policy,load_memory
 old,ends=old_context(env)
 from m9_learning import sha
 assert sha(completed('train2')/'projected-z-final.pt')=='f003afd8953b55e8f96ae1f47aea2beb80622f257cef24c451ea34fe28ab48a6'
 legacy_net=load_net(completed('train2')/'projected-z-final.pt',dev)
 memory_net=None;ablation_net=None
 if True:
  checkpoint=P/'bridge-v2-owned/runs/m9-feedback-continuation-20261101B/train1-attempt-01/outputs/memory-final.pt'
  import hashlib
  assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='57b346158c9ce1bbc18dc0be59991c5f4710bd4077746bf913d4a0c4d3f12634'
  assert hashlib.sha256(checkpoint.with_name('no-memory-final.pt').read_bytes()).hexdigest()=='ad81e5736f165353a3d45afa3dc5f8a11b76c153f4ac7ddc6db5660f8dcbac78'
  memory_net=load_memory(checkpoint,dev)
  ablation_net=load_net(checkpoint.with_name('no-memory-final.pt'),dev)
 transfer_net=None
 if True:
  transfer_path=new_completed('train1')/'student-final.pt'
  if stage=='final':assert sha(transfer_path)==whole['transfer']['final_sha256']
  transfer_net=load_net(transfer_path,dev)
 composition=False
 composition=composition or (learning and whole.get('use_composition_expert',False))
 if composition:turn_z=torch.tensor(json.loads((completed('infer')/'reward-candidates.json').read_text())['rotate-z-0.35-0.65']['z'],device=dev)
 refs={k:torch.tensor(json.loads((completed('tracking')/(k+'-reference.json')).read_text())['z'],device=dev) for k in ['straight','left','right']}
 oldrun=P/'bridge-v2-owned/runs/m8-contact-body-20261101A';qa=[a for a in json.loads((oldrun/'compute-ledger.json').read_text())['attempts'] if a['stage']=='qualification' and a.get('result_status')=='completed'];assert len(qa)==1;freepath=Path(qa[0]['path'])/'outputs/standing.json.gz'
 with gzip.open(freepath,'rt') as f:free=json.load(f)['physics']
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5;rows=[]
 from m10_stage import completed as m10_completed
 candidate_net=load_net(m10_completed('train' if stage=='collect' else 'train1')/'student-final.pt',dev) if stage in ['collect','final','handover'] else None
 if stage in ['final','handover']:assert sha(m10_completed('train1')/'student-final.pt')==whole['final_candidate_sha256']
 from m10_reference import load_variants
 variants,body_goals=load_variants(env,model)
 original_cycles=cycles
 for name,ctl in plan['schedule']:
  cycles=variants['roundtrip' if ctl=='ref_roundtrip' else ('arm' if ctl=='ref_arm' else 'task')] if ctl in ['candidate','candidate_handover','ref_roundtrip','ref_arm','ref_task','ref_projected'] else original_cycles
  spec=plan['conditions'][name];label=name+'__'+ctl;parity=reset_common(env,model,orig);save(out,label+'-prefix.json',parity);rotation=None
  if spec.get('heading_deg'):
   from m9_heading import rotate_ready
   rotation,pivot,initialization=rotate_ready(env,model,spec['heading_deg']);save(out,label+'-heading.json',initialization)
  rr=min(free,key=lambda r:abs(r['time']-spec['onset']));site=spec.get('body','right_wrist_yaw_link');point=np.array(spec.get('local_point',[0,0,0]));anchor=np.array(rr['wrist']) if site=='right_wrist_yaw_link' else np.array(rr['body_positions']['torso_link'])+np.array(rr['torso_matrix'])@point;original_measure=env.tether.measure
  if rotation is not None:anchor=pivot+rotation@(anchor-pivot)
  if learning:
   import mujoco
   env.cfg.body=site;env.cfg.local_point=point.tolist();env.tether.cfg.body=site;env.tether.cfg.local_point=point.tolist();env.tether.body=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_BODY,site);oldgate=ContactGate();tick=0
   from m1_stage import cached_point
   env.last_point=cached_point(env)
   if spec.get('family')=='old' and site=='right_wrist_yaw_link':
    from m6_reference import wrist_at_time
    with gzip.open(P/'bridge-v2-owned/runs/m5-contact-lpp-20261031A/qualification-attempt-01/outputs/free__teacher.json.gz','rt') as f:oldfree=json.load(f)['physics']
    anchor=np.asarray(wrist_at_time(oldfree,spec['onset']))
  for c in [env.cfg,env.tether.cfg]:c.enabled=spec['enabled'];c.activation_time=spec['onset'];c.ramp_duration=spec['ramp'];c.release_time=spec['release']
  install_partner(env.tether,spec,anchor);start=len(env.trace);initial=copy.deepcopy(env.trace[-1]);history=ContactHistory();feedback=NativeFeedback();cs=[]
  initial['wrist']=env.last_point.tolist()
  if cycles is not None:
   from m9_cyclic import CyclicFeedback,sample
   feedback=CyclicFeedback({k:r['period'] for k,r in cycles.items()},ctl!='feedback')
   if ctl=='filtered':
    from m9b_feedback import GuideFilteredFeedback
    feedback=GuideFilteredFeedback({k:r['period'] for k,r in cycles.items()},True)
   if ctl=='cadence':
    from m9b_feedback import CadenceFeedback
    feedback=CadenceFeedback({k:r['period'] for k,r in cycles.items()},True)
  from m9c_feedback import ArmFeedback
  arm_feedback=ArmFeedback(env,contract);projection_previous=None;learned_query=None
  if whole.get('transfer',{}).get('teacher')=='rl' and ctl in ['transfer','learned_expert']:
   from m9c_rl import LearnedQuery
   assert sha(new_completed('rl')/'expert-final.pt')==whole['expert_final_sha256']
   learned_query=LearnedQuery(env,model,cycles,new_completed('rl')/'expert-final.pt')
  from m8_controller import BodyFeedback
  body_feedback=BodyFeedback(json.loads(Path(__file__).with_name('m8_protocol.json').read_text())['body_feedback'])
  boundary=None;last=qof(env.previous_action);previous_z=env.standing.clone();env.qualification_active=True
  try:
   for _ in range(round((spec['end']-float(env.tether.time))*50)):
    check_deadline();t=float(env.tether.time);R=env.sim.data.xmat[1].reshape(3,3).copy();f=env.tether.force.copy();sensor=history.append(f,site,point,R);guide=None if cycles is None else getattr(env.tether,'partner_velocity',np.zeros(3))+.5*(env.tether.anchor-env.last_point)
    phase_before=feedback.phase
    state=feedback.step(f,R) if cycles is None else feedback.step(f,R,guide=guide)
    if learned_query is not None:
     feedback.phase=phase_before+(feedback.phase-phase_before)*learned_query.rate;state['phase']=feedback.phase
    idx=min(300,int(round(state['phase']/.02)));turn=state['turn'];base=refs['straight'][idx:idx+1] if cycles is None else sample(cycles['straight'],state['phase']);other=refs['left' if turn>=0 else 'right'][idx:idx+1] if cycles is None else sample(cycles['left' if turn>=0 else 'right'],state['phase']);native=spherical(base,other,abs(turn));native=model.project_z(sample(cycles['straight'],state['phase'])+(.03+.03*turn)*(turn_z-env.standing)) if composition else native;oracle=spherical(env.standing,native,state['alpha']);z=env.standing if ctl=='standing' else oracle
    if learning:
     if tick%5==0:oldgate.update(t,f)
     lg=oldgate.value(t);h,h8=inputs(model,env.obs,env.keys,history,R);world=np.zeros((320,3));world[-len(history.rows):]=np.array([r['force_world'] for r in history.rows]);legacy=student_input(model,env.obs,[world.tolist()],env.keys);enc=torch.tensor([[float(site=='right_wrist_yaw_link'),float(site=='torso_link'),*point]],device=dev,dtype=legacy.dtype)*(legacy[:,465:].abs().amax(-1,keepdim=True)>1e-8);oldz=old_latent(model,ends,old(torch.cat([legacy,enc],-1)),lg)
     if spec.get('family')=='old':oracle=oldz
     elif learned_query is not None:oracle=learned_query.step(state,tick)
     if ctl in ['legacy','takeover','late_takeover']:z=policy(model,old,ends,legacy_net,h,h8,lg)
     elif ctl=='ablation':z=policy(model,old,ends,ablation_net,h,h8,lg)
     elif ctl=='transfer':z=policy(model,old,ends,transfer_net,h,h8,lg)
     elif ctl=='student':z=memory_policy(model,old,ends,memory_net,h,h8,lg,previous_z)
     if (ctl=='takeover' and t>=6.) or (ctl=='late_takeover' and t>=10.):z=oracle
     elif ctl=='m8':z=oldz
     elif ctl=='expert':z=oracle
     tick+=1
    if ctl=='learned_expert':z=oracle
    if ctl in ['body_original','body_neutral']:
     state={'alpha':body_feedback.step(f),'reference':'original' if ctl=='body_original' else 'neutral'}
     z=spherical(env.standing,body_goals[state['reference']],state['alpha']);oracle=z
    if ctl in ['ref_roundtrip','ref_arm','ref_task','ref_projected']:oracle=z
    if ctl in ['candidate','candidate_handover']:
     if site=='torso_link':
      state={'alpha':body_feedback.step(f),'reference':'neutral'}
      oracle=spherical(env.standing,body_goals['neutral'],state['alpha'])
     z=policy(model,old,ends,candidate_net,h,h8,lg)
    if not torch.isfinite(z).all() or abs(float(z.norm())-16)>1e-4:
     boundary=dict(kind='invalid_student_latent',time=t);break
    a=orig(env.obs,z,mean=True)
    correction=None;projection=None
    if ctl in ['arm','projected','transfer','ref_projected','candidate','candidate_handover'] and learned_query is None and spec.get('family')!='old':
     target,correction=arm_feedback.step(orig(env.obs,oracle,mean=True),t,tick)
     if ctl=='arm':a=target
     else:
      from m9c_feedback import project_action
      projected_z,projection=project_action(model,env.obs,oracle,target,contract,projection_previous)
      projection_previous=projected_z.clone();oracle=projected_z
      if ctl in ['projected','ref_projected']:z=projected_z;a=orig(env.obs,z,mean=True)
    handover_alpha=0.
    if ctl=='candidate_handover':
     handover_alpha=smooth((t-(spec['onset']+3.))/.6)
     z=spherical(z,oracle,handover_alpha);a=orig(env.obs,z,mean=True)
    if not torch.isfinite(a).all():boundary=dict(kind='nonfinite_actor_action',time=t);break
    q=qof(a);oq=qof(orig(env.obs,oracle,mean=True))
    cs.append(dict(diagnostic_handover_alpha=handover_alpha,learned_query_raw=None if learned_query is None else learned_query.raw.tolist(),correction=correction,projection=projection,time=t,controller=ctl,expert_query_guide_world=None if guide is None else guide.tolist(),own_previous_z=previous_z.tolist(),oracle_controller_state=state,native_execution_state=state if ctl in ['feedback','privileged'] or (ctl=='expert' and spec.get('family')!='old') else None,executed_z=z.tolist(),action=a[0].tolist(),q_PD=q[0].tolist(),oracle_z=oracle.tolist(),oracle_q_PD=oq[0].tolist(),oracle_query_valid=True,q_PD_step_max_rad=float((q-last).abs().max()),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),sensor=sensor,measured_force_xyz=f.tolist(),bfm_obs={k:v.tolist() for k,v in env.obs.items()},action_saturation_fraction=float(((a*5).abs()>=contract['clip']).float().mean())))
    last=q.clone();previous_z=z.detach().clone()
    try:
     if advance(env,a):boundary=dict(kind='native_termination',time=float(env.tether.time));break
    except UnsafePhysics as e:boundary=dict(kind='physics_review',physics=e.row);break
  finally:env.qualification_active=False;env.tether.measure=original_measure
  ep=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=cs,boundary=boundary,condition=spec,anchor=anchor.tolist(),phases={'pre':[3,spec['onset']],'move':[spec['onset'],spec['onset']+spec['ramp']],'hold':[spec['onset']+spec['ramp'],spec['release']],'recover':[spec['release'],spec['end']]});save(out,label+'.json',ep)
  rows.append(dict(label=label,complete=boundary is None,seconds=len(ep['physics'])/200,boundary=boundary));save(out,'contact-summary.json',dict(episodes=rows));print('M9_CONTACT',label,'complete',boundary is None,flush=True)
 return dict(status='completed',episodes=rows,student_optimizer_updates=0,scope=('Frozen execution; reference ablation or Student same-state expert labeling, no online updates' if learning else 'Native feedback expert qualification; actual failures remain evidence'))
