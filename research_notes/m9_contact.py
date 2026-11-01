"""Finite contact-driven native walking feedback pilot, no optimization/weights.

The partner specifies a real curved tether trajectory. The controller sees only
measured force, pelvis orientation, and its own command/phase memory. Native
reference exhaustion is recorded explicitly, not interpreted as learned intent.
"""
import copy,json,math,types,gzip
from pathlib import Path
import numpy as np,torch
from m9_stage import RUN,P,completed
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
  displacement=np.array([s,0,0]) if abs(k)<1e-9 else np.array([np.sin(k*s)/k,(1-np.cos(k*s))/k,0])
  self.anchor=anchor+displacement;self.partner_velocity=speed*np.array([np.cos(k*s),np.sin(k*s),0]);self.active=bool(spec['enabled'] and spec['onset']<=self.time<spec['release']);point,velocity=self.point_state()
  self.force=spring_force(self.anchor,point,self.partner_velocity,velocity,self.cfg.stiffness,self.cfg.damping,self.cfg.max_force) if self.active else np.zeros(3)
  return point,self.force.copy()
 tether.measure=types.MethodType(measure,tether)

def run(env,model,contract,out):
 plan=json.loads(Path(__file__).with_name('m9_protocol.json').read_text())['contact_pilot'];orig=model.act;dev=env.standing.device
 refs={k:torch.tensor(json.loads((completed('tracking')/(k+'-reference.json')).read_text())['z'],device=dev) for k in ['straight','left','right']}
 freepath=P/'bridge-v2-owned/runs/m8-contact-body-20261101A/qualification-attempt-01/outputs/standing.json.gz'
 with gzip.open(freepath,'rt') as f:free=json.load(f)['physics']
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qof=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5;rows=[]
 for name,ctl in plan['schedule']:
  spec=plan['conditions'][name];label=name+'__'+ctl;parity=reset_common(env,model,orig);save(out,label+'-prefix.json',parity);anchor=np.array(min(free,key=lambda r:abs(r['time']-spec['onset']))['wrist']);original_measure=env.tether.measure
  for c in [env.cfg,env.tether.cfg]:c.enabled=spec['enabled'];c.activation_time=spec['onset'];c.ramp_duration=spec['ramp'];c.release_time=spec['release']
  install_partner(env.tether,spec,anchor);start=len(env.trace);initial=copy.deepcopy(env.trace[-1]);history=ContactHistory();feedback=NativeFeedback();cs=[];boundary=None;last=qof(env.previous_action);env.qualification_active=True
  try:
   for _ in range(round((spec['end']-3)*50)):
    check_deadline();t=float(env.tether.time);R=env.sim.data.xmat[1].reshape(3,3).copy();f=env.tether.force.copy();sensor=history.append(f,'right_wrist_yaw_link',[0,0,0],R);state=feedback.step(f,R)
    idx=min(300,int(round(state['phase']/.02)));turn=state['turn'];base=refs['straight'][idx:idx+1];other=refs['left' if turn>=0 else 'right'][idx:idx+1];native=spherical(base,other,abs(turn));z=env.standing if ctl=='standing' else spherical(env.standing,native,state['alpha']);a=orig(env.obs,z,mean=True);q=qof(a)
    cs.append(dict(time=t,**state,executed_z=z.tolist(),action=a[0].tolist(),q_PD=q[0].tolist(),q_PD_step_max_rad=float((q-last).abs().max()),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),sensor=sensor,measured_force_xyz=f.tolist(),bfm_obs={k:v.tolist() for k,v in env.obs.items()},action_saturation_fraction=float(((a*5).abs()>=contract['clip']).float().mean())))
    last=q.clone()
    try:
     if advance(env,a):boundary=dict(kind='native_termination',time=float(env.tether.time));break
    except UnsafePhysics as e:boundary=dict(kind='physics_review',physics=e.row);break
  finally:env.qualification_active=False;env.tether.measure=original_measure
  ep=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=cs,boundary=boundary,condition=spec,anchor=anchor.tolist(),phases={'pre':[3,spec['onset']],'move':[spec['onset'],spec['onset']+spec['ramp']],'hold':[spec['onset']+spec['ramp'],spec['release']],'recover':[spec['release'],spec['end']]});save(out,label+'.json',ep)
  rows.append(dict(label=label,complete=boundary is None,seconds=len(ep['physics'])/200,boundary=boundary));save(out,'contact-summary.json',dict(episodes=rows));print('M9_CONTACT',label,'complete',boundary is None,flush=True)
 return dict(status='completed',episodes=rows,student_optimizer_updates=0,scope='Native feedback pilot. No Student trained; actual failures remain evidence.')
