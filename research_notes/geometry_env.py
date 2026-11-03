"""Expanded current-feedback privileged contact slots; frozen motor execution unchanged."""
import copy,json,os,math
from pathlib import Path
import numpy as np,torch
from clean_env import Task as Original
from geometry_contact import install as install_contact,frame
PLAN=json.loads(Path(__file__).with_name('geometry_protocol.json').read_text())

def wrap(a):return math.atan2(math.sin(a),math.cos(a))
class Task(Original):
 def __init__(self,*a):
  super().__init__(*a);self.geometry_enabled=False;self.original_apply=self.e.tether.apply;self.original_measure=self.e.tether.measure
 def reset(self,*a,**kw):
  import mujoco
  self.geometry_enabled=False;t=self.e.tether;t.apply=self.original_apply;t.measure=self.original_measure;t.body=mujoco.mj_name2id(t.model,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');t.cfg.local_point=[0,0,0]
  super().reset(*a,**kw);install_contact(self);self.prepared=False;self.prepare();return self.features()
 def sample_spec(self):
  arm=os.environ.get('GEOMETRY_ARM','B');names=PLAN['train_'+arm];spec=copy.deepcopy(PLAN['dev'][str(self.rng.choice(names))])
  spec['onset']+=float(self.rng.uniform(-.12,.12));spec['ramp']+=float(self.rng.uniform(-.3,.3));spec['distance']*=float(self.rng.uniform(.92,1.08))
  if 'rotation_deg' in spec:spec['rotation_deg']*=float(self.rng.uniform(.9,1.1))
  spec['release']=spec['onset']+spec['ramp']+2;spec['end']=spec['release']+5;return spec
 def prepare(self):
  fresh=not self.prepared;out=super().prepare()
  if fresh and self.geometry_enabled:
   rows=copy.deepcopy(self.e.tether.geometry_rows);self.contact_history.append(rows)
  return out
 def contact_features(self,privileged):
  R=self.e.sim.data.xmat[1].reshape(3,3);h=np.zeros((100,3,3));rows=list(self.contact_history)
  for i,sample in enumerate(rows):
   for r in sample:h[-len(rows)+i,r['slot']]=R.T@np.array(r['force'])/5
  enc=np.zeros((3,6));extra=np.zeros((3,6))
  for r in self.e.tether.geometry_rows:
   j=r['slot'];enc[j]=[1,float(r['link']=='right_wrist_yaw_link'),float(r['link']=='torso_link'),*r['local']]
   if privileged:
    extra[j,:3]=R.T@(np.array(r['anchor'])-r['point'])/.25 if r['active'] else 0
    extra[j,3:]=R.T@(np.array(r['anchor_velocity'])-r['velocity'])/.5 if r['active'] else 0
  # Slot mount-present is a known instrumented attachment, not hidden contact activation.
  v=np.r_[h[-25:].reshape(-1),enc.reshape(-1),extra.reshape(-1)] if privileged else np.r_[h.reshape(-1),enc.reshape(-1)]
  if privileged:
   torso=self.e.sim.data.xmat[self.e.tether.body].reshape(3,3) if self.spec.get('geometry')=='torso_pair' else R
   err=wrap(self.e.tether.geometry_target_yaw-math.atan2(torso[1,0],torso[0,0]));v=np.r_[v,math.sin(err),math.cos(err)]
  return torch.tensor(v,dtype=torch.float32,device=self.e.standing.device)[None]
 def features(self):
  base=super().features()
  if not self.geometry_enabled:return torch.cat([base,torch.zeros((1,263),device=base.device)],-1)
  return torch.cat([base,self.contact_features(True)],-1)
 def student_features(self):
  self.prepare();return torch.cat([self.pro.clamp(-10,10),self.last_command/16,self.contact_features(False)],-1)
 def step(self,*a,**kw):
  # Retain raw-coordinate action, native 50Hz actor, physics, terminal treatment.
  # Replace ONLY task reward/statistics from exact per-tick recorded contact loads.
  self._geometry_start=len(self.physics);x,re,done,st=super().step(*a,**kw)
  rows=self.physics[self._geometry_start:] if self.record else []
  if not rows:raise RuntimeError('Geometry task requires recorded physical contact rows')
  active=[r for r in rows if r['active']];fs=sum(sum(np.dot(c['force'],c['force']) for c in r['contacts']) for r in active)
  # Equal budget normalization: n*sum||Fi||²/25. At symmetric full cap this is1 for either1 or2points.
  forcecost=np.mean([len(r['contacts'])*sum(np.dot(c['force'],c['force']) for c in r['contacts'])/25 for r in rows])
  ecost=np.mean([np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in r['contacts']]) if r['active'] else 0 for r in rows])
  orientation=[]
  for r in rows:
   if not r['active']:orientation.append(0.);continue
   P=np.array(r['pelvis_matrix']);T=np.array(r['torso_matrix']);p=wrap(r['target_yaw']-math.atan2(P[1,0],P[0,0]));t=wrap(r['target_yaw']-math.atan2(T[1,0],T[0,0]));pair=self.spec.get('geometry')=='torso_pair'
   orientation.append(min(((.8*t*t+.2*p*p) if pair else p*p)/math.radians(20)**2,9)*(.5 if pair else .35*min(self.spec['distance']/.8,1)))
  old=st['terms'];new=dict(old);new.update(force=1.5*forcecost,tracking=.15*min(ecost/.04,16),heading=float(np.mean(orientation)))
  # Replace centroid demand with mean per-point demand; never cancel opposing forces.
  new['uncapped']=.15*np.mean([np.mean([min(np.log1p(np.dot(c['raw_force'],c['raw_force'])/25),8) for c in r['contacts']]) for r in rows])
  new['relative_velocity']=.1*np.mean([np.mean([min(np.sum((np.array(c['anchor_velocity'])-c['velocity'])**2)/.25,16) for c in r['contacts']]) if r['active'] else 0 for r in rows])
  delta=-.02*st['control_steps']*(sum(new.values())-sum(old.values()));re+=delta;self.episode_return+=delta;st.update(terms=new,reward=re,active_force_squared=fs,error_squared=sum(np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in r['contacts']]) for r in active),cap_steps=sum(any(np.linalg.norm(c['force'])>=c['cap']-.001 for c in r['contacts']) for r in active))
  for c in self.controls[-st['control_steps']:]:
   if 'reward_terms' in c:c['legacy_reward_terms_not_used']=c.pop('reward_terms');c['legacy_reward_integral_not_used']=c.pop('reward_integral')
  st['reward_source']='per-tick per-contact geometry reward in rl_stats; legacy per-control terms are not optimized'
  return x,re,done,st
 def save(self,out):
  row=super().save(out);active=[r for r in self.physics if r['active']];moving=[r for r in active if r['force_time']<self.spec['onset']+self.spec['ramp']]
  yaw=lambda r,key:math.atan2(np.array(r[key])[1,0],np.array(r[key])[0,0])
  row.update(geometry_contract=self.geometry_contract,torso_heading_rms_deg=float(np.degrees(np.sqrt(np.mean([wrap(r['target_yaw']-yaw(r,'torso_matrix'))**2 for r in active])))) if active else 0.,moving_heading_rms_deg=float(np.degrees(np.sqrt(np.mean([wrap(r['target_yaw']-yaw(r,'pelvis_matrix'))**2 for r in active])))) if active else 0.,yaw_moment_rms=float(np.sqrt(np.mean([r['net_yaw_moment_com']**2 for r in active]))) if active else 0.,work_j=sum(r['external_power']*.005 for r in self.physics),sum_force_norm_rms=float(np.sqrt(np.mean([r['sum_force_norm']**2 for r in active]))) if active else 0.,actual_pelvis_turn_deg=float(np.degrees(wrap(yaw(moving[-1],'pelvis_matrix')-yaw(self.physics[0],'pelvis_matrix')))) if moving else 0.,actual_torso_turn_deg=float(np.degrees(wrap(yaw(moving[-1],'torso_matrix')-yaw(self.physics[0],'torso_matrix')))) if moving else 0.)
  return row

def install():
 import clean_env
 clean_env.Task=Task
