"""Same live physics/Direct interface, only native source prefixes and curriculum.
No old high-level actor is instantiated or queried by this module.
"""
import copy,json,hashlib,gzip,math
from pathlib import Path
import numpy as np,torch
from geometry_env import Task as Base
from clean_policy import direct_command
from locomotion_rollout import spherical
from m9_cyclic import sample
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-contact-20261106')

def install():
 import geometry_env,reach_contact
 geometry_env.install_contact=reach_contact.install

def spec(distance=.72,ramp=8.,onset=4.9,source='standing',until=4.,enabled=True,stop=None,cap=10.):
 s=dict(geometry='wrist',enabled=enabled,distance=distance,ramp=ramp,onset=onset,release=onset+ramp+2,end=onset+ramp+6,rotation_deg=0.,curvature=0.,heading_deg=0.,family='walking',source=source,source_until=until,total_cap=cap,anchor_mode='onset_actual')
 if stop is not None:s.update(stop_time=stop,deceleration=.7,release=stop+.7+2,end=stop+.7+6)
 return s

def dev_specs():
 return {'zero':spec(0.,4.),'none':spec(0.,4.,enabled=False),'near':spec(.07,1.8),'mid':spec(.2,3.),'long':spec(.72,8.),'stop':spec(.72,8.,stop=8.1),'arm':spec(.2,3.1,6.2,'goal_arm5',5.6),'walk_continue':spec(.5,6.,6.5,'motion_walk_task',5.9),'walk_stop':spec(.5,6.,6.5,'motion_walk_task',5.9,stop=9.3)}

class Task(Base):
 def __init__(self,*args):
  super().__init__(*args);self.source_physics_total=0;self.assets=json.loads((RUN/'native-assets.json').read_text());self.native_z={k:torch.tensor(v,device=self.e.standing.device,dtype=torch.float32).reshape(1,256) for k,v in self.assets['static_z'].items()}
 def sample_spec(self):
  names=['zero','none','near','mid','long','stop','arm','walk_continue','walk_stop'];weights=[.08,.08,.18,.14,.18,.10,.08,.08,.08];name=str(self.rng.choice(names,p=weights));s=copy.deepcopy(dev_specs()[name]);dt=float(self.rng.uniform(-.1,.1))
  # Random prefixes are actual source-policy execution; onset stays beyond handoff.
  s['source_until']+=float(self.rng.uniform(-.15,.15));s['distance']*=float(self.rng.uniform(.9,1.1));s['ramp']*=float(self.rng.uniform(.95,1.05))
  for k in ['onset','release','end','stop_time']:
   if k in s:s[k]+=dt
  s['demand_family']=name;return s
 def native(self,name,t):
  if name=='standing':return self.e.standing
  u=float(np.clip((t-3.)/1.5,0,1));u=u*u*(3-2*u)
  if name=='motion_walk_task':target=sample(self.refs['straight'],max(0,t-4.5)/self.refs['straight']['period'])
  else:target=self.native_z[name]
  return spherical(self.e.standing,target,u)
 def reset(self,s=None,**kw):
  s=copy.deepcopy(s or self.sample_spec());self.prefix_data=None
  x=super().reset(s,**kw);source=s.get('source','standing');until=s.get('source_until',4.)
  while float(self.e.tether.time)<until-1e-7:
   _,_,done,_=super().step(self.native(source,float(self.e.tether.time))/16,prefix=True)
   if done:raise RuntimeError('native source prefix failed: '+str(self.boundary))
  import mujoco
  state=np.zeros(mujoco.mj_stateSize(self.e.sim.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(self.e.sim.model,self.e.sim.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
  audit={'source':source,'time':float(self.e.tether.time),'integration_state':state.tolist(),'bfm_obs':{k:v.tolist() for k,v in self.e.obs.items()},'previous_action':self.e.previous_action.tolist(),'last_z':self.last_command.tolist(),'last_qpd':self.lastq.tolist(),'force_history':list(self.contact_history),'features':self.features().tolist()}
  audit['sha256']=hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest();self.source_physics_total+=len(self.physics)
  self.prefix_data=dict(physics=self.physics,control=self.controls,audit=audit,initial=self.initial);self.spec['handoff_time']=float(self.e.tether.time)
  if self.physics:self.initial=copy.deepcopy(self.physics[-1])
  self.physics=[];self.controls=[];self.stats=[];self.decision_rows=[];self.execution_stats=[];self.episode_return=0.;self.root_max=0.
  return self.features()
 def step(self,*a,**kw):
  t=float(self.e.tether.time);x,r,d,s=super().step(*a,**kw)
  from transition_env import phase
  s.update(phase=phase(self.spec,t),demand_family=self.spec.get('demand_family',self.label));return x,r,d,s
 def save(self,out):
  r=super().save(out)
  from m39a_metrics import extra
  r.update(extra(self));r['source']=self.spec.get('source','standing');r['handoff_sha']=self.prefix_data['audit']['sha256'];r['source_physics_steps']=len(self.prefix_data['physics'])
  # Keep existing suffix phase metrics; helper reads only real saved physics.
  from transition_metrics import extra as transition_extra
  r.update(transition_extra(self))
  with gzip.open(out/(self.label+'-source-prefix.json.gz'),'wt') as f:json.dump(self.prefix_data,f)
  return r
