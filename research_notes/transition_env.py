"""Live source-policy execution then takeover. No qpos-only resets or history clearing.
Source identity/time are environment curriculum metadata, not network features.
"""
import copy,json,os,hashlib,gzip
from pathlib import Path
import numpy as np,torch
from geometry_env import Task as Base
from m39a_env import install_frame as old_install
PLAN=json.loads(Path(__file__).with_name('transition_protocol.json').read_text())
def install_frame():
 old_install()
 import geometry_env,transition_contact
 geometry_env.install_contact=transition_contact.install

def phase(s,t):
 if t<s.get('handoff_time',3.):return 'source_prefix'
 if not s['enabled']:return 'no_actual_contact'
 if t<s['onset']:return 'pre_contact'
 if t>=s['release']:return 'released_recovery'
 if 'rotation_onset' in s:
  if t<s['rotation_onset']:return 'source_translation'
  if t<s['rotation_onset']+s['rotation_ramp']:return 'rotation_entry'
  if t<s['rotation_return']:return 'rotation_hold'
  if t<s['rotation_return']+s['rotation_return_ramp']:return 'rotation_withdrawal'
  return 'translation_after_rotation'
 if 'stop_time' in s:
  if t<s['stop_time']:return 'moving'
  if t<s['stop_time']+s['deceleration']:return 'decelerating'
  return 'contact_hold'
 return 'moving' if t<s['onset']+s['ramp'] else 'contact_hold'

class Task(Base):
 def __init__(self,*a):
  super().__init__(*a);self.source_nets={};self.source_physics_total=0
 def source_net(self,name):
  from clean_policy import Expert
  from geometry_policy import expand
  if name not in self.source_nets:
   ident=PLAN['sources'][name];assert hashlib.sha256(Path(ident['path']).read_bytes()).hexdigest()==ident['sha256']
   ck=torch.load(ident['path'],map_location=self.e.standing.device,weights_only=True)
   net=Expert(1088,'direct').to(self.e.standing.device);expand(net,ck);self.source_nets[name]=net.eval().requires_grad_(False)
  return self.source_nets[name]
 def sample_spec(self):
  keys=list(PLAN['curriculum_weights']);key=str(self.rng.choice(keys,p=[PLAN['curriculum_weights'][k] for k in keys]));s=copy.deepcopy(PLAN['dev'][key])
  dt=float(self.rng.uniform(-.12,.12));s['onset']+=dt;s['ramp']+=float(self.rng.uniform(-.2,.2));s['distance']*=float(self.rng.uniform(.92,1.08));s['rotation_deg']*=float(self.rng.uniform(.9,1.1));s['source_until']+=float(self.rng.uniform(-.2,.2))
  for k in ['stop_time','rotation_onset','rotation_return','release','end']:
   if k in s:s[k]+=dt
  if os.environ.get('TRANSITION_ARM')=='A':s['source_until']=3.
  s['total_cap']=float(self.rng.choice(PLAN['force_range']['training_total_caps']))
  return s
 def reset(self,spec=None,**kw):
  # Base reset creates standing prefix; source is then run in SAME env and clock.
  s=copy.deepcopy(spec if spec is not None else self.sample_spec());self.prefix_data=None
  obs=super().reset(s,**kw);source=s.get('source','standing');until=s.get('source_until',3.)
  with torch.no_grad():
   while float(self.e.tether.time)<until-1e-7:
    raw=self.e.standing/16 if source=='standing' else self.source_net(source).distribution(self.features()).mean
    _,_,done,_=super().step(raw,prefix=True)
    if done:raise RuntimeError('Source prefix failed; not silently replaced with standing: '+str(self.boundary))
  # Preserve physics, BFM env.obs/history, previous_action, last_command, lastq,
  # contact_history, h, prepared flag, tick and physical solver warmstart untouched.
  self.spec['handoff_time']=float(self.e.tether.time)
  audit=dict(time=float(self.e.tether.time),source=source,source_identity=PLAN['sources'].get(source),qpos=self.e.sim.data.qpos.tolist(),qvel=self.e.sim.data.qvel.tolist(),last_z=self.last_command.cpu().tolist(),last_qpd=self.lastq.cpu().tolist(),previous_action=self.e.previous_action.cpu().tolist(),bfm_obs={k:v.cpu().tolist() for k,v in self.e.obs.items()},input=self.features().cpu().tolist(),force_history=list(self.contact_history),contact_history_length=len(self.contact_history),support=list(map(float,self.e.last_support)))
  payload=json.dumps(audit,sort_keys=True).encode();audit['sha256']=hashlib.sha256(payload).hexdigest()
  self.source_physics_total+=len(self.physics)
  self.prefix_data=dict(physics=self.physics,control=self.controls,stats=self.stats,decisions=self.decision_rows,audit=audit,initial=self.initial)
  if self.physics:self.initial=copy.deepcopy(self.physics[-1])
  self.physics=[];self.controls=[];self.stats=[];self.decision_rows=[];self.execution_stats=[];self.episode_return=0.;self.root_max=0.
  return self.features()
 def step(self,*a,**kw):
  t=float(self.e.tether.time);result=super().step(*a,**kw);result[3].update(phase=phase(self.spec,t),demand_family=self.spec.get('demand_family',self.label));return result
 def summary(self):
  r=super().summary();r['source_physics_steps']=len(self.prefix_data['physics']) if self.prefix_data else 0
  r['handoff']=self.prefix_data['audit'] if self.prefix_data else None
  return r
 def save(self,out):
  r=super().save(out)
  # Full actual source prefix preserved separately; task metrics above are suffix-only.
  if self.prefix_data:
   with gzip.open(out/(self.label+'-source-prefix.json.gz'),'wt') as f:json.dump(self.prefix_data,f)
  return r
