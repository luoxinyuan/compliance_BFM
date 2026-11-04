"""Shared combination-prioritized curriculum; no new actor feature or reward."""
import copy,json
from pathlib import Path
from geometry_env import Task as Base
from m39a_env import install_frame
PLAN=json.loads(Path(__file__).with_name('m39b_protocol.json').read_text())
def phase(spec,t):
 if not spec['enabled']:return 'no_actual_contact'
 if t<spec['onset']:return 'pre_contact'
 if t>=spec['release']:return 'released_recovery'
 if 'rotation_onset' in spec:
  if t<spec['rotation_onset']:return 'translation_before_rotation'
  if t<spec['rotation_onset']+spec['rotation_ramp']:return 'rotation_entry'
  if t<spec['rotation_return']:return 'rotation_hold'
  if t<spec['rotation_return']+spec['rotation_return_ramp']:return 'rotation_withdrawal'
  return 'translation_after_rotation'
 if 'stop_time' in spec:
  if t<spec['stop_time']:return 'moving'
  if t<spec['stop_time']+spec['deceleration']:return 'decelerating'
  return 'contact_hold'
 return 'moving' if t<spec['onset']+spec['ramp'] else 'contact_hold'
class Task(Base):
 def sample_spec(self):
  names=list(PLAN['curriculum_weights']);name=str(self.rng.choice(names,p=[PLAN['curriculum_weights'][k] for k in names]));s=copy.deepcopy(PLAN['dev'][name]);s['demand_family']=name
  dt=float(self.rng.uniform(-.2,.2));s['onset']+=dt;s['ramp']+=float(self.rng.uniform(-.4,.4));s['distance']*=float(self.rng.uniform(.85,1.15));s['rotation_deg']*=float(self.rng.uniform(.7,1.1))
  if name in ('combined','sequence'):s['rotation_deg']*=float(self.rng.choice([-1,1]))
  if name=='sequence':
   s['rotation_onset']=s['onset']+float(self.rng.uniform(1.5,2.3));s['rotation_ramp']=float(self.rng.uniform(1.6,2.4));s['rotation_return']=s['rotation_onset']+s['rotation_ramp']+float(self.rng.uniform(.6,1.4));s['rotation_return_ramp']=float(self.rng.uniform(1.5,2.2))
  if 'stop_time' in s:s['stop_time']+=dt
  s['release']=s['stop_time']+2.8 if 'stop_time' in s else s['onset']+s['ramp']+2;s['end']=s['release']+5
  if name=='sequence':assert s['rotation_return']+s['rotation_return_ramp']<s['release']
  return s
 def step(self,*args,**kwargs):
  t=float(self.e.tether.time);result=super().step(*args,**kwargs);result[3].update(phase=phase(self.spec,t),demand_family=self.spec.get('demand_family',self.label));return result
