"""Only the finite current-feedback demand curriculum changes; original native Task retained."""
import json,copy
from pathlib import Path
from native_env import Task as Base, install
PLAN=json.loads(Path(__file__).with_name('p4000_protocol.json').read_text())
class Task(Base):
 def sample_spec(self):
  from native_env import spec
  names=list(PLAN['weights']);k=str(self.rng.choice(names,p=list(PLAN['weights'].values())));a=PLAN['training_ranges'];on=float(self.rng.uniform(*a['onset']));source='standing';until=4.
  if k=='near':d=float(self.rng.uniform(*a['near_distance']));r=float(self.rng.uniform(*a['near_ramp']))
  elif k in ['mid','arm']:
   d=float(self.rng.uniform(*a['mid_distance']));r=d/float(self.rng.uniform(*a['mid_average']))
   if k=='arm':source='goal_arm5';until=5.6;on=6.2+float(self.rng.uniform(-.1,.1))
  elif k in ['long','stop']:d=float(self.rng.uniform(*a['long_distance']));r=d/float(self.rng.uniform(*a['long_average']))
  else:d=0.;r=4.
  stop=on+r*float(self.rng.uniform(*a['stop_progress'])) if k=='stop' else None
  s=spec(d,r,on,source,until,k!='none',stop,10.)
  hold=float(self.rng.uniform(*a['hold_seconds']));s['release']=(stop+.7 if stop is not None else on+r)+hold;s['end']=s['release']+4;s['demand_family']=k
  return s
