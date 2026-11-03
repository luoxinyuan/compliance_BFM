"""M39-A partner demand distribution, unchanged physical spring and Direct input."""
import copy,json,os
from pathlib import Path
import numpy as np
from geometry_env import Task as Base
PLAN=json.loads(Path(__file__).with_name('m39a_protocol.json').read_text())

def install_frame():
 import geometry_contact as gc
 if getattr(gc.frame,'m39a',False):return
 original=gc.frame
 def frame(spec,t):
  p,v,a,w=original(spec,t)
  if 'rotation_onset' in spec:
   u,du=gc.smooth(t,spec['rotation_onset'],spec['rotation_ramp']);r,dr=gc.smooth(t,spec['rotation_return'],spec['rotation_return_ramp']);angle=np.deg2rad(spec['rotation_deg']);a=angle*u*(1-r);w=angle*(du*(1-r)-u*dr)
  return p,v,a,w
 frame.m39a=True;gc.frame=frame
class Task(Base):
 def sample_spec(self):
  names=sorted(PLAN['dev']);s=copy.deepcopy(PLAN['dev'][str(self.rng.choice(names))]);dt=float(self.rng.uniform(-.12,.12));dr=float(self.rng.uniform(-.25,.25));s['onset']+=dt;s['ramp']+=dr;s['distance']*=float(self.rng.uniform(.9,1.1));s['rotation_deg']*=float(self.rng.uniform(.85,1.15))
  if 'rotation_onset' in s or (s['distance']>0 and s['rotation_deg']):s['rotation_deg']*=float(self.rng.choice([-1,1]))
  for k in ['rotation_onset','rotation_return','stop_time']:
   if k in s:s[k]+=dt
  s['release']=(s['stop_time']+2.8 if 'stop_time' in s else s['onset']+s['ramp']+2);s['end']=s['release']+5
  return s

def install():
 install_frame()
 import clean_env
 clean_env.Task=Task
