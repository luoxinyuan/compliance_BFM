"""Finite feedback demand distribution; only native source prefix uses a motion clock."""
import json,copy
from pathlib import Path
import numpy as np
from transition_env import Task as Base,install_frame as old_install
PLAN=json.loads(Path(__file__).with_name('coexist_protocol.json').read_text())
def install_frame():
 old_install()
 import geometry_env,reach_contact
 geometry_env.install_contact=reach_contact.install
class Task(Base):
 def sample_spec(self):
  names=list(PLAN['weights']);name=str(self.rng.choice(names,p=list(PLAN['weights'].values())));s=copy.deepcopy(PLAN['dev'][name]);dt=float(self.rng.uniform(-.08,.08));s['onset']+=dt;s['ramp']*=float(self.rng.uniform(.95,1.05));s['distance']*=float(self.rng.uniform(.95,1.05))
  for k in ['stop_time','release','end']:
   if k in s:s[k]+=dt
  s['demand_family']=name;return s
 def source_net(self,name):
  if name!='armready':return super().source_net(name)
  import torch
  from types import SimpleNamespace
  from locomotion_rollout import spherical
  import reach_contact
  if not hasattr(self,'ready_goal'):
   p=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105/qualify-02/native-sources.json');self.ready_goal=torch.tensor(json.loads(p.read_text())['goals']['0.05'],device=self.e.standing.device)
  def distribution(_):
   u,_=reach_contact.smooth(float(self.e.tether.time),4.,1.5);z=spherical(self.e.standing,self.ready_goal,self.spec['ready_progress']*u);return SimpleNamespace(mean=z/16)
  return SimpleNamespace(distribution=distribution)
