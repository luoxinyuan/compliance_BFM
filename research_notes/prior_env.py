"""Common finite demand curriculum; no policy-visible task IDs or clocks."""
import copy,json
from pathlib import Path
import numpy as np
from transition_env import Task as Base,install_frame as old_install
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105')
PLAN=json.loads(Path(__file__).with_name('prior_training.json').read_text())
def install_frame():
 old_install()
 import geometry_env,reach_contact
 geometry_env.install_contact=reach_contact.install
class Task(Base):
 def sample_spec(self):
  names=list(PLAN['weights']);name=str(self.rng.choice(names,p=list(PLAN['weights'].values())));s=copy.deepcopy(PLAN['dev'][name]);dt=float(self.rng.uniform(-.1,.1));s['onset']+=dt;s['ramp']*=float(self.rng.uniform(.95,1.05));s['distance']*=float(self.rng.uniform(.9,1.1));s['rotation_deg']*=float(self.rng.uniform(.9,1.1));s['release']=s['onset']+s['ramp']+2;s['end']=s['release']+4;s['demand_family']=name;return s
