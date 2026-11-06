"""Original Issue47 demand distribution; no online contact-to-motion rules."""
import json,copy
from pathlib import Path
from coexist_env import Task as Base,install_frame
PLAN=json.loads(Path(__file__).with_name('c0_engagement_protocol.json').read_text())

class Task(Base):
    def sample_spec(self):
        weights=PLAN['conditional_training']['task_weights'];names=list(weights)
        key=str(self.rng.choice(names,p=list(weights.values())))
        s=copy.deepcopy(PLAN['development'][key]);dt=float(self.rng.uniform(-.08,.08))
        s['onset']+=dt;s['distance']*=float(self.rng.uniform(.95,1.05));s['ramp']*=float(self.rng.uniform(.95,1.05))
        for k in ['stop_time','release','end']:
            if k in s:s[k]+=dt
        s['demand_family']=key
        return s
