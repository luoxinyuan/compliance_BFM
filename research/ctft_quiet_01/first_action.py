import gzip,pickle,json
from pathlib import Path
import numpy as np
def read(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)
def a(x):return np.asarray(x['controls'][0]['action']).reshape(-1)
p=Path.cwd()
x=read(p/'paired/B0-no_force.gz')
y=read(p/'resumed/U-no_force.gz')
assert np.array_equal(x['initial_qpos'],y['initial_qpos'])
dc=a(y)-a(x)
c=x['contract']
def pd(x):return np.asarray(c['q_PD_offset']).reshape(-1)+np.clip(5*a(x),-c['clip'],c['clip'])*np.asarray(c['q_PD_scale']).reshape(-1)/5
dq=pd(y)-pd(x)
print(json.dumps(dict(first_no_force_qPD_RMS_rad=float(np.sqrt(np.mean(dq*dq))),qPD_max_rad=float(abs(dq).max()),action_RMS=float(np.sqrt(np.mean(dc*dc))),standing_z_identical=np.array_equal(x['controls'][0]['z'],y['controls'][0]['z'])),indent=2))
