import copy
import numpy as np
from prefix_check import check_fork
from reward_v2 import step_reward
q=dict(active=True,wrist=[.1,0,0],anchor=[0,0,0])
ticks=[q]*4
assert np.allclose(step_reward(-.2,1,[10,0,0],ticks),[-1.45,.25,1])
assert step_reward(-.2,0,[0,0,0],ticks)==(-.2,0.,0.)
assert np.allclose(step_reward(0.,1,[0,0,0],[q,dict(q,active=False),q,dict(q,active=False)]),[-.5,0,.5])
assert step_reward(-.2,1,[0,0,0],[dict(q,active=False)]*4)==(-.2,0.,0.)
rows=[dict(force_time=i*.005,anchor=[0],anchor_velocity=[0],qpos=[1],qvel=[0],force=[0],active=True,actor_input_or_hash='a',executed_action_or_hash='b') for i in range(6)]
b=copy.deepcopy(rows[:4]);b[3]['anchor_velocity']=[.1]
keys=dict(source_sha_a='s',source_sha_b='s',rng_sha_a='r',rng_sha_b='r')
assert check_fork(rows,b,**keys)==(3,.015)
b[1]['force']=[1]
try:check_fork(rows,b,**keys)
except ValueError as exc:assert exc.args[0]==('premature branch difference',1,'force')
else:raise AssertionError
print("reward and paired-prefix fixtures passed")
