import ast,math,json
from types import SimpleNamespace as O
from pathlib import Path
from gate_v11 import OnlineMonitor
p=Path(__file__).parent/'qualification_v11.py';module=ast.parse(p.read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='install_gate');ns={'OnlineMonitor':OnlineMonitor};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
e=O(trace=[],sim=O());t=O(e=e,spec={'demand_family':'none'},stage_boundary=None);env=O(task=t)
def original():
 if t.stage_boundary:return
 e.trace.append(dict(time=.005*(len(e.trace)+1),pelvis_matrix=[[1,0,0],[0,1,0],[0,0,1]],root=[0,0,1],support=[True,True],joint_limit_excess_max_rad=0.,stage_loaded_slip_mps=0.,active=False,force=[0,0,0]))
original._dual_track_terminal=True;e.sim.simulate_at_each_physics_step=original
m=ns['install_gate'](env);assert e.sim.simulate_at_each_physics_step._dual_track_terminal
for _ in range(5):e.sim.simulate_at_each_physics_step()
assert len(m.rows)==len(e.trace)==5
m2=ns['install_gate'](env,e.trace)
assert e.sim.simulate_at_each_physics_step._v11_underlying is original
for _ in range(5):e.sim.simulate_at_each_physics_step()
assert len(m.rows)==5 and len(m2.rows)==len(e.trace)==10
print(json.dumps(dict(status='PASS',kind='synthetic callback binding; no physics',observations=10,old_observer_not_nested=True)))
