"""Experiment-local extension: restore the installed callback graph across Task.reset.
Original rolling_snapshot.BranchState remains unchanged for historical same-reset use.
"""
from rolling_snapshot import BranchState as SameResetState,clone,state_of,put

class CrossResetState(SameResetState):
 def __init__(self,task):
  # Callbacks close over these concrete objects. Values alone cannot restore their binding.
  self.source_tether=task.e.tether;self.source_spec=task.spec
  self.source_history=task.h;self.source_handler=task.e.base.history_handler
  self.live_data=task.e.sim.data;self.apply_fn=task.e.tether.apply;self.measure_fn=task.e.tether.measure
  self.physics_fn=task.e.sim.simulate_at_each_physics_step
  self.original_apply=task.original_apply;self.original_measure=task.original_measure
  self.cfg_obj=task.e.cfg;self.tcfg_obj=task.e.tether.cfg
  self.cfg_values=state_of(self.cfg_obj);self.tcfg_values=state_of(self.tcfg_obj)
  self.source_initial=clone(task.initial);self.source_prefix=task.prefix_data
  super().__init__(task)
 def restore(self,check=True):
  t,e=self.t,self.e
  e.tether=self.source_tether;t.spec=self.source_spec;t.h=self.source_history;e.base.history_handler=self.source_handler
  e.sim.data=self.live_data;e.cfg=self.cfg_obj;e.tether.cfg=self.tcfg_obj
  put(e.cfg,self.cfg_values);put(e.tether.cfg,self.tcfg_values)
  e.tether.apply=self.apply_fn;e.tether.measure=self.measure_fn;e.sim.simulate_at_each_physics_step=self.physics_fn
  t.original_apply=self.original_apply;t.original_measure=self.original_measure
  t.initial=clone(self.source_initial);t.prefix_data=self.source_prefix
  super().restore(check=check)
  assert e.tether is self.source_tether and t.spec is self.source_spec
  assert e.tether.measure.__func__ is self.measure_fn.__func__ and e.sim.simulate_at_each_physics_step is self.physics_fn
  cells=dict(zip(e.tether.measure.__func__.__code__.co_freevars,e.tether.measure.__func__.__closure__))
  assert cells['spec'].cell_contents is t.spec
  if self.origin_cell is not None:assert cells['base'] is self.origin_cell
