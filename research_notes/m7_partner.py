"""Piecewise partner trajectory; original Tether.measure/apply and force law untouched.
The controller does not receive this task description. Never re-anchor at reversal.
"""
import types
import numpy as np

def segment(spec,t):
 tr=spec['transition'];f=spec['delta_xyz'][0]
 if t < tr['restart']:
  return 0.,f,spec['onset'],spec['ramp'],bool(spec['enabled'] and not(tr['unload']<=t<tr['restart']))
 return f,tr['reverse_target']-f,tr['restart'],tr['reverse_end']-tr['restart'],bool(spec['enabled'])

def install(tether,spec,anchor):
 if hasattr(tether,'_m7_original_measure'):return
 tether._m7_original_measure=tether.measure
 def measure(self):
  base,delta,start,ramp,connected=segment(spec,self.time)
  old=(self.origin,self.delta,self.cfg.activation_time,self.cfg.ramp_duration,self.cfg.enabled)
  self.origin=np.asarray(anchor)+[base,0,0];self.delta=np.asarray([delta,0,0.]);self.cfg.activation_time=start;self.cfg.ramp_duration=ramp;self.cfg.enabled=connected
  u=float(np.clip((self.time-start)/ramp,0,1));self.partner_velocity=self.delta*6*u*(1-u)/ramp if 0<u<1 else np.zeros(3)
  try:return self._m7_original_measure()
  finally:self.origin,self.delta,self.cfg.activation_time,self.cfg.ramp_duration,self.cfg.enabled=old
 tether.measure=types.MethodType(measure,tether)

def uninstall(tether):
 if hasattr(tether,'_m7_original_measure'):
  tether.measure=tether._m7_original_measure;del tether._m7_original_measure
 if hasattr(tether,'partner_velocity'):del tether.partner_velocity
