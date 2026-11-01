"""Task-specific native-height feedback expert; no anchor, clock or joint correction.
The integration state is teacher memory; deployable Student can maintain history.
This is a new expert algorithm, not an engineering fix to the wrist expert.
"""
import numpy as np
class BodyFeedback:
 def __init__(self,cfg):self.cfg=cfg;self.alpha=0.;self.low=0
 def step(self,force):
  c=self.cfg;f=np.asarray(force);self.low=self.low+1 if np.linalg.norm(f)<c['disconnect_force_n'] else 0
  if self.low>=c['disconnect_samples']:rate=-c['return_rate']
  else:rate=float(np.clip(-c['force_integral_gain']*f[2],-c['progress_rate'],c['progress_rate']))
  self.alpha=float(np.clip(self.alpha+c['dt']*rate,0,1));return self.alpha
