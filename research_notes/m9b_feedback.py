"""Single speed-coupling hypothesis; no gain search or PD smoothing."""
import numpy as np
from m9_cyclic import CyclicFeedback
class CadenceFeedback(CyclicFeedback):
 def step(self,force_world,R,dt=.02,guide=None):
  before=self.phase;s=super().step(force_world,R,dt,guide)
  # Native reference is ~0.35m/s; current desired forward speed, not future path.
  desired=max(0.,float((R.T@guide)[0]));rate=float(np.clip(desired/.35,.5,1.))
  self.phase=before+(self.phase-before)*rate
  s.update(phase=self.phase,cadence_ratio=rate,desired_forward_mps=desired)
  return s
