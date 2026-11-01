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

class GuideFilteredFeedback(CyclicFeedback):
 """One fixed half-stride filter on current privileged wrist-error guide.
 Keeps alpha, turn gains, period and PD unchanged; rejects no observations.
 """
 def __init__(self,periods,privileged=True):
  super().__init__(periods,privileged);self.guide_filtered=np.zeros(3)
 def step(self,force_world,R,dt=.02,guide=None):
  self.guide_filtered+=dt/(self.periods['straight']/2+dt)*(np.asarray(guide)-self.guide_filtered)
  s=super().step(force_world,R,dt,guide=self.guide_filtered)
  s['guide_filtered_world']=self.guide_filtered.tolist()
  return s
