"""Continuous admittance reference, not a skill selector or learned U.
All physical parameters are explicit; no defaults are a scientific approval.
"""
from dataclasses import dataclass
import math
from .contracts import CausalObservation,Guidance,Tier

@dataclass(frozen=True)
class AdmittanceParameters:
    mass: float
    damping: float
    speed_limit: float
    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in (self.mass,self.damping,self.speed_limit)): raise ValueError('parameters')

class ForwardAdmittance:
    def __init__(self, parameters): self.parameters=parameters
    def step(self, obs:CausalObservation, velocity:float, dt=.1):
        # force_history must already be in the declared yaw-body frame.
        p=self.parameters;f=obs.force_history[-1][0]
        v=(p.mass*velocity+dt*f)/(p.mass+dt*p.damping)
        v=max(0.,min(p.speed_limit,v))
        return Guidance(v,0.,0.,True,Tier.CS),v
