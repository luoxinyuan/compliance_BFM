"""Pure causal sensor FIFO; no simulator/teacher metadata accepted."""
import math
from collections import deque


class ForceMemory:
    def __init__(self,samples=320):
        self.samples=samples
        self.values=deque([[0.,0.,0.] for _ in range(samples)],maxlen=samples)
    def append(self,measured_force_xyz):
        if len(measured_force_xyz)!=3 or not all(math.isfinite(float(v)) for v in measured_force_xyz):
            raise ValueError('Finite measured XYZ force required')
        self.values.append([float(v) for v in measured_force_xyz])
        return list(self.values)
