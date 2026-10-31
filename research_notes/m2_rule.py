"""Continuous initial-body context from deployable joint proprio, not task identity."""
import math

def infer_coordinate(q, references):
    if len(references) != 3 or len(q) != 7:
        raise ValueError('Three existing seven-joint reference poses required')
    best = None
    for i in range(2):
        a,b = references[i],references[i+1]
        d=[y-x for x,y in zip(a,b)];v=[y-x for x,y in zip(a,q)]
        norm=sum(x*x for x in d)
        if norm < 1e-12:raise ValueError('Degenerate reference segment')
        f=max(0.,min(1.,sum(x*y for x,y in zip(v,d))/norm))
        distance=sum((x-f*y)**2 for x,y in zip(v,d))
        candidate=(distance,(i+f)*.5)
        if best is None or candidate[0]<best[0]:best=candidate
    if not math.isfinite(best[0]):raise ValueError('Nonfinite proprioception')
    return dict(u=best[1],joint_reference_distance_rad=best[0]**.5)
