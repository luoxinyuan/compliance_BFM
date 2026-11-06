"""Predeclared near/mid/far/stop selection; terminal tests never used here."""
import numpy as np
GROUPS={'near':['local7','local9'],'mid':['mid18','mid22','ready18'],
        'far':['long65','long80','continue'],'stop':['stop','early','late']}

def selection(rows,baseline):
    from m39a_metrics import cost
    values={r['label'].split('__')[0]:cost(r) for r in rows}
    if not baseline:baseline.update(values)
    ratios={k:v/max(baseline[k],.1) for k,v in values.items()}
    groups={k:float(np.mean([ratios[n] for n in ns])) for k,ns in GROUPS.items()}
    safe=not any(r['boundary'] for r in rows)
    eligible=safe and groups['near']<=1.10
    return dict(score=float(np.mean(list(groups.values()))),groups=groups,ratios=ratios,
                worst=max(ratios.values()),eligible=eligible,no_safety_termination=safe,
                rule='Original Issue47: local mean <=1.10 initial, minimize equal near/mid/far/stop normalized mean; report all failures')
