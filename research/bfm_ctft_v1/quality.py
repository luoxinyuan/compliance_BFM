"""CTFT-only causal support guard and independent offline stance segmentation.

Thresholds are approved scientific constants, not tunable fallback parameters.
No policy, old checkpoint, contact future, or training algorithm is imported.
"""
from dataclasses import dataclass
import numpy as np

DT = .005
LOAD_WEIGHT_FRACTION = .05
NO_SUPPORT_LIMIT_S = .05
STANCE_MIN_S = .10


@dataclass
class OnlineSupport:
    weight_n: float
    no_support_ticks: int = 0
    maximum_no_support_ticks: int = 0
    ticks: int = 0

    def update(self, normal_force_n):
        load = np.asarray(normal_force_n, dtype=np.float64)
        assert load.shape == (2,) and np.isfinite(load).all()
        assert self.weight_n > 0
        loaded = load >= LOAD_WEIGHT_FRACTION * self.weight_n
        self.no_support_ticks = 0 if loaded.any() else self.no_support_ticks + 1
        self.maximum_no_support_ticks = max(self.maximum_no_support_ticks, self.no_support_ticks)
        self.ticks += 1
        # Integer time comparison avoids binary .05 edge changing the rule.
        stop = self.no_support_ticks > round(NO_SUPPORT_LIMIT_S / DT)
        return dict(loaded=loaded.tolist(), no_support_ticks=self.no_support_ticks,
                    no_support_s=self.no_support_ticks * DT, stop=bool(stop))


def stance_segments(normal_forces_n, weight_n):
    """Offline only: qualify complete runs, including their first samples."""
    load = np.asarray(normal_forces_n, dtype=np.float64)
    assert load.ndim == 2 and load.shape[1] == 2 and np.isfinite(load).all()
    loaded = load >= LOAD_WEIGHT_FRACTION * weight_n
    required = round(STANCE_MIN_S / DT)
    segments = []
    for foot in range(2):
        edges = np.diff(np.r_[False, loaded[:, foot], False].astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            segments.append(dict(foot=foot, start=int(start), end_exclusive=int(end),
                ticks=int(end-start), duration_s=float((end-start)*DT),
                qualified=bool(end-start >= required)))
    return segments


def verify_causal_guard_boundaries():
    g = OnlineSupport(100.)
    for _ in range(10):
        assert not g.update([0., 0.])['stop']
    assert g.update([0., 0.])['stop']
    # Newly landed foot immediately restores support, no confirmation delay.
    assert not g.update([5., 0.])['stop'] and g.no_support_ticks == 0
    for _ in range(10):
        assert not g.update([0., 0.])['stop']
    g = OnlineSupport(100.)
    # Alternating fresh feet every15ms still continuously supports weight.
    for t in range(120):
        assert not g.update([5., 0.] if (t//3)%2 == 0 else [0., 5.])['stop']
    # Such short steps are not qualified OFFLINE stance segments.
    loads = np.array([[5.,0.] if (t//3)%2 == 0 else [0.,5.] for t in range(120)])
    assert not any(s['qualified'] for s in stance_segments(loads,100.))
    for n,expected in [(19,False),(20,True),(21,True)]:
        seg = stance_segments(np.array([[5.,0.]]*n),100.)
        assert seg[0]['qualified'] == expected and seg[0]['start'] == 0
    return dict(boundary='stop strictly after .05s; stance qualifies at .10s',
                causal_fresh_foot_immediate=True, offline_qualification_independent=True)
