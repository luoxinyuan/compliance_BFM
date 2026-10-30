"""Privileged teacher modification: bounded integration, not a model or PD change."""
import numpy as np


def executable_bounds(base_q, joint_limits, scale, offset, clip, joint_margin=.02):
    """Intersect original mechanical and normalized-action ranges, in PD-target radians."""
    action_radius = (clip/5.-1e-4)*np.asarray(scale)
    lower = np.maximum(joint_limits[:,0]+joint_margin, offset-action_radius)-base_q
    upper = np.minimum(joint_limits[:,1]-joint_margin, offset+action_radius)-base_q
    return lower, upper


class AccumulatedCorrection:
    def __init__(self, caps, total_norm=.40, velocity=.40, norm_velocity=.80, release_s=.80):
        self.caps = np.asarray(caps, dtype=float)
        assert self.caps.shape == (7,) and np.all(self.caps > 0)
        self.total_norm, self.velocity, self.norm_velocity = total_norm, velocity, norm_velocity
        self.release_s = release_s
        assert 1.5 * max(self.caps) / release_s <= velocity + 1e-12
        assert 1.5 * total_norm / release_s <= norm_velocity + 1e-12
        self.applied = np.zeros(7)
        self.target = np.zeros(7)
        self.release_start = None
        self.release_from = np.zeros(7)
        self.was_active = False
        self.last_time = None
        self.last_proposal = {}

    def project(self, x, lower=None, upper=None):
        lo = -self.caps if lower is None else np.maximum(-self.caps, lower)
        hi = self.caps if upper is None else np.minimum(self.caps, upper)
        if np.any(lo > 0) or np.any(hi < 0):
            raise ValueError('Original BFM command outside reserved joint range')
        y = np.clip(x, lo, hi)
        return y * min(1., self.total_norm / max(np.linalg.norm(y), 1e-12))

    def propose(self, increment, active, lower, upper):
        """10Hz bounded DLS increment. Integrate from applied state, never hidden windup."""
        if not active or self.release_start is not None:
            self.last_proposal = dict(active=False)
            return
        increment = np.asarray(increment)
        assert np.isfinite(increment).all()
        assert max(abs(increment)) <= .04 + 1e-10 and np.linalg.norm(increment) <= .08 + 1e-10
        raw = self.applied + increment
        self.target = self.project(raw, lower, upper)
        self.last_proposal = dict(active=True, increment=increment.tolist(), raw_target=raw.tolist(),
                                  target=self.target.tolist(), rejected=(raw-self.target).tolist())

    def tick(self, now, active, lower, upper):
        """50Hz slew, joint envelope and anti-windup; causal cubic release on loss of contact."""
        dt = .02 if self.last_time is None else now-self.last_time
        assert 0 < dt <= .0200001
        before = self.applied.copy()
        if not active and self.was_active and self.release_start is None:
            self.release_start = now
            self.release_from = before.copy()
            self.target[:] = 0
        self.was_active = bool(active)
        if self.release_start is not None:
            u = np.clip((now-self.release_start)/self.release_s, 0., 1.)
            candidate = self.release_from * (1.-u*u*(3.-2.*u))
        elif active:
            self.target = self.project(self.target, lower, upper)
            step = np.clip(self.target-before, -self.velocity*dt, self.velocity*dt)
            step *= min(1., self.norm_velocity*dt/max(np.linalg.norm(step), 1e-12))
            candidate = before + step
        else:
            candidate = np.zeros(7)
        applied = self.project(candidate, lower, upper)
        # Abort rather than silently violate the fixed speed cap if moving BFM bounds conflict.
        change = applied-before
        if max(abs(change)) > self.velocity*dt+1e-9 or np.linalg.norm(change) > self.norm_velocity*dt+1e-9:
            raise ValueError('Joint envelope and correction velocity constraints conflict')
        projected = max(abs(candidate-applied)) > 1e-12
        if projected:
            self.target = applied.copy()
        self.applied = applied
        self.last_time = now
        return applied.copy(), dict(dt=dt, applied=applied.tolist(), target=self.target.tolist(),
              velocity=(change/dt).tolist(), projected=bool(projected),
              release_start=self.release_start,
              at_box=bool(np.any(abs(applied) >= self.caps-1e-9)),
              at_norm=bool(np.linalg.norm(applied) >= self.total_norm-1e-9))
