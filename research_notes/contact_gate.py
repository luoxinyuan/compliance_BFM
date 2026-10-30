"""Causal force-history skill gate. No simulator, anchor, clock or task inputs.

Only elapsed *observed samples* are used for debounce and the existing skill
transition. This is a new, fixed teacher rule, not a learned contact policy.
"""
import math
from collections import deque


def smooth(u):
    u = min(1., max(0., u))
    return u*u*(3.-2.*u)


def inverse_smooth(alpha):
    lo, hi = 0., 1.
    for _ in range(40):
        mid = (lo+hi)/2
        if smooth(mid) < alpha: lo = mid
        else: hi = mid
    return (lo+hi)/2


class ContactGate:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.forces = deque([[0., 0., 0.]]*cfg['filter_samples'], maxlen=cfg['filter_samples'])
        self.state = 'standing'
        self.alpha = self.u = 0.
        self.enter_count = self.low_count = self.return_count = 0
        self.return_alpha = 0.
        self.entries = self.exits = 0

    def step(self, measured_force_xyz):
        if len(measured_force_xyz) != 3 or not all(math.isfinite(x) for x in measured_force_xyz):
            raise ValueError('Expected finite measured force XYZ')
        self.forces.append(list(measured_force_xyz))
        f = [sum(v[i] for v in self.forces)/len(self.forces) for i in range(3)]
        norm = math.sqrt(sum(v*v for v in f))
        c = self.cfg
        # The original task/force observation frame is world XYZ; no direction label.
        evidence = f[0] >= c['entry_forward_force_n']
        self.enter_count = self.enter_count+1 if evidence else 0
        low = norm <= c['exit_force_norm_n']
        event = None
        if self.state == 'standing' and self.enter_count >= c['entry_samples']:
            self.state = 'active'
            self.u = 0.
            self.low_count = 0
            self.entries += 1
            event = 'enter'
        elif self.state == 'returning' and self.enter_count >= c['entry_samples']:
            self.state = 'active'
            self.u = inverse_smooth(self.alpha)
            self.low_count = 0
            self.entries += 1
            event = 'reenter'
        if self.state == 'active':
            self.low_count = self.low_count+1 if low else 0
            if self.low_count >= c['exit_samples']:
                self.state = 'returning'
                self.return_alpha = self.alpha
                self.return_count = 0
                self.enter_count = 0
                self.exits += 1
                event = 'exit'
            elif not low:
                # Finite force dropout pauses progress but retains posture/history.
                self.u = min(1., self.u+c['dt']/c['entry_transition_s'])
                self.alpha = smooth(self.u)
        if self.state == 'returning':
            self.return_count += 1
            self.alpha = self.return_alpha*(1.-smooth(self.return_count*c['dt']/c['return_transition_s']))
            if self.return_count*c['dt'] >= c['return_transition_s']:
                self.state = 'standing'
                self.alpha = self.u = 0.
                self.low_count = self.enter_count = 0
                event = 'standing'
        return dict(alpha=self.alpha, state=self.state, event=event,
                    filtered_force_xyz=f, filtered_force_norm=norm,
                    enter_count=self.enter_count, low_count=self.low_count,
                    entries=self.entries, exits=self.exits)
