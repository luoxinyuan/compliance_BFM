"""Same native coordinate actor; exact AR(1) conditional action likelihood.

The replay wrapper contains actual previous observations/commands, not a new
actor input or task answer. Recompute BOTH means under candidate parameters.
Holding an old residual fixed would be a different, incorrect history policy.
"""
import math
import torch
from native_policy_coordinate import Expert as CoordinateExpert


class Expert(CoordinateExpert):
    def __init__(self, n=1088, kind='direct', rho=0.):
        super().__init__(n, kind)
        self.n = n
        self.register_buffer('correlation', torch.tensor(float(rho)))

    @property
    def replay_dim(self):
        return 2*self.n + 257

    def current(self, x):
        if x.shape[-1] == self.n:
            return x
        assert x.shape[-1] == self.replay_dim
        return x[..., :self.n]

    def raw_mean(self, x):
        return super().raw_mean(self.current(x))

    def value(self, x):
        # Keep the existing critic architecture/permissions in BOTH arms.
        # It approximates V(current observation), without exact AR noise state.
        return super().value(self.current(x))

    def pack(self, current, previous, previous_u, valid):
        flag = torch.full_like(current[..., :1], float(valid))
        return torch.cat([current, previous, previous_u, flag], -1)

    def distribution(self, x):
        if x.shape[-1] == self.n:
            return super().distribution(x)  # stationary marginal / eval mean
        assert x.shape[-1] == self.replay_dim
        cur = x[..., :self.n]
        previous = x[..., self.n:2*self.n]
        previous_u = x[..., 2*self.n:2*self.n+256]
        valid = x[..., -1:]
        d = super().distribution(cur)
        prev_mean = super().distribution(previous).mean
        r = self.correlation * valid
        mean = d.mean + r*(previous_u-prev_mean)
        # At episode/policy-version starts r=0: stationary covariance I.
        # Thereafter innovation covariance (1-rho²)I preserves marginal I.
        return torch.distributions.Normal(mean, d.stddev*torch.sqrt(1-r.square()))


def rho_for(arm):
    assert arm in ['independent', 'continuous']
    return 0. if arm == 'independent' else math.exp(-.1/.5)
