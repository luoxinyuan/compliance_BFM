import torch
from torch import nn
from torch.distributions import Normal


class ForceLatentPolicy(nn.Module):
    def __init__(self, observation_dim, latent_dim, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, latent_dim))
        self.critic = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((latent_dim,), -2.))

    def distribution(self, observation):
        return Normal(self.actor(observation), self.log_std.clamp(-5, 1).exp())

    def value(self, observation):
        return self.critic(observation).squeeze(-1)


def latent_command(model, standing, raw_action, force_norm, config, previous_z=None):
    # Tangent increments are relative to the selected base, never to an old anchor.
    # Probability ratios are evaluated on the original Gaussian sample, not z.
    if not model.cfg.archi.norm_z:
        raise ValueError('Initial experiment requires the normalized BFM latent convention')
    if config.latent_update_mode == 'accumulate':
        if previous_z is None:
            raise ValueError('Accumulating latent commands require previous_z')
        base = previous_z
    else:
        base = standing
    stand = model.project_z(base)
    residual = torch.tanh(raw_action)
    residual = residual - (residual * stand).sum(-1, keepdim=True) / stand.square().sum(-1, keepdim=True) * stand
    gate = torch.as_tensor(force_norm / config.gate_force, device=stand.device).clamp(0, 1)
    if gate.ndim == stand.ndim - 1:
        gate = gate.unsqueeze(-1)
    updated = model.project_z(stand + config.residual_scale * gate * residual)
    # Preserve the command exactly at zero force, including after nonzero updates.
    return torch.where(gate == 0, base, updated)
