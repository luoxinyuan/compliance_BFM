"""Original BFM executor. Only explicit caller-owned model and normalization."""
import torch


def direct_command(raw, standing):
    if raw.shape[-1]!=256 or not torch.isfinite(raw).all(): raise ValueError('raw latent')
    norm=raw.norm(dim=-1,keepdim=True)
    return torch.where(norm>1e-6,16*raw/norm.clamp_min(1e-6),standing.expand_as(raw))


def action_to_pd(action, offset, scale, clip=5.):
    # Same operation order as the archived clean_env.Task.qof. Scale already includes x5.
    return offset+(action*5).clamp(-clip,clip)*scale/5


class OriginalExecutor:
    dimensions={'state':64,'last_action':29,'history_actor':372}
    def __init__(self, model, offset, scale):
        self.model=model; self.offset=offset; self.scale=scale
        if model.training or any(p.requires_grad for p in model.parameters()): raise ValueError('requires frozen eval model')
        keys=model.cfg.archi.actor.input_filter.key
        if tuple(keys)!=tuple(self.dimensions): raise ValueError('actor filter mismatch')
    def command(self, obs, z):
        if set(obs)!=set(self.dimensions): raise ValueError('pass only permitted BFM actor fields')
        for key,n in self.dimensions.items():
            if obs[key].shape!=(1,n) or not torch.isfinite(obs[key]).all(): raise ValueError(key)
        if z.shape!=(1,256) or not torch.isfinite(z).all(): raise ValueError('z shape/value')
        # This is input validation, not a relaxation of physics/probability tolerances.
        if abs(float(z.norm())-16)>1e-4: raise ValueError('illegal z norm')
        with torch.no_grad():
            a=self.model.act(obs,z,mean=True)
            q=action_to_pd(a,self.offset,self.scale)
        return a,q
