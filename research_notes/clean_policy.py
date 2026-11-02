"""Two learned motor interfaces. No force/contact-to-motion rules live here."""
import os
import torch

def direct_command(raw,standing):
    norm=raw.norm(dim=-1,keepdim=True)
    return torch.where(norm>1e-6,16*raw/norm.clamp_min(1e-6),standing.expand_as(raw))

def structured_command(model,standing,reference,raw,alpha,phase,dt=.02):
    from m9_cyclic import sample
    from locomotion_rollout import spherical
    target=float(raw[0,0].clamp(0,1));rate=float(raw[0,1].clamp(0,1.5))
    # Command slew is the already-qualified motor transition bound; it reads no contact or task state.
    alpha=max(alpha-.9*dt,min(alpha+.6*dt,target))
    phase+=dt*rate/reference['period'] # policy owns rate even when participation is zero
    base=spherical(standing,sample(reference,phase),alpha)
    d=16*raw[:,2:];d=d-(d*base).sum(-1,keepdim=True)*base/256
    norm=d.norm(dim=-1,keepdim=True);d=d*(16/norm.clamp_min(1e-8)).clamp(max=1)
    return model.project_z(base+d),base,alpha,phase,rate,norm

class Expert(torch.nn.Module):
    def __init__(self,n,kind=None):
        super().__init__();self.kind=kind or os.environ.get('CLEAN_KIND','structured');d=258 if self.kind=='structured' else 256
        self.register_buffer('center',torch.zeros(n));self.register_buffer('scale',torch.ones(n))
        def mlp(out):return torch.nn.Sequential(torch.nn.Linear(n,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,out))
        self.actor=mlp(d);self.critic=mlp(1)
        torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)
        std=torch.full((d,),.008)
        if self.kind=='structured':std[:2]=torch.tensor([.10,.15])
        self.log_std=torch.nn.Parameter(std.log(),requires_grad=False)
    def normalized(self,x):return ((x-self.center)/self.scale).clamp(-10,10)
    def distribution(self,x):return torch.distributions.Normal(self.actor(self.normalized(x)),self.log_std.exp())
    def value(self,x):return 10*self.critic(self.normalized(x)).squeeze(-1)

# Same verified raw-coordinate PPO/GAE update, separate actor/critic and analytic joint KL.
from narrow_policy import update
