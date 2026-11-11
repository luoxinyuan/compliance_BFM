"""Metadata-only external-backend adapters; no implicit download or simulation."""
import torch

def pbfm_velocity_command(world_xy,current_heading,valid):
    if world_xy.shape!=(1,2) or current_heading.shape!=(1,2) or valid.shape!=(1,1): raise ValueError('shape')
    if not torch.isfinite(world_xy).all() or not torch.isfinite(current_heading).all(): raise ValueError('nonfinite')
    n=current_heading.norm(dim=-1,keepdim=True)
    if bool((n<1e-6).any()): raise ValueError('heading')
    c,s=(current_heading/n).unbind(-1);x,y=world_xy.unbind(-1)
    local=torch.stack((c*x+s*y,-s*x+c*y),-1)
    v=valid.bool()
    return torch.cat((torch.where(v,local,torch.zeros_like(local)),v.to(local.dtype)),-1)

class UnqualifiedBackend:
    def __init__(self,name,reason):self.name=name;self.reason=reason
    def command(self,*args,**kwargs):raise RuntimeError(self.name+': '+self.reason)
