"""Shared small MLP and original spherical interface, independent of task labels."""
import math
import torch
from torch import nn


class AlphaStudent(nn.Module):
    def __init__(self,dim,center=None,scale=None):
        super().__init__()
        self.register_buffer('center',torch.zeros(dim) if center is None else center.clone())
        self.register_buffer('scale',torch.ones(dim) if scale is None else scale.clone())
        self.net=nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,math.log(.01/.99))
    def forward(self,x):return self.net((x-self.center)/self.scale).sigmoid()


def student_input(model,obs,force_memory,keys):
    assert set(keys)=={'state','last_action','history_actor'}
    normalized=model._normalize(obs)
    proprio=torch.cat([normalized[k].reshape(len(obs[k]),-1).float() for k in keys],-1)
    force=torch.as_tensor(force_memory,device=proprio.device,dtype=torch.float32).reshape(len(proprio),-1)/5.
    return torch.cat([proprio,force],-1)


def path_z(standing,goal,alpha):
    theta=torch.acos(((standing*goal).sum()/256.).clamp(-1,1)).detach()
    z=(torch.sin((1-alpha)*theta)*standing+torch.sin(alpha*theta)*goal)/torch.sin(theta)
    z=z*(16./z.norm(dim=-1,keepdim=True))
    return torch.where(alpha<=0,standing.expand_as(z),torch.where(alpha>=1,goal.expand_as(z),z))


def load_student(path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    net=AlphaStudent(len(checkpoint['state_dict']['center'])).to(device)
    net.load_state_dict(checkpoint['state_dict']);net.eval().requires_grad_(False)
    return net,checkpoint
