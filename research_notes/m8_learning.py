"""Shared multi-contact legal latent interface and same-state action projection."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np,torch
from contact_projector import constrained,check_deadline
from m1_interface import slerp
from m3_learning import support
from m8_stage import RUN,completed

def blend(a,b,u):
 u=torch.as_tensor(u,device=a.device,dtype=a.dtype).reshape(-1,1);z=slerp(a,b,u)
 return torch.where(u==0,a.expand_as(z),torch.where(u==1,b.expand_as(z),z))
def policy_latent(model,ends,raw,gate):
 base=blend(blend(ends[0],ends[1],raw[:,0].clamp(0,1)),ends[2],raw[:,1].clamp(0,1))
 return constrained(model,base,raw[:,2:],gate)[0]
def project(model,obs,base,gate,target,qof,body=False):
 norm={k:v.detach() for k,v in model._normalize(obs).items()};actor=lambda z:model._actor(norm,z,model.cfg.actor_std).mean.float()
 initial=qof(actor(base))-target
 # Exact representable task expert actions need no numerical drift from needless optimization.
 if float(initial.abs().max())<1e-6:return base.detach(),initial.detach()
 with torch.enable_grad():
  raw=torch.zeros(len(base),256,device=base.device,requires_grad=True);opt=torch.optim.Adam([raw],lr=.05)
  for _ in range(160):
   check_deadline();opt.zero_grad();z=constrained(model,base,raw,gate)[0];e=qof(actor(z))-target
   loss=e.square().mean() if body else e[:,22:].square().mean()+.1*e[:,:22].square().mean();loss.backward();opt.step()
  z=constrained(model,base,raw,gate)[0];e=qof(actor(z))-target
 return z.detach(),e.detach()
class Projector(torch.nn.Module):
 def __init__(self,h):
  super().__init__();self.register_buffer('center',h.mean(0));self.register_buffer('scale',h.std(0,unbiased=False).clamp_min(.1));self.net=torch.nn.Sequential(torch.nn.Linear(h.shape[1],128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,258))
 def forward(self,h):return self.net((h-self.center)/self.scale)
def load_net(p,dev):
 x=torch.load(p,map_location=dev,weights_only=True);net=Projector(torch.zeros(2,len(x['state_dict']['center']),device=dev)).to(dev);net.load_state_dict(x['state_dict']);return net.eval().requires_grad_(False)
