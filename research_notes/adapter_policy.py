"""One raw-coordinate residual on frozen I; no private I rollout or extra noise."""
import torch
from clean_policy import Expert
from geometry_policy import expand
class ResidualExpert(torch.nn.Module):
 def __init__(self,ck):
  super().__init__();self.base=Expert(1088,'direct');expand(self.base,ck);self.base.eval().requires_grad_(False)
  self.actor=torch.nn.Sequential(torch.nn.Linear(1088,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,256))
  torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)
 def distribution(self,x):
  # Both paths use the same ACTUALLY executed observation and normalization.
  mean=self.base.distribution(x).mean+self.actor(self.base.normalized(x))
  return torch.distributions.Normal(mean,self.base.log_std.exp())
 def correction(self,x):return self.actor(self.base.normalized(x))
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self

def load(ck,device):
 if ck.get('architecture')=='frozen-I-adapter':
  base={ 'state_dict':{k[5:]:v for k,v in ck['state_dict'].items() if k.startswith('base.')} }
  n=ResidualExpert(base).to(device);n.load_state_dict(ck['state_dict'])
 else:
  n=Expert(1088,'direct').to(device);expand(n,ck)
 return n.eval().requires_grad_(False)
