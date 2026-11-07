"""Same full256D MLP function class; align mean-head trainable coordinates with covariance.
W_eff=W_native+(L/.008)*W_delta; all hidden layers still train. No second actor,
old high-level network, skill gate or residual radius. Delta head starts exactly0.
"""
import torch
import torch.nn.functional as F
from native_policy import Expert as RawExpert
class Expert(RawExpert):
 def __init__(self,n=1088,kind='direct'):
  super().__init__(n,'direct')
  self.register_buffer('base_weight',torch.zeros_like(self.actor[-1].weight))
  self.register_buffer('base_bias',torch.zeros_like(self.actor[-1].bias))
 def initialize_from_raw(self,state):
  # This source is native_policy.Expert (whitened probability coordinates),
  # not clean_policy.Expert (raw probability coordinates). Equal shapes do
  # not make their log_std or covariance contracts interchangeable.
  required={'raw_L','raw_Linv','log_std'}
  missing=required-set(state)
  if missing:
   raise ValueError('Expected a native whitened-policy state with explicit covariance buffers; '
                    'legacy raw-Gaussian state requires explicit coordinate conversion. '
                    f'Missing keys: {sorted(missing)}')
  self.load_state_dict(state,strict=False)
  with torch.no_grad():
   self.base_weight.copy_(state['actor.4.weight']);self.base_bias.copy_(state['actor.4.bias']);self.actor[-1].weight.zero_();self.actor[-1].bias.zero_()
 def parts(self,x):
  h=self.actor[:-1](self.normalized(x));return F.linear(h,self.base_weight,self.base_bias),self.actor[-1](h)
 def raw_mean(self,x):
  base,delta=self.parts(x);return base+delta@(self.raw_L/.008).T
 def distribution(self,x):
  base,delta=self.parts(x)
  return torch.distributions.Normal(base@self.raw_Linv.T+delta/.008,self.log_std.exp())
