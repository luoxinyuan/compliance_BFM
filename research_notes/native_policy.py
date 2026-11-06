"""Independent native-initialized Direct policy, full-rank native-guided Gaussian.
Actor still outputs all256 raw coordinates, with no residual radius or candidate menu.
"""
import torch
from clean_policy import Expert as Base
class Expert(Base):
 def __init__(self,n=1088,kind='direct'):
  super().__init__(n,'direct')
  self.register_buffer('raw_L',torch.eye(256)*.008)
  self.register_buffer('raw_Linv',torch.eye(256)/.008)
  with torch.no_grad():self.log_std.zero_()
 def distribution(self,x):
  mean=self.actor(self.normalized(x))@self.raw_Linv.T
  return torch.distributions.Normal(mean,self.log_std.exp())
 def to_raw(self,u):return u@self.raw_L.T
 def raw_mean(self,x):return self.actor(self.normalized(x))
 def set_covariance(self,directions=None):
  cov=torch.eye(256,device=self.raw_L.device)*.008**2
  if directions is not None:cov+=.15**2*(directions.T@directions)
  L=torch.linalg.cholesky(cov)
  self.raw_L.copy_(L);self.raw_Linv.copy_(torch.linalg.inv(L))
