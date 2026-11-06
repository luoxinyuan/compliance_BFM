"""Match the real single-robot actor arithmetic in PPO probability evaluation.

The PPU FP32 GEMM and GEMV paths differ at about 1e-6 raw mean; joint
Gaussian logp amplifies this beyond the existing 3e-4 startup assertion.
Keep that assertion, original raw sampling/logp, network and std unchanged.
Only actor probability evaluation uses the same batch-one compute path;
the critic remains batched. No CPU execution or detached actor gradients.
"""
import torch
from transfer_policy import Expert as Original

class Expert(Original):
    def distribution(self,x):
        assert x.ndim==2
        means=[self.actor(self.normalized(row[None])) for row in x]
        mean=torch.cat(means,dim=0)
        return torch.distributions.Normal(mean,self.log_std.exp())
