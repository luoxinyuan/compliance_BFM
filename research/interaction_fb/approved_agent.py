"""Yi-approved R2 v1.0 fresh initialization. No historical Guidance is loaded.
This module initially implements qualification only; learning is explicitly gated.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .execution_agent import ExecutionAgent
from .agent import mlp
from .schema import deploy, privileged

class NonPositive(nn.Module):
    def forward(self,x):
        return -F.softplus(x)

class ApprovedAgent(ExecutionAgent):
    def __init__(self,base,obs,goal,scale,arm,lock):
        assert arm in ('E','I')
        self.base=base;self.goal=goal.clone();self.scale=scale.clone();self.arm=arm
        self.lock=copy.deepcopy(lock)
        base._actor.requires_grad_(False)
        assert len(base._actor.policy)==7
        self.tail=copy.deepcopy(base._actor.policy[5:]).requires_grad_(True)
        self.target_tail=copy.deepcopy(self.tail).requires_grad_(False)
        ds=deploy(obs).shape[-1];dp=privileged(obs).shape[-1]
        # Shared draw order and seed; arm-specific draws cannot change global sampling RNG.
        with torch.random.fork_rng(devices=[goal.device.index or 0]):
            torch.manual_seed(lock['initialization_seed'])
            common=nn.ModuleDict(dict(u=mlp(ds+256,256,True),goal_delta=mlp(32,256,True),
                q1=nn.Sequential(mlp(dp+29+256+32,1),NonPositive()),
                q2=nn.Sequential(mlp(dp+29+256+32,1),NonPositive()),goal_encoder=mlp(16,32)))
            if arm=='I':
                common['bi']=copy.deepcopy(common['goal_encoder'])
                del common['goal_encoder']
                common['fi']=mlp(dp+29+256+32,32)
        self.nets=common
        self.targets=copy.deepcopy(self.nets).requires_grad_(False)
        self.opts={k:torch.optim.Adam(v.parameters(),lr=lock['optimizer_lr']) for k,v in self.nets.items()}
        self.opts['actor_tail']=torch.optim.Adam(self.tail.parameters(),lr=lock['optimizer_lr'])
        self.prompt=torch.zeros(1,16,device=goal.device)
        self.exploration_std=.01;self.exploration_clip=.06;self.envelope=None;self.last_action_audit=None
        self.updates=0;self.z_buffer=[];self.audit=[];self.retention_lambda=0.
        self.u_retention_lambda=0.;self.sf_coefficient=None
    def state(self):
        s=super().state();s.update(version='R2-approved-v1',u_retention_lambda=self.u_retention_lambda,
            sf_coefficient=self.sf_coefficient)
        return s
    def update(self,*args,**kwargs):
        raise RuntimeError('APPROVED_QUALIFICATION_ONLY: complete scientific lock and Q gates required')
