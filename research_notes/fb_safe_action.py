"""Issue58 P0-B executed-action contract, no shared model/PD modifications.

The sole extra key is execution metadata (handoff fraction), not a learned
contact feature. Actor/F/B/critics retain the approved97D neural input schema.
Every actor sample used by FB targets/actor loss is transformed identically.
"""
import types
import torch
KEY='execution_handoff_fraction_v1'
VERSION='FB_SAFE_ANKLE_EXECUTION_V1'

class SafeAction:
    def __init__(self,contract,ranges,device):
        self.offset=torch.as_tensor(contract['q_PD_offset'],device=device).reshape(1,-1)
        self.scale=torch.as_tensor(contract['q_PD_scale'],device=device).reshape(1,-1)
        self.bound=contract['clip']/contract['normalize_ratio']
        self.indices=[contract['dof_names'].index(x+'_ankle_roll_joint') for x in ('left','right')]
        # Locked engineering candidate. This is a target inset, not a relaxed
        # actual-q boundary. Independent physical validation is still required.
        self.margin=.05
        self.lower=torch.as_tensor([ranges[x][0]+self.margin for x in self.indices],device=device)
        self.upper=torch.as_tensor([ranges[x][1]-self.margin for x in self.indices],device=device)
        assert bool((self.lower<self.upper).all())
        self.last=None

    def __call__(self,proposal,original,alpha):
        raw=proposal;source=original.clamp(-self.bound,self.bound)
        mixed=(1-alpha)*source+alpha*proposal.clamp(-self.bound,self.bound)
        qpd=self.offset+mixed*self.scale
        action=mixed.clone()
        lower=(self.lower-self.offset[...,self.indices])/self.scale[...,self.indices]
        upper=(self.upper-self.offset[...,self.indices])/self.scale[...,self.indices]
        action[...,self.indices]=mixed[...,self.indices].clamp(lower,upper)
        actual_qpd=self.offset+action.clamp(-self.bound,self.bound)*self.scale
        self.last=dict(raw_proposal=raw.detach().clone(),pre_bound_action=mixed.detach().clone(),
            actual_action=action.detach().clone(),q_PD=actual_qpd.detach().clone(),
            active=(action-mixed).abs().gt(0).detach(),alpha=alpha.detach().clone())
        return action

class PhysicalDistribution:
    def __init__(self,raw,transform,source,alpha):
        self.raw=raw;self.transform=transform;self.source=source;self.alpha=alpha
    @property
    def mean(self):return self.transform(self.raw.mean,self.source,self.alpha)
    def sample(self,*a,**kw):return self.transform(self.raw.sample(*a,**kw),self.source,self.alpha)
    def log_prob(self,*a,**kw):
        raise RuntimeError('Bounded physical FB action is not a Gaussian PPO raw command')


def install(model,original,transform):
    assert not getattr(model,'_safe_action_installed',False)
    oldnorm=model._obs_normalizer.forward
    def normalized(self,obs):
        values=oldnorm(obs)
        if isinstance(obs,dict) and KEY in obs:values[KEY]=obs[KEY]
        return values
    model._obs_normalizer.forward=types.MethodType(normalized,model._obs_normalizer)
    forward=model._actor.forward
    def physical(self,obs,z,std):
        raw=forward(obs,z,std)
        alpha=obs.get(KEY,torch.ones(z.shape[0],1,device=z.device))
        with torch.no_grad():source=original._actor(obs,z,0.).mean
        return PhysicalDistribution(raw,transform,source,alpha)
    model._actor.forward=types.MethodType(physical,model._actor)
    model._safe_action_installed=True


def bind_task(task):
    old=task.extended_observation
    def extended(self):
        values=old();elapsed=max(0.,float(self.e.tether.time)-float(self.spec['source_until']))
        u=min(1.,elapsed/.35);alpha=u*u*(3-2*u)
        values[KEY]=torch.tensor([[alpha]],device=self.e.standing.device,dtype=torch.float32)
        return values
    task.extended_observation=types.MethodType(extended,task)
