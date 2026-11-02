"""Privileged PPO over the frozen M10 prior; raw Gaussian coordinates, legal sphere z."""
import torch
class Expert(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        def mlp(out):return torch.nn.Sequential(torch.nn.Linear(n,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,out))
        self.actor=mlp(256);self.critic=mlp(1)
        torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)
        self.log_std=torch.nn.Parameter(torch.full((256,),-2.5257286443)) # initial .08
    def distribution(self,x):return torch.distributions.Normal(self.actor(x),self.log_std.exp())
    def value(self,x):return 10*self.critic(x).squeeze(-1)

def legal_command(model,base,raw):
    d=torch.tanh(raw)
    d=d-(d*base).sum(-1,keepdim=True)*base/256
    norm=d.norm(dim=-1,keepdim=True)
    d=d*torch.clamp(8/norm.clamp_min(1e-8),max=1)
    z=model.project_z(base+d)
    return torch.where((raw==0).all(-1,keepdim=True),base,z),norm

def update(net,opt,batch,epochs=6,minibatch=256,actor_lr=None):
    x=batch['obs'];raw=batch['action'];old=batch['logp'];adv=batch['advantages'];ret=batch['returns'];ov=batch['value']
    with torch.no_grad():
        d0=net.distribution(x);om=d0.mean.clone();os=d0.scale.clone();logerr=float((d0.log_prob(raw).sum(-1)-old).abs().max())
        assert logerr<3e-4,logerr
    logs=[];steps=0;stop=False;gradients=[];frozen_std=net.log_std.detach().clone() if not net.log_std.requires_grad else None
    with torch.enable_grad():
        for epoch in range(epochs):
            for ix in torch.randperm(len(x),device=x.device).split(minibatch):
                d=net.distribution(x[ix]);ratio=(d.log_prob(raw[ix]).sum(-1)-old[ix]).exp()
                pg=-torch.minimum(ratio*adv[ix],ratio.clamp(.85,1.15)*adv[ix]).mean()
                vl=.5*((net.value(x[ix])-ret[ix])/10).square().mean()
                loss=pg+vl
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward()
                norm=lambda ps:float(torch.sqrt(sum((p.grad.detach().square().sum() for p in ps if p.grad is not None),torch.zeros((),device=x.device))))
                gradients.append(dict(actor=norm(net.actor.parameters()),critic=norm(net.critic.parameters()),std=norm([net.log_std])))
                total_norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),.5));gradients[-1].update(total=total_norm,clip_scale=min(1.,.5/max(total_norm,1e-12)));opt.step()
                with torch.no_grad():
                    if net.log_std.requires_grad:net.log_std.clamp_(-4.,-1.)
                    else:assert torch.equal(net.log_std,frozen_std), 'External std changed inside PPO batch'
                steps+=1
                with torch.no_grad():
                    d=net.distribution(x);kl=torch.distributions.kl_divergence(torch.distributions.Normal(om,os),d).sum(-1).mean()
                if kl>.02:stop=True;break
            with torch.no_grad():
                d=net.distribution(x);lr=d.log_prob(raw).sum(-1)-old;ratio=lr.exp();v=net.value(x)
                logs.append(dict(epoch=epoch,joint_gaussian_KL=float(kl),sample_KL=float((ratio-1-lr).mean()),clip_fraction=float((abs(ratio-1)>.15).float().mean()),value_loss=float(((v-ret)/10).square().mean())))
            if stop:break
    ev=lambda v:float(1-(ret-v).var(unbiased=False)/ret.var(unbiased=False).clamp_min(1e-8))
    return dict(optimizer_updates=steps,pre_logprob_max_error=logerr,pre_ratio_max_error=float((d0.log_prob(raw).sum(-1)-old).exp().sub(1).abs().max()),old_GAE_value_EV=ev(ov),updated_value_EV=ev(net.value(x).detach()),returns_mean=float(ret.mean()),returns_std=float(ret.std()),old_value_mean=float(ov.mean()),raw_adv_std=float(batch['raw_adv_std']),std_mean=float(net.log_std.exp().mean()),raw_mean_change_rms=float((net.distribution(x).mean-om).square().mean().sqrt()),gradients=gradients,epochs=logs)
