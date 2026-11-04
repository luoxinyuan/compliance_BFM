"""Two learned motor interfaces. No force/contact-to-motion rules live here."""
import os
import torch

def direct_command(raw,standing):
    norm=raw.norm(dim=-1,keepdim=True)
    return torch.where(norm>1e-6,16*raw/norm.clamp_min(1e-6),standing.expand_as(raw))

def structured_command(model,standing,reference,raw,alpha,phase,dt=.02):
    from m9_cyclic import sample
    from locomotion_rollout import spherical
    target=float(raw[0,0].clamp(0,1));rate=float(raw[0,1].clamp(0,1.5))
    # Command slew is the already-qualified motor transition bound; it reads no contact or task state.
    alpha=max(alpha-.9*dt,min(alpha+.6*dt,target))
    phase+=dt*rate/reference['period'] # policy owns rate even when participation is zero
    base=spherical(standing,sample(reference,phase),alpha)
    d=16*raw[:,2:];d=d-(d*base).sum(-1,keepdim=True)*base/256
    norm=d.norm(dim=-1,keepdim=True);d=d*(16/norm.clamp_min(1e-8)).clamp(max=1)
    return model.project_z(base+d),base,alpha,phase,rate,norm

class Expert(torch.nn.Module):
    def __init__(self,n,kind=None):
        super().__init__();self.kind=kind or os.environ.get('CLEAN_KIND','structured');d=258 if self.kind=='structured' else 256
        self.register_buffer('center',torch.zeros(n));self.register_buffer('scale',torch.ones(n))
        def mlp(out):return torch.nn.Sequential(torch.nn.Linear(n,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,out))
        self.actor=mlp(d);self.critic=mlp(1)
        torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)
        std=torch.full((d,),.008)
        if self.kind=='structured':std[:2]=torch.tensor([.10,.15])
        self.log_std=torch.nn.Parameter(std.log(),requires_grad=False)
    def normalized(self,x):return ((x-self.center)/self.scale).clamp(-10,10)
    def distribution(self,x):return torch.distributions.Normal(self.actor(self.normalized(x)),self.log_std.exp())
    def value(self,x):return 10*self.critic(self.normalized(x)).squeeze(-1)

# Same verified raw-coordinate PPO/GAE update, separate actor/critic and analytic joint KL.

# Transactional candidate update: an early-stop AFTER an oversized Adam step is
# not a KL bound. Restore both parameters and Adam moments before smaller retry.
def update(net,opt,batch,epochs=6,minibatch=256,actor_lr=None,replay=None,bc_weight=0.):
    import copy
    x=batch['obs'];raw=batch['action'];old=batch['logp'];adv=batch['advantages'];ret=batch['returns'];ov=batch['value']
    with torch.no_grad():
        d0=net.distribution(x);om=d0.mean.clone();oscale=d0.scale.clone();logerr=float((d0.log_prob(raw).sum(-1)-old).abs().max());assert logerr<3e-4
    opt.param_groups[0].setdefault('nominal_actor_lr',opt.param_groups[0]['lr'])
    opt.param_groups[0]['lr']=opt.param_groups[0]['nominal_actor_lr']
    logs=[];gradient_components=[];gradients=[];rejections=[];steps=0;stop=False;frozen_std=net.log_std.detach().clone()
    with torch.enable_grad():
        for epoch in range(epochs):
            for ix in torch.randperm(len(x),device=x.device).split(minibatch):
                replay_indices=replay.sample(128) if replay is not None else None
                before={k:v.detach().clone() for k,v in net.state_dict().items()};adam_before=copy.deepcopy(opt.state_dict());accepted=False
                for retry in range(7):
                    d=net.distribution(x[ix]);ratio=(d.log_prob(raw[ix]).sum(-1)-old[ix]).exp()
                    pg=-torch.minimum(ratio*adv[ix],ratio.clamp(.85,1.15)*adv[ix]).mean();vl=.5*((net.value(x[ix])-ret[ix])/10).square().mean();bc=pg.new_zeros(())
                    if replay is not None and bc_weight:
                        bc=replay.loss(net, replay_indices)
                    if not gradient_components and replay is not None:
                        ap=tuple(net.actor.parameters())
                        g1=torch.cat([v.reshape(-1) for v in torch.autograd.grad(pg,ap,retain_graph=True)])
                        g2=torch.cat([v.reshape(-1) for v in torch.autograd.grad(bc_weight*bc,ap,retain_graph=True)])
                        gradient_components.append(dict(ppo=float(g1.norm()),weighted_retention=float(g2.norm()),ratio=float(g2.norm()/g1.norm().clamp_min(1e-12)),cosine=float(torch.nn.functional.cosine_similarity(g1,g2,dim=0))))
                    loss=pg+vl+bc_weight*bc
                    assert torch.isfinite(loss);opt.zero_grad();loss.backward()
                    an=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),.5));cn=float(torch.nn.utils.clip_grad_norm_(net.critic.parameters(),1.));opt.step()
                    with torch.no_grad():
                        d=net.distribution(x);kl=torch.distributions.kl_divergence(torch.distributions.Normal(om,oscale),d).sum(-1).mean()
                    if torch.isfinite(kl) and kl<=.02:
                        accepted=True;steps+=1;gradients.append(dict(actor=an,critic=cn,std=0.,actor_preclip=an,critic_preclip=cn,actor_clip_scale=min(1.,.5/max(an,1e-12)),actor_lr=opt.param_groups[0]['lr'],backtracks=retry));break
                    rejections.append(dict(epoch=epoch,retry=retry,joint_KL=float(kl),candidate_actor_lr=opt.param_groups[0]['lr']))
                    net.load_state_dict(before);opt.load_state_dict(copy.deepcopy(adam_before));opt.param_groups[0]['lr']=adam_before['param_groups'][0]['lr']*(.5**(retry+1))
                if not accepted or float(kl)>.015:stop=True;break
            with torch.no_grad():
                assert torch.equal(net.log_std,frozen_std)
                d=net.distribution(x);kl=torch.distributions.kl_divergence(torch.distributions.Normal(om,oscale),d).sum(-1).mean();lr=d.log_prob(raw).sum(-1)-old;ratio=lr.exp();v=net.value(x)
                logs.append(dict(epoch=epoch,joint_gaussian_KL=float(kl),sample_KL=float((ratio-1-lr).mean()),clip_fraction=float((abs(ratio-1)>.15).float().mean()),value_loss=float(((v-ret)/10).square().mean()),retention_loss=float(bc.detach()),retention_kind=replay.kind if replay else None,bc_weight=bc_weight,positive_adv_logprob_change=float(lr[adv>0].mean()) if (adv>0).any() else 0.,negative_adv_logprob_change=float(lr[adv<0].mean()) if (adv<0).any() else 0.))
                assert kl<=.02001
            if stop:break
    ev=lambda v:float(1-(ret-v).var(unbiased=False)/ret.var(unbiased=False).clamp_min(1e-8))
    return dict(optimizer_updates=steps,pre_logprob_max_error=logerr,pre_ratio_max_error=float((d0.log_prob(raw).sum(-1)-old).exp().sub(1).abs().max()),old_GAE_value_EV=ev(ov),updated_value_EV=ev(net.value(x).detach()),returns_mean=float(ret.mean()),returns_std=float(ret.std()),old_value_mean=float(ov.mean()),raw_adv_std=float(batch['raw_adv_std']),std_mean=float(net.log_std.exp().mean()),raw_mean_change_rms=float((net.distribution(x).mean-om).square().mean().sqrt()),gradient_components=gradient_components,gradients=gradients,epochs=logs,rejected_candidates=rejections,actor_lr_after=opt.param_groups[0]['lr'])
