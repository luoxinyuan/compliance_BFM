"""New-entry configurable guard; archived update unchanged."""
import torch
from dual_track_runtime import PPOContract
def update(net,opt,batch,epochs=6,minibatch=256,actor_lr=None, contract=None):
    contract = contract or PPOContract()
    import copy
    x=batch['obs'];raw=batch['action'];old=batch['logp'];adv=batch['advantages'];ret=batch['returns'];ov=batch['value']
    with torch.no_grad():
        d0=net.distribution(x);om=d0.mean.clone();oscale=d0.scale.clone();logerr=float((d0.log_prob(raw).sum(-1)-old).abs().max());assert logerr<contract.logprob_atol
    opt.param_groups[0].setdefault('nominal_actor_lr',opt.param_groups[0]['lr'])
    opt.param_groups[0]['lr']=opt.param_groups[0]['nominal_actor_lr']
    logs=[];gradients=[];rejections=[];steps=0;stop=False;frozen_std=net.log_std.detach().clone()
    with torch.enable_grad():
        for epoch in range(epochs):
            for ix in torch.randperm(len(x),device=x.device).split(minibatch):
                before={k:v.detach().clone() for k,v in net.state_dict().items()};adam_before=copy.deepcopy(opt.state_dict());accepted=False
                for retry in range(contract.retries):
                    d=net.distribution(x[ix]);ratio=(d.log_prob(raw[ix]).sum(-1)-old[ix]).exp()
                    pg=-torch.minimum(ratio*adv[ix],ratio.clamp(1-contract.clip,1+contract.clip)*adv[ix]).mean();vl=.5*((net.value(x[ix])-ret[ix])/10).square().mean();loss=pg+vl
                    assert torch.isfinite(loss);opt.zero_grad();loss.backward()
                    an=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),.5));cn=float(torch.nn.utils.clip_grad_norm_(net.critic.parameters(),1.));opt.step()
                    with torch.no_grad():
                        d=net.distribution(x);kl=torch.distributions.kl_divergence(torch.distributions.Normal(om,oscale),d).sum(-1).mean()
                    if torch.isfinite(kl) and kl<=contract.joint_kl_limit:
                        accepted=True;steps+=1;gradients.append(dict(actor=an,critic=cn,std=0.,actor_preclip=an,critic_preclip=cn,actor_clip_scale=min(1.,.5/max(an,1e-12)),actor_lr=opt.param_groups[0]['lr'],backtracks=retry));break
                    rejections.append(dict(epoch=epoch,retry=retry,joint_KL=float(kl),candidate_actor_lr=opt.param_groups[0]['lr']))
                    net.load_state_dict(before);opt.load_state_dict(copy.deepcopy(adam_before));opt.param_groups[0]['lr']=adam_before['param_groups'][0]['lr']*(contract.backtrack**(retry+1))
                if not accepted or float(kl)>contract.joint_kl_stop:stop=True;break
            with torch.no_grad():
                assert torch.equal(net.log_std,frozen_std)
                d=net.distribution(x);kl=torch.distributions.kl_divergence(torch.distributions.Normal(om,oscale),d).sum(-1).mean();lr=d.log_prob(raw).sum(-1)-old;ratio=lr.exp();v=net.value(x)
                logs.append(dict(epoch=epoch,joint_gaussian_KL=float(kl),sample_KL=float((ratio-1-lr).mean()),clip_fraction=float((abs(ratio-1)>contract.clip).float().mean()),value_loss=float(((v-ret)/10).square().mean()),positive_adv_logprob_change=float(lr[adv>0].mean()) if (adv>0).any() else 0.,negative_adv_logprob_change=float(lr[adv<0].mean()) if (adv<0).any() else 0.))
                assert kl<=contract.joint_kl_assert
            if stop:break
    ev=lambda v:float(1-(ret-v).var(unbiased=False)/ret.var(unbiased=False).clamp_min(1e-8))
    return dict(optimizer_updates=steps,pre_logprob_max_error=logerr,pre_ratio_max_error=float((d0.log_prob(raw).sum(-1)-old).exp().sub(1).abs().max()),old_GAE_value_EV=ev(ov),updated_value_EV=ev(net.value(x).detach()),returns_mean=float(ret.mean()),returns_std=float(ret.std()),old_value_mean=float(ov.mean()),raw_adv_std=float(batch['raw_adv_std']),std_mean=float(net.log_std.exp().mean()),raw_mean_change_rms=float((net.distribution(x).mean-om).square().mean().sqrt()),gradients=gradients,epochs=logs,rejected_candidates=rejections,actor_lr_after=opt.param_groups[0]['lr'])
