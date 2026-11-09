"""Separate critic fitting from the unchanged raw-Gaussian actor KL guard.

No extra physics, reward change, action override, or change to PPO probabilities.
The original actor routine is called with critic parameters temporarily frozen.
Then the independent critic completes all six epochs on fixed rollout targets.
"""
import copy
import torch
from dual_track_ppo import update as original_update


def actor_adam(opt, net):
    return [copy.deepcopy(opt.state.get(p, {})) for p in net.actor.parameters()]


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def value_metrics(net, batch):
    with torch.no_grad():
        target = batch['returns'].detach()
        value = net.value(batch['obs'])
        return dict(mse=float((target-value).square().mean()),
                    EV=float(1-(target-value).var(unbiased=False)/target.var(unbiased=False).clamp_min(1e-8)),
                    mean_residual=float((target-value).mean()))


def update(net, opt, batch, epochs=6, minibatch=256, *, critic_shuffle_seed, contract=None):
    critic_params = list(net.critic.parameters())
    flags = [p.requires_grad for p in critic_params]
    opt.zero_grad(set_to_none=True)
    try:
        for p in critic_params:
            p.requires_grad_(False)
        # Actor gradients, permutation stream, clipping, rollback, and KL guards
        # are all the original implementation. The critic cannot alter actor KL.
        logs = original_update(net, opt, batch, epochs=epochs, minibatch=minibatch, contract=contract)
    finally:
        for p, flag in zip(critic_params, flags):
            p.requires_grad_(flag)

    before_actor = {k: v.detach().clone() for k, v in net.actor.state_dict().items()}
    before_adam = actor_adam(opt, net)
    pre = value_metrics(net, batch)
    with torch.no_grad():
        before_mean = net.distribution(batch['obs']).mean.detach().clone()
    # A local stream prevents additional critic work from changing the next
    # actor minibatch permutation stream or the workers' actual sampling stream.
    generator = torch.Generator(device=batch['obs'].device).manual_seed(critic_shuffle_seed)
    gradients = []
    with torch.enable_grad():
        for _ in range(epochs):
            indices = torch.randperm(len(batch['obs']), device=batch['obs'].device, generator=generator)
            for ix in indices.split(minibatch):
                loss = .5*((net.value(batch['obs'][ix])-batch['returns'][ix].detach())/10).square().mean()
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is None for p in net.actor.parameters())
                norm = torch.nn.utils.clip_grad_norm_(critic_params, 1.)
                assert torch.isfinite(norm)
                opt.step()  # Adam skips actor params because their grad is None.
                gradients.append(float(norm))
    assert all(torch.equal(v, before_actor[k]) for k, v in net.actor.state_dict().items())
    assert equal_tree(before_adam, actor_adam(opt, net))
    with torch.no_grad():
        assert torch.equal(before_mean, net.distribution(batch['obs']).mean)
    post = value_metrics(net, batch)
    logs['actor_stage_value_EV'] = logs['updated_value_EV']
    logs['updated_value_EV'] = post['EV']
    logs['critic_fit'] = dict(optimizer_updates=len(gradients), pre=pre, post=post,
                             gradient_norms=gradients, shuffle_seed=critic_shuffle_seed,
                             actor_weights_and_adam_bitwise_unchanged=True,
                             targets='unchanged rollout GAE returns; no relabeling of raw old logprob',
                             independent_of_actor_KL=True)
    return logs
