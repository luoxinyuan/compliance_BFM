# Copyright (c) Meta Platforms, Inc. and affiliates.
# Derived portions follow the original CC BY-NC 4.0 LICENSE.
"""New-runtime discriminator gradients with explicit branch accumulation order.

Retain expert, unlabeled and WGAN-GP contributions and the original six-step
agent semantics. No modifications to shared agent classes or old runtimes.
"""
import types
import torch
from torch.amp import autocast

VERSION = 'FB_DISCRIMINATOR_ORDERED_COMPONENTS_V1'


def ordered_component_gradients(losses, parameters):
    """Evaluate disjoint loss graphs sequentially; FP32 sum expert, train, GP."""
    params = tuple(parameters)
    sums = [None] * len(params)
    for loss in losses:
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            value = gradient.detach()
            if sums[index] is None:
                sums[index] = value.clone()
            else:
                sums[index].add_(value)
    return sums


def update_discriminator(self, expert_obs, expert_z, train_obs, train_z, grad_penalty):
    with autocast(device_type=self.device, dtype=self._model.amp_dtype,
                  enabled=self.cfg.model.amp):
        expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
        train_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
        expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
        train_loss = torch.nn.functional.softplus(train_logits)
        loss = (expert_loss + train_loss).mean()
        components = [expert_loss.mean(), train_loss.mean()]
        if grad_penalty is not None:
            gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
            components.append(grad_penalty * gp)
            loss = loss + grad_penalty * gp
    self.discriminator_optimizer.zero_grad(set_to_none=True)
    parameters = tuple(self._model._discriminator.parameters())
    for parameter, gradient in zip(parameters, ordered_component_gradients(components, parameters)):
        parameter.grad = gradient
    self.discriminator_optimizer.step()
    result = {'disc_loss': loss.detach(), 'disc_expert_loss': expert_loss.detach().mean(),
              'disc_train_loss': train_loss.detach().mean()}
    if grad_penalty is not None:
        result['disc_wgan_gp_loss'] = gp.detach()
    return result


def install(agent, record_gradients=True):
    assert not getattr(agent, '_ordered_discriminator_installed', False)
    agent.update_discriminator = types.MethodType(update_discriminator, agent)
    agent._ordered_discriminator_installed = True
    agent._repair_events = []
    if not record_gradients:return agent
    # Compact exact pre-step gradients for every participating optimizer.
    from diagnostic_fb_resume import tree_digest
    names = {id(p): n for n, p in agent._model.named_parameters()}
    for name in agent.optimizer_dict:
        optimizer = getattr(agent, name)
        original_step = optimizer.step
        def step(*args, _name=name, _optimizer=optimizer, _step=original_step, **kwargs):
            gradients = {names[id(p)]: None if p.grad is None else tree_digest(p.grad)
                         for group in _optimizer.param_groups for p in group['params']}
            agent._repair_events.append(dict(optimizer=_name, gradients=gradients))
            return _step(*args, **kwargs)
        optimizer.step = step
    return agent
