"""Post-step housekeeping only; qualify exact future execution before use.

All five optimizers have consumed the gradients. Every subsequent native
update zeros these buffers before backward. We do not delete any loss branch,
change Adam moments, clip, steps, parameters, RNG or effective gradient.
"""
from learning_checkpoint import fingerprint,digest


def clear_consumed(agent):
    before=fingerprint(agent)
    summaries={name:dict(sha256=digest(p.grad),elements=0 if p.grad is None else p.grad.numel())
               for name,p in agent._model.named_parameters()}
    count=sum(p.grad is not None for p in agent._model.parameters())
    for name in agent.optimizer_names:getattr(agent,name).zero_grad(set_to_none=True)
    after=fingerprint(agent)
    assert all(before[k]==after[k] for k in before if k!='gradients')
    assert all(p.grad is None for p in agent._model.parameters())
    return dict(protocol='POST_COMPLETE_NATIVE_STEP_ZERO_CONSUMED_GRADS_V1',
        actual_grad_buffers_cleared=count,consumed_gradient_summaries=summaries,
        model_Adam_RNG_modes_targets_unchanged=True,
        no_effective_gradient_branch_removed=True,strict_next_update_equivalence='REQUIRES_PHYSICAL_QUALIFICATION')
