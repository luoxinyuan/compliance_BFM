"""Execution-only controls; native losses, Adam configuration and data unchanged."""
import torch


def eager_adam_boundaries(agent):
    """Pin native Adam.step outside graph-cache-dependent compiled/fallback paths.

    All five original Adam instances, parameters, states, gradients, rates and
    step order are retained. Network forward/backward compilation stays enabled.
    """
    for name in agent.optimizer_names:
        optimizer=getattr(agent,name)
        optimizer.step=torch.compiler.disable(optimizer.step)
    return dict(optimizer_execution='native_eager_step_all_five',network_compile=agent.cfg.compile)
