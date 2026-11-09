"""New-stage physical-time returns; original archived GAE remains untouched."""
import torch


def advantages(reward, value, next_value, discount, terminated, truncated, trace_decay):
    """Bootstrap time limits from their REAL pre-reset successor, stop the trace.

    A genuine terminal neither bootstraps nor carries a trace. A real contact
    release alone sets neither mask. Arrays are [time, workers] or [time].
    """
    assert reward.shape == value.shape == next_value.shape == discount.shape
    assert terminated.shape == truncated.shape == reward.shape
    assert terminated.dtype == truncated.dtype == torch.bool
    assert not torch.any(terminated & truncated)
    result = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        bootstrap = (~terminated[t]).to(reward.dtype)
        continuation = (~(terminated[t] | truncated[t])).to(reward.dtype)
        delta = reward[t] + discount[t] * bootstrap * next_value[t] - value[t]
        carry = delta + discount[t] * trace_decay * continuation * carry
        result[t] = carry
    return result, result + value


def physical_discount(duration_seconds, gamma_100ms=.99):
    assert 0 < gamma_100ms <= 1
    return gamma_100ms ** (duration_seconds / .1)
