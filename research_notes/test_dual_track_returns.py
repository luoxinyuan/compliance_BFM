"""Analytic terminal/truncation tests plus original-contract numerical regression."""
import torch
from dual_track_returns import advantages, physical_discount
from humanoidverse.force_compliance.ppo import advantages as legacy


def main():
    torch.manual_seed(5604)
    r, v, nv = (torch.randn(12, 2) for _ in range(3))
    d = torch.full_like(r, .99)
    term = torch.zeros_like(r, dtype=torch.bool); term[4, 0] = True; term[-1] = True
    trunc = torch.zeros_like(term)
    old = legacy(r, v, nv, d, term, .95)
    new = advantages(r, v, nv, d, term, trunc, .95)
    assert all(torch.equal(a, b) for a, b in zip(old, new))
    r = torch.tensor([1., 900.]); v = torch.tensor([2., 0.]); nv = torch.tensor([10., 0.]); d = torch.tensor([.9, .9])
    term = torch.tensor([False, True]); trunc = torch.tensor([True, False])
    a, ret = advantages(r, v, nv, d, term, trunc, .95)
    assert torch.equal(a, torch.tensor([8., 900.]))
    assert torch.equal(ret, torch.tensor([10., 900.]))
    term[0] = True; trunc[0] = False
    a, _ = advantages(r, v, nv, d, term, trunc, .95)
    assert a[0] == -1
    assert abs(physical_discount(.02)**5 - physical_discount(.1)) < 1e-14
    print('PASS: legacy-equivalent genuine terminals, real-successor truncation bootstrap, no cross-reset trace, matched physical-time discount')


if __name__ == '__main__':
    main()
