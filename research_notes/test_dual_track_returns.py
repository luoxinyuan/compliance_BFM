"""Analytic terminal/truncation tests plus original-contract numerical regression."""
import torch
from dual_track_returns import advantages, physical_discount,high_reward_from_native
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
    rs=[-.1,-.2,-.3,-.4,-.5,-.6,-.7,-.8,-.9,-1.]
    grouped=high_reward_from_native(rs[:5],[.02]*5)+physical_discount(.1)*high_reward_from_native(rs[5:],[.02]*5)
    fine=sum(physical_discount(.02*i)*r for i,r in enumerate(rs))
    assert abs(grouped-fine)<1e-12
    print('PASS: legacy-equivalent genuine terminals, real-successor truncation bootstrap, no cross-reset trace, matched physical-time discount')


if __name__ == '__main__':
    main()
