"""Synthetic missing-bookkeeping regression; no physics, no CUDA initialization."""
import torch
from .restore_fields import restore_value
x={'time':4.9,'point':[.15,-.21,.84],'base':[.15,-.21,.84],'scope':'once only at onset; never tracks robot afterward'}
assert restore_value(x)==x
v=restore_value({'a':torch.tensor([1.]),'new':x},{'a':torch.zeros(1)})
assert torch.equal(v['a'],torch.tensor([1.])) and v['new']==x
try:restore_value({'missing':torch.tensor([1.])})
except AssertionError:pass
else:raise AssertionError('Missing tensor device silently inferred')
try:restore_value(torch.ones(2),torch.zeros(1))
except AssertionError:pass
else:raise AssertionError('Changed tensor schema accepted')
assert not torch.cuda.is_initialized()
print('PASS: missing scalar bookkeeping restored; missing tensor device or changed schema rejected')
