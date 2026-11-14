"""Exact checkpoint bookkeeping restore; never infer a missing tensor device."""
import copy
from collections import deque
import torch
MISSING=object()
def restore_value(value,template=MISSING,path=''):
    if torch.is_tensor(value):
        assert torch.is_tensor(template),('Missing tensor device template',path)
        assert value.dtype==template.dtype and value.shape==template.shape,('Tensor schema differs',path)
        return value.to(template.device)
    if isinstance(value,dict):
        assert template is MISSING or isinstance(template,dict),('Mapping schema differs',path)
        return {k:restore_value(v,template.get(k,MISSING) if isinstance(template,dict) else MISSING,path+'/'+str(k)) for k,v in value.items()}
    if isinstance(value,(list,tuple,deque)):
        assert template is MISSING or type(value)==type(template),('Sequence schema differs',path)
        if template is not MISSING:assert len(value)==len(template),('History length differs',path)
        vv=[restore_value(v,template[i] if template is not MISSING else MISSING,path+'/'+str(i)) for i,v in enumerate(value)]
        return deque(vv,maxlen=value.maxlen) if isinstance(value,deque) else tuple(vv) if isinstance(value,tuple) else vv
    return copy.deepcopy(value)
