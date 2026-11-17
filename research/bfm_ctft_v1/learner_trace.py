"""Outer-method boundary capture; no extra network calls or random draws."""
import os
from pathlib import Path
import torch
from learning_checkpoint import cpu,digest,rng,write


def install(agent,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);saved={};rows=[]
    config=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        grad_enabled=torch.is_grad_enabled(),deterministic=torch.are_deterministic_algorithms_enabled(),
        num_threads=torch.get_num_threads(),default_dtype=str(torch.get_default_dtype()),torch_version=torch.__version__,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)
    write(folder/'execution-context.json',config)
    def meta(x):
        if torch.is_tensor(x):return dict(hash=digest(x),shape=list(x.shape),stride=list(x.stride()),
             offset=x.storage_offset(),dtype=str(x.dtype),device=str(x.device),requires_grad=x.requires_grad)
        if isinstance(x,dict):return {k:meta(v) for k,v in x.items()}
        return dict(hash=digest(x))
    names=['update_discriminator','sample_mixed_z','update_fb','update_critic','update_aux_critic','update_actor']
    for name in names:
        original=getattr(agent,name);saved[name]=original
        def observed(*a,_name=name,_fn=original,**kw):
            x=dict(method=_name,input_metadata=meta(kw),input_rng=digest(rng()))
            torch.save(cpu(kw),folder/(_name+'-inputs.pt'))
            r=_fn(*a,**kw)
            x.update(output_metadata=meta(r),output_rng=digest(rng()))
            rows.append(x);write(folder/'boundaries.json',rows)
            return r
        setattr(agent,name,observed)
    def close():
        for name,value in saved.items():setattr(agent,name,value)
    return close
