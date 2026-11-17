"""A2 immutable learner snapshot and exact fresh-process update regression.

No live Task/partner is serialized here. This qualifies learner continuation
only; complete physical training recovery remains a separate required S0 gate.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch


def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)


def digest(x):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a=v.detach().cpu();h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for a in v:visit(a)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def set_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['cpu']);torch.cuda.set_rng_state_all(x['cuda'])


def fingerprint(agent):
    def leaves(x,prefix=''):
        if isinstance(x,dict):
            result={}
            for k,v in x.items():result.update(leaves(v,prefix+'/'+str(k)))
            return result
        if isinstance(x,(list,tuple)):
            result={}
            for i,v in enumerate(x):result.update(leaves(v,prefix+'/'+str(i)))
            return result
        return {prefix:digest(x)}
    return dict(model={k:digest(v) for k,v in agent._model.state_dict().items()},
        optimizers=leaves(agent.optimizer_dict),
        gradients={k:digest(p.grad) for k,p in agent._model.named_parameters()},
        z_buffer=digest(dict(storage=agent.z_buffer._storage,idx=agent.z_buffer._idx,is_full=agent.z_buffer._is_full)),
        rng=digest(rng()),module_modes={k:m.training for k,m in agent._model.named_modules()},
        requires_grad={k:p.requires_grad for k,p in agent._model.named_parameters()},actor_enabled=agent.actor_enabled)


def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def atomic_save(agent,buffers,expert_groups,folder,step):
    folder=Path(folder);assert not folder.exists();tmp=folder.with_name(folder.name+'.pending')
    assert not tmp.exists();tmp.mkdir();begun=time.monotonic()
    state=dict(optimizers=cpu(agent.optimizer_dict),
        gradients={k:cpu(p.grad) for k,p in agent._model.named_parameters()},
        z_buffer=cpu(dict(storage=agent.z_buffer._storage,idx=agent.z_buffer._idx,is_full=agent.z_buffer._is_full)),
        rng=cpu(rng()),module_modes={k:m.training for k,m in agent._model.named_modules()},
        requires_grad={k:p.requires_grad for k,p in agent._model.named_parameters()},
        actor_enabled=agent.actor_enabled,step=step,aux_scale=float(agent._model._aux_reward_normalizer.scale),
        replay=cpu(buffers['train'].get_full_buffer()),replay_capacity=buffers['train'].capacity,
        replay_idx=buffers['train']._idx,replay_full=buffers['train']._is_full,
        expert_groups=cpu(expert_groups),configuration=agent.cfg.model_dump(),
        execution_contract=cpu(agent.execution_contract),
        aot_donated_buffer=torch._functorch.config.donated_buffer)
    expected=fingerprint(agent)
    torch.save(cpu(agent._model.state_dict()),tmp/'model.pt')
    torch.save(state,tmp/'learner.pt')
    files={p.name:dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in tmp.iterdir()}
    assert sum(v['bytes'] for v in files.values())<12*2**30,'A2 output guard: snapshot alone exceeds12GiB'
    manifest=dict(protocol='CTFT_A2_LEARNER_ONLY_V1',complete=True,step=step,files=files,
        runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        expected_initial=expected,scope='learner/replay/RNG, not a physical Task snapshot',save_s=time.monotonic()-begun)
    write(tmp/'manifest.json',manifest)
    for p in tmp.iterdir():
        with p.open('rb') as f:os.fsync(f.fileno())
    fd=os.open(tmp,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    os.rename(tmp,folder)
    fd=os.open(folder.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    write(folder.parent/'learner-latest.pending.json',dict(package=folder.name,manifest_sha256=file_sha(folder/'manifest.json')))
    os.replace(folder.parent/'learner-latest.pending.json',folder.parent/'learner-latest.json')
    # Saving and hashing must not consume any RNG or mutate model/Adam.
    assert fingerprint(agent)==expected,'Serialization changed actual learner or RNG'
    return manifest


class Audited:
    def __init__(self,underlying,name,trace):self.underlying=underlying;self.name=name;self.trace=trace
    def sample(self,n):
        x=self.underlying.sample(n);self.trace.append(dict(provider=self.name,n=n,sha256=digest(x)));return x


def one_update(agent,buffers,step):
    trace=[];audited={k:Audited(v,k,trace) for k,v in buffers.items()}
    torch.cuda.synchronize();t=time.monotonic();metrics=agent.update(audited,step);torch.cuda.synchronize()
    assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
    free,total=torch.cuda.mem_get_info()
    assert torch.cuda.max_memory_reserved()<=.88*total and free>=max(6*2**30,.08*total)
    return dict(state=fingerprint(agent),sample_trace=trace,metrics={k:digest(v) for k,v in metrics.items()},
        update_s=time.monotonic()-t,peak_reserved=torch.cuda.max_memory_reserved(),free_bytes=free)


def save_and_reference(agent,buffers,expert_groups,out,step):
    out=Path(out);manifest=atomic_save(agent,buffers,expert_groups,out/'learner-checkpoint-4',step)
    write(out/'learner-save.json',dict(bytes=sum(v['bytes'] for v in manifest['files'].values()),save_s=manifest['save_s']))
    reference=one_update(agent,buffers,step)
    write(out/'uninterrupted-update5.json',reference)
    print('CTFT_A2_LEARNER_REFERENCE_SAVED',reference['update_s'],flush=True)


def resume_and_compare(package,out):
    from agent import build
    from humanoidverse.agents.buffers.transition import DictBuffer
    package=Path(package);out=Path(out);out.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    write(out/'launch.json',dict(pid=os.getpid(),scope='new-process learner restore and one actual update',started_unix=time.time()))
    try:
        m=json.loads((package/'manifest.json').read_text());assert m['complete']
        for name,info in m['files'].items():assert file_sha(package/name)==info['sha256'],('Checkpoint bytes/hash mismatch',name)
        state=torch.load(package/'learner.pt',map_location='cpu',weights_only=False)
        assert state['aot_donated_buffer'] is False
        agent,manifest=build(aux_scale=state['aux_scale'])
        assert agent.execution_contract==state['execution_contract'],'Execution contract differs from saved learner'
        agent._model.load_state_dict(torch.load(package/'model.pt',map_location='cpu',weights_only=False),strict=True)
        assert manifest['aot_donated_buffer'] is False and agent.cfg.model_dump()==state['configuration']
        for name,s in state['optimizers'].items():getattr(agent,name).load_state_dict(s)
        parameters=dict(agent._model.named_parameters())
        for name,g in state['gradients'].items():parameters[name].grad=None if g is None else g.to(agent.device).clone()
        modules=dict(agent._model.named_modules())
        for name,training in state['module_modes'].items():modules[name].training=training
        assert {k:p.requires_grad for k,p in parameters.items()}==state['requires_grad']
        zb=state['z_buffer'];agent.z_buffer._storage.copy_(zb['storage'].to(agent.device));agent.z_buffer._idx=zb['idx'];agent.z_buffer._is_full=zb['is_full']
        agent.actor_enabled=state['actor_enabled']
        replay=DictBuffer(capacity=state['replay_capacity'],device='cpu');replay.extend(state['replay'])
        assert replay._idx==state['replay_idx'] and replay._is_full==state['replay_full']
        groups=state['expert_groups']
        class Sequences:
            def sample(self,n):
                rows=[]
                for _ in range(n//8):
                    g=groups[int(torch.randint(len(groups),()).item())];start=int(torch.randint(len(g)-7,()).item());rows.extend(g[start:start+8])
                def combine(xs):
                    if isinstance(xs[0],dict):return {k:combine([v[k] for v in xs]) for k in xs[0]}
                    return torch.cat(xs)
                x=combine(rows);return dict(observation=x['observation'],next=dict(observation=x['next']['observation']))
        buffers=dict(train=replay,expert_slicer=Sequences());set_rng(state['rng'])
        actual=fingerprint(agent)
        assert actual==m['expected_initial'],'New-process initial learner state differs'
        result=one_update(agent,buffers,state['step'])
        reference=json.loads((package.parent/'uninterrupted-update5.json').read_text())
        differences=[]
        def compare(x,y,path=''):
            if isinstance(x,dict):
                for k in sorted(set(x)|set(y)):
                    if k not in x or k not in y:differences.append(path+'/'+k)
                    else:compare(x[k],y[k],path+'/'+k)
            elif x!=y:differences.append(path)
        for key in ['state','sample_trace','metrics']:compare(reference[key],result[key],key)
        write(out/'comparison.json',dict(status='PASS' if not differences else 'FAIL',
            new_process_initial_exact=True,post_update_exact=not differences,differences=differences,
            restored_update=result,original_update_s=reference['update_s'],wall_s=time.monotonic()-begun,
            scope='not full Task/partner physical recovery'))
        assert not differences,('New-process actual update mismatch',differences[:8])
        print('CTFT_A2_RESTORE_EXACT',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),wall_s=time.monotonic()-begun));raise
    finally:write(out/'cost.json',dict(wall_s=time.monotonic()-begun,real_physics_ticks=0))
