"""Lossless tensor-aware atomic storage; actual values, no precision conversion.

Shared blobs and optional XOR against an immutable milestone reduce storage.
Replay is append-only outside the learner core. Failures are never collected.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from lossless_chunks import encode,decode,CHUNK


def sha(x):return hashlib.sha256(x).hexdigest()


def fsync_dir(p):
    fd=os.open(p,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.pending')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fsync_dir(path.parent)


class TensorStore:
    def __init__(self,root):
        self.root=Path(root);self.blobs=self.root/'blobs';self.blobs.mkdir(parents=True,exist_ok=True)
        self.raw_cache={};self.new_bytes=0

    def get(self,key):
        info=json.loads((self.blobs/(key+'.json')).read_text())
        packed=(self.blobs/(key+'.bin')).read_bytes()
        assert sha(packed)==info['packed_sha256']
        raw=decode(packed,info['n'])
        if info['base']:
            base=self.get(info['base']);assert len(base)==len(raw)
            raw=np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(base,dtype=np.uint8)).tobytes()
        assert sha(raw)==key
        return raw

    def put(self,raw,base=None):
        key=sha(raw);meta=self.blobs/(key+'.json')
        if meta.exists():return key
        n,packed=encode(raw);used=None
        if base:
            previous=self.get(base)
            if len(previous)==len(raw):
                xor=np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(previous,dtype=np.uint8)).tobytes()
                _,alternative=encode(xor)
                if len(alternative)<len(packed):packed=alternative;used=base
        durable(self.blobs/(key+'.bin'),packed)
        durable(meta,json.dumps(dict(n=n,packed_sha256=sha(packed),base=used),sort_keys=True).encode())
        self.new_bytes+=len(packed)
        return key

    def pack(self,obj,base=None):
        if torch.is_tensor(obj):
            t=obj.detach().cpu().contiguous();raw=t.reshape(-1).view(torch.uint8).numpy().tobytes()
            previous=base.get('chunks',[]) if isinstance(base,dict) and base.get('__ctft_tensor__') else []
            chunks=[self.put(raw[i:i+CHUNK],previous[j] if j<len(previous) else None)
                    for j,i in enumerate(range(0,len(raw),CHUNK))]
            return dict(__ctft_tensor__=True,dtype=str(t.dtype).split('.')[-1],shape=tuple(t.shape),chunks=chunks)
        if isinstance(obj,dict):return {k:self.pack(v,base.get(k) if isinstance(base,dict) else None) for k,v in obj.items()}
        if isinstance(obj,list):return [self.pack(v,base[i] if isinstance(base,list) and i<len(base) else None) for i,v in enumerate(obj)]
        if isinstance(obj,tuple):return tuple(self.pack(v,base[i] if isinstance(base,tuple) and i<len(base) else None) for i,v in enumerate(obj))
        return obj

    def unpack(self,obj):
        if isinstance(obj,dict) and obj.get('__ctft_tensor__'):
            raw=b''.join(self.get(k) for k in obj['chunks'])
            dtype=getattr(torch,obj['dtype'])
            if not raw:return torch.empty(obj['shape'],dtype=dtype)
            return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(obj['shape']).clone()
        if isinstance(obj,dict):return {k:self.unpack(v) for k,v in obj.items()}
        if isinstance(obj,list):return [self.unpack(v) for v in obj]
        if isinstance(obj,tuple):return tuple(self.unpack(v) for v in obj)
        return obj

    def save(self,name,obj,*,base=None,anchor=False,inject=None):
        path=self.root/name;pending=path.with_name(path.name+'.pending');assert not path.exists() and not pending.exists()
        pending.mkdir();start=time.monotonic();self.new_bytes=0
        packed=self.pack(obj,None if anchor else base)
        if inject=='after_blobs':raise RuntimeError('INJECTED_AFTER_BLOBS_BEFORE_COMMIT')
        content=pickle.dumps(packed,protocol=5);durable(pending/'state.pkl',content)
        manifest=dict(protocol='CTFT_CAS_LOSSLESS_V1',complete=True,anchor=anchor,state_sha256=sha(content),
            newly_written_blob_bytes=self.new_bytes,save_s=time.monotonic()-start)
        durable(pending/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
        if inject=='before_publish':raise RuntimeError('INJECTED_BEFORE_ATOMIC_PUBLISH')
        fsync_dir(pending);os.rename(pending,path);fsync_dir(self.root)
        # Verify every referenced tensor before latest may advertise this state.
        restored=self.load(name)
        from learning_checkpoint import digest
        assert digest(restored)==digest(obj),'Lossless readback differs'
        durable(self.root/'latest.json',json.dumps(dict(package=name,manifest_sha256=sha((path/'manifest.json').read_bytes()))).encode())
        return packed,dict(manifest,save_verify_s=time.monotonic()-start)

    def metadata(self,name):
        path=self.root/name;m=json.loads((path/'manifest.json').read_text());assert m['complete']
        raw=(path/'state.pkl').read_bytes();assert sha(raw)==m['state_sha256']
        return pickle.loads(raw)

    def load(self,name):return self.unpack(self.metadata(name))


def learner_state(agent,*,step,replay_segments,replay_capacity,replay_idx,reference):
    from learning_checkpoint import cpu,rng,fingerprint
    return dict(model=cpu(agent._model.state_dict()),optimizers=cpu(agent.optimizer_dict),
        gradients={k:cpu(p.grad) for k,p in agent._model.named_parameters()},
        z_buffer=cpu(dict(storage=agent.z_buffer._storage,idx=agent.z_buffer._idx,is_full=agent.z_buffer._is_full)),
        rng=cpu(rng()),module_modes={k:m.training for k,m in agent._model.named_modules()},
        requires_grad={k:p.requires_grad for k,p in agent._model.named_parameters()},actor_enabled=agent.actor_enabled,
        step=step,aux_scale=float(agent._model._aux_reward_normalizer.scale),configuration=agent.cfg.model_dump(),
        execution_contract=cpu(agent.execution_contract),initial_fingerprint=fingerprint(agent),
        replay_segments=replay_segments,replay_capacity=replay_capacity,replay_idx=replay_idx,reference=str(reference))


def restore_learner(state):
    from agent import build
    from learning_checkpoint import set_rng,fingerprint,file_sha
    from native_reference import load_reference
    from humanoidverse.agents.buffers.transition import DictBuffer
    agent,_=build(aux_scale=state['aux_scale'],compile_execution=state['configuration']['compile'])
    assert agent.cfg.model_dump()==state['configuration'] and agent.execution_contract==state['execution_contract']
    agent._model.load_state_dict(state['model'],strict=True)
    for name,s in state['optimizers'].items():getattr(agent,name).load_state_dict(s)
    params=dict(agent._model.named_parameters());modules=dict(agent._model.named_modules())
    for name,g in state['gradients'].items():params[name].grad=None if g is None else g.to(agent.device).clone()
    for name,training in state['module_modes'].items():modules[name].training=training
    assert {k:p.requires_grad for k,p in params.items()}==state['requires_grad']
    zb=state['z_buffer'];agent.z_buffer._storage.copy_(zb['storage'].to(agent.device));agent.z_buffer._idx=zb['idx'];agent.z_buffer._is_full=zb['is_full']
    agent.actor_enabled=state['actor_enabled']
    replay=DictBuffer(capacity=state['replay_capacity'],device='cpu')
    for segment in state['replay_segments']:
        path=Path(segment['path']);assert file_sha(path)==segment['sha256']
        replay.extend(torch.load(path,map_location='cpu',weights_only=False))
    assert replay._idx==state['replay_idx'] and not replay._is_full
    reference=load_reference(state['reference']);set_rng(state['rng'])
    assert fingerprint(agent)==state['initial_fingerprint'],'Restored learner not exact'
    return agent,dict(train=replay,expert_slicer=reference)
