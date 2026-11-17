"""Lossless per-generation packs, with XOR against a permanent milestone.

Same TensorStore API; bundles writes to avoid thousands of CPFS fsync calls.
No tensor precision changes; references never require a retired non-anchor.
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
from cas_checkpoint import sha,durable,fsync_dir


class PackedTensorStore:
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def raw(self,ref):
        with (self.root/ref['file']).open('rb') as f:
            f.seek(ref['offset']);packed=f.read(ref['packed_n'])
        assert len(packed)==ref['packed_n'] and sha(packed)==ref['packed_sha256']
        raw=decode(packed,ref['n'])
        if ref['base']:
            before=self.raw(ref['base']);assert len(before)==len(raw)
            raw=np.bitwise_xor(np.frombuffer(raw,np.uint8),np.frombuffer(before,np.uint8)).tobytes()
        assert sha(raw)==ref['raw_sha256']
        return raw

    def unpack(self,obj):
        if isinstance(obj,dict) and obj.get('__ctft_tensor__'):
            raw=obj['inline'] if obj['inline'] is not None else b''.join(self.raw(r) for r in obj['chunks'])
            dtype=getattr(torch,obj['dtype'])
            if not raw:return torch.empty(obj['shape'],dtype=dtype,device=obj['device'])
            return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(obj['shape']).clone().to(obj['device'])
        if isinstance(obj,dict):return {k:self.unpack(v) for k,v in obj.items()}
        if isinstance(obj,list):return [self.unpack(v) for v in obj]
        if isinstance(obj,tuple):return tuple(self.unpack(v) for v in obj)
        return obj

    def metadata(self,name):
        path=self.root/name;assert not (path/'retired.json').exists(),'Generation explicitly retired by retention policy'
        m=json.loads((path/'manifest.json').read_text());assert m['complete']
        raw=(path/'state.pkl').read_bytes();assert sha(raw)==m['state_sha256']
        return pickle.loads(raw)

    def load(self,name):return self.unpack(self.metadata(name))

    def save(self,name,obj,*,base=None,anchor=False,inject=None):
        path=self.root/name;pending=path.with_name(path.name+'.pending')
        assert not path.exists() and not pending.exists();pending.mkdir();start=time.monotonic()
        stream=(pending/'data.bin').open('xb');new_bytes=0
        # References in metadata use the final immutable generation name.
        def pack(v,previous=None):
            nonlocal new_bytes
            if torch.is_tensor(v):
                t=v.detach().cpu().contiguous();raw=t.reshape(-1).view(torch.uint8).numpy().tobytes()
                refs=[];prior=previous.get('chunks',[]) if isinstance(previous,dict) else []
                inline=raw if len(raw)<=65536 else None
                if inline is None:
                    for j,i in enumerate(range(0,len(raw),CHUNK)):
                        block=raw[i:i+CHUNK];raw_sha=sha(block)
                        old=prior[j] if j<len(prior) else None
                        if old and old['raw_sha256']==raw_sha:refs.append(old);continue
                        n,encoded=encode(block);used=None
                        if old and old['n']==len(block):
                            # Caller only supplies a permanent anchor metadata.
                            baseline=self.raw(old)
                            delta=np.bitwise_xor(np.frombuffer(block,np.uint8),np.frombuffer(baseline,np.uint8)).tobytes()
                            _,encoded_delta=encode(delta)
                            if len(encoded_delta)<len(encoded):encoded=encoded_delta;used=old
                        offset=stream.tell();stream.write(encoded);new_bytes+=len(encoded)
                        if inject=='after_first_chunk':
                            stream.flush();os.fsync(stream.fileno())
                            raise RuntimeError('INJECTED_DURING_ACTUAL_FULL_STATE_WRITE')
                        refs.append(dict(file=name+'/data.bin',offset=offset,packed_n=len(encoded),
                            packed_sha256=sha(encoded),n=n,raw_sha256=raw_sha,base=used))
                return dict(__ctft_tensor__=True,dtype=str(t.dtype).split('.')[-1],shape=tuple(t.shape),
                    device=str(v.device),chunks=refs,inline=inline)
            if isinstance(v,dict):return {k:pack(z,previous.get(k) if isinstance(previous,dict) else None) for k,z in v.items()}
            if isinstance(v,list):return [pack(z,previous[i] if isinstance(previous,list) and i<len(previous) else None) for i,z in enumerate(v)]
            if isinstance(v,tuple):return tuple(pack(z,previous[i] if isinstance(previous,tuple) and i<len(previous) else None) for i,z in enumerate(v))
            return v
        try:
            # A milestone can depend on older permanent milestones. Its
            # complete logical state stays independently addressable forever.
            packed=pack(obj,base);stream.flush();os.fsync(stream.fileno())
        finally:stream.close()
        if inject=='after_blobs':raise RuntimeError('INJECTED_AFTER_BLOBS_BEFORE_COMMIT')
        content=pickle.dumps(packed,protocol=5);durable(pending/'state.pkl',content)
        m=dict(protocol='CTFT_PACKED_LOSSLESS_V1',complete=True,anchor=anchor,state_sha256=sha(content),
            newly_written_blob_bytes=new_bytes,save_s=time.monotonic()-start)
        durable(pending/'manifest.json',(json.dumps(m,indent=2)+'\n').encode())
        if inject=='before_publish':raise RuntimeError('INJECTED_BEFORE_ATOMIC_PUBLISH')
        fsync_dir(pending);os.rename(pending,path);fsync_dir(self.root)
        from learning_checkpoint import digest
        assert digest(self.load(name))==digest(obj),'Packed readback differs'
        durable(path/'readback.json',json.dumps(dict(complete=True,state_sha256=m['state_sha256'],all_tensor_raw_hashes_verified=True)).encode())
        durable(self.root/'latest.json',json.dumps(dict(package=name,manifest_sha256=sha((path/'manifest.json').read_bytes()))).encode())
        return packed,dict(m,save_verify_s=time.monotonic()-start)

    def retain(self,completed_names,permanent_names):
        """Only this store's successful nonanchors, after two verified commits.

        No stage-wide cleanup, no failed/pending package or shared asset.
        Caller supplies prelocked milestone names; all file dependencies of
        retained logical states are protected even if an anchor list is wrong.
        """
        assert len(completed_names)>=2
        latest=json.loads((self.root/'latest.json').read_text())['package']
        assert latest==completed_names[-1]
        keep=set(permanent_names)|set(completed_names[-2:])
        for n in completed_names[-2:]:
            m=json.loads((self.root/n/'manifest.json').read_text())
            v=json.loads((self.root/n/'readback.json').read_text())
            assert v['complete'] and v['state_sha256']==m['state_sha256'] and v['all_tensor_raw_hashes_verified']
        def dependencies(v):
            if isinstance(v,dict):
                if 'file' in v and 'raw_sha256' in v:keep.add(Path(v['file']).parts[0])
                for x in v.values():dependencies(x)
            elif isinstance(v,(list,tuple)):
                for x in v:dependencies(x)
        for n in list(keep):
            if n in completed_names:dependencies(self.metadata(n))
        retired=[]
        for n in completed_names:
            if n in keep:continue
            p=self.root/n;assert p.parent==self.root and n.startswith('round-')
            if (p/'retired.json').exists():continue
            m=json.loads((p/'manifest.json').read_text());assert m['complete'] and not m['anchor']
            record=dict(policy='latesttwo-plus-permanent-and-all-dependencies',manifest_sha256=sha((p/'manifest.json').read_bytes()),
                state_sha256=m['state_sha256'],data_bytes=(p/'data.bin').stat().st_size,state_bytes=(p/'state.pkl').stat().st_size,
                retired_unix=time.time(),not_a_failed_or_pending_package=True)
            durable(p/'retired.json',json.dumps(record,indent=2).encode())
            (p/'data.bin').unlink();(p/'state.pkl').unlink();fsync_dir(p);retired.append(n)
        return retired
