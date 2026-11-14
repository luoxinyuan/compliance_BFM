"""Bitwise lossless recursive checkpoint deltas (trusted private artifacts only).
XOR bytes are representation deltas, never FP32 arithmetic/quantization.
"""
import pickle,zlib,hashlib,os
from pathlib import Path
from collections import deque,OrderedDict
import numpy as np
import torch

def h(b):return hashlib.sha256(b).hexdigest()
def raw(x):
 if torch.is_tensor(x):
  a=x.detach().cpu().contiguous();return ('torch',str(a.dtype).split('.')[-1],tuple(a.shape)),a.reshape(-1).view(torch.uint8).numpy().tobytes()
 if isinstance(x,np.ndarray):
  assert not x.dtype.hasobject
  a=np.ascontiguousarray(x);return ('numpy',a.dtype.str,a.shape),a.tobytes()
 return None

def encode(x,old=None):
 a=raw(x)
 if a is not None:
  meta,b=a;prior=raw(old);kind='full'
  if prior is not None and prior[0]==meta:
   if prior[1]==b:return ('array-same',meta,h(b))
   b2=np.bitwise_xor(np.frombuffer(b,dtype=np.uint8),np.frombuffer(prior[1],dtype=np.uint8)).tobytes()
   if len(zlib.compress(b2,1))<len(zlib.compress(b,1)):kind='xor';b=b2
  return ('array',meta,kind,b,h(a[1]))
 if isinstance(x,dict):
  oo=old if isinstance(old,dict) else {}
  return ('dict',isinstance(x,OrderedDict),[(k,encode(v,oo.get(k))) for k,v in x.items()])
 if isinstance(x,(tuple,list,deque)):
  oo=old if isinstance(old,(tuple,list,deque)) else []
  return ('sequence',type(x).__name__,x.maxlen if isinstance(x,deque) else None,[encode(v,oo[i] if i<len(oo) else None) for i,v in enumerate(x)])
 return ('literal',x)

def decode(e,old=None):
 kind=e[0]
 if kind=='array-same':
  meta,b=raw(old);assert meta==e[1] and h(b)==e[2]
 elif kind=='array':
  _,meta,mode,b,sha=e
  if mode=='xor':
   m,prior=raw(old);assert m==meta and len(prior)==len(b)
   b=np.bitwise_xor(np.frombuffer(b,dtype=np.uint8),np.frombuffer(prior,dtype=np.uint8)).tobytes()
  assert h(b)==sha
 else:
  if kind=='dict':
   oo=old if isinstance(old,dict) else {}
   return (OrderedDict if e[1] else dict)((k,decode(v,oo.get(k))) for k,v in e[2])
  if kind=='sequence':
   oo=old if isinstance(old,(tuple,list,deque)) else []
   v=[decode(s,oo[i] if i<len(oo) else None) for i,s in enumerate(e[3])]
   return deque(v,maxlen=e[2]) if e[1]=='deque' else tuple(v) if e[1]=='tuple' else v
  assert kind=='literal';return e[1]
 typ,dtype,shape=meta
 if typ=='numpy':return np.frombuffer(b,dtype=dtype).copy().reshape(shape)
 dt=getattr(torch,dtype)
 return torch.frombuffer(bytearray(b),dtype=dt).clone().reshape(shape) if b else torch.empty(shape,dtype=dt)

def dump(path,state,old=None):
 path=Path(path);assert not path.exists()
 encoded=encode(state,old);payload=zlib.compress(pickle.dumps(encoded,protocol=5),1)
 with path.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
 return dict(sha256=h(payload),bytes=len(payload)),decode(encoded,old)

def load(path,meta,old=None):
 b=Path(path).read_bytes();assert h(b)==meta['sha256'] and len(b)==meta['bytes']
 return decode(pickle.loads(zlib.decompress(b)),old)
