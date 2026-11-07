"""Stage-local full-state branches. No qpos-only reset, no differentiable physics claim."""
import copy, hashlib, json
from collections import deque
import numpy as np
import torch
import mujoco

SKIP=object()
def clone(v):
    if torch.is_tensor(v): return v.detach().clone()
    if isinstance(v,np.ndarray): return v.copy()
    if v is None or isinstance(v,(str,int,float,bool,np.generic)):return copy.deepcopy(v)
    if isinstance(v,(list,tuple,deque)):
        a=[clone(x) for x in v]
        if any(x is SKIP for x in a):return SKIP
        return deque(a,maxlen=v.maxlen) if isinstance(v,deque) else tuple(a) if isinstance(v,tuple) else a
    if isinstance(v,dict):
        a={k:clone(x) for k,x in v.items()}
        return SKIP if any(x is SKIP for x in a.values()) else a
    return SKIP

def state_of(obj,exclude=()):
    return {k:c for k,v in vars(obj).items() if k not in exclude and (c:=clone(v)) is not SKIP}
def put(obj,state):
    for k,v in state.items():
        if torch.is_tensor(v) and torch.is_tensor(getattr(obj,k,None)):
            getattr(obj,k).copy_(v)
        elif isinstance(v,dict) and isinstance(getattr(obj,k,None),dict):
            getattr(obj,k).clear();getattr(obj,k).update(clone(v))
        else:setattr(obj,k,clone(v))

class BranchState:
    def __init__(self,t):
        self.t=t;self.e=t.e;self.m=self.e.sim.model
        # These prefixes are append-only dictionaries. Keep immutable row references,
        # not a quadratic deepcopy of all previous qpos and BFM observations.
        self.logs={k:list(getattr(t,k)) for k in ['physics','controls','stats','decision_rows','execution_stats','bootstrap_rows']}
        self.task=state_of(t,exclude=tuple(self.logs)+('prefix_data','assets','native_z','refs','free','initial'))
        self.env=state_of(self.e,exclude=('trace',))
        self.trace=list(self.e.trace)
        self.base=state_of(self.e.base);self.sim=state_of(self.e.sim);self.wrapped=state_of(self.e.wrapped)
        self.tether=state_of(self.e.tether);self.history=clone(self.e.base.history_handler.history);self.force_history=state_of(t.h)
        self.mask=mujoco.mjtState.mjSTATE_INTEGRATION
        self.integration=np.empty(mujoco.mj_stateSize(self.m,self.mask));mujoco.mj_getState(self.m,self.e.sim.data,self.integration,self.mask)
        self.data=copy.copy(self.e.sim.data)
        self.rng=torch.cuda.get_rng_state();self.cpu_rng=torch.get_rng_state()
        self.fingerprint=self.digest()
    def digest(self):
        t,e=self.t,self.e;d=e.sim.data;h=hashlib.sha256()
        def add(v):
            if torch.is_tensor(v):h.update(v.detach().cpu().numpy().tobytes())
            elif isinstance(v,np.ndarray):h.update(v.tobytes())
            elif isinstance(v,dict):
                for k in sorted(v):h.update(str(k).encode());add(v[k])
            elif isinstance(v,(list,tuple,deque)):
                for x in v:add(x)
            else:h.update(str(v).encode())
        a=np.empty_like(self.integration);mujoco.mj_getState(self.m,d,a,self.mask)
        for v in [a,d.qacc,d.qacc_warmstart,d.qfrc_constraint,e.obs,e.previous_action,t.last_command,t.lastq,t.prevz,t.raw,e.history,e.base.history_handler.history,t.contact_history,t.h.rows,t.tick,t.prepared,e.tether.time,e.tether.force,e.tether.geometry_rows,e.last_support,t.features()]:add(v)
        return h.hexdigest()
    def restore(self,check=True):
        t,e=self.t,self.e;d=e.sim.data
        put(t,self.task);put(e,self.env);put(e.base,self.base);put(e.sim,self.sim);put(e.wrapped,self.wrapped);put(e.tether,self.tether);put(t.h,self.force_history)
        e.base.history_handler.history=clone(self.history)
        for k,v in self.logs.items():setattr(t,k,list(v))
        e.trace=list(self.trace)
        # Rebuild contact allocation, then restore integration AND all matching
        # writable MjData arrays incl. solver warmstart and constraint caches.
        mujoco.mj_setState(self.m,d,self.integration,self.mask);mujoco.mj_forward(self.m,d)
        for key in dir(self.data):
            if key.startswith('_'):continue
            a=getattr(self.data,key);b=getattr(d,key)
            if isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.shape==b.shape and b.flags.writeable:
                b[...]=a
        mujoco.mj_setState(self.m,d,self.integration,self.mask)
        torch.cuda.set_rng_state(self.rng);torch.set_rng_state(self.cpu_rng)
        if check:assert self.digest()==self.fingerprint,'complete branch state mismatch'
