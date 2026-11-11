"""F1-only validity comparison. Raw arrays remain archived; fail closed on config changes."""
import copy
import mujoco
from frl_v3_counterfactual import tensorsame
from f1_pointer_layout import null_arrays,slot,OFFSETS
NAMES=('iefc_frictionloss','efc_AR_rownnz','map_efc2iefc','iefc_D','iefc_J_rownnz','iefc_aref','iefc_J_rowsuper','efc_island','dof_island','iefc_state','iefc_JT_colind','iefc_id','iefc_JT','map_iefc2efc','iefc_J_colind','iefc_force','efc_AR_rowadr','iefc_R','iefc_J_rowadr','iLD','map_idof2dof','iefc_J','iM_colind','iefc_type','map_dof2idof')
def metadata(model,data):
 for key in ('qpos','qvel','qacc','ctrl','efc_force'):
  if getattr(data,key).size:assert slot(data,key).value==getattr(data,key).ctypes.data,('header ABI or binding mismatch',key)
 return dict(null_arrays=null_arrays(data),version=mujoco.__version__,nv=int(model.nv),nC=int(model.nC),solver=int(model.opt.solver),noslip_iterations=int(model.opt.noslip_iterations),enableflags=int(model.opt.enableflags),disableflags=int(model.opt.disableflags),jacobian=int(model.opt.jacobian),is_sparse=bool(mujoco.mj_isSparse(model)),is_dual=bool(mujoco.mj_isDual(model)),counters={k:int(getattr(data,k)) for k in ('ncon','ne','nf','nl','nefc','nisland','nidof','nJ','nA')})
def inactive(meta):
 # Qualified only for the current immutable no-island/Newton/dense model.
 assert meta['version']=='3.3.3'
 assert meta['solver']==int(mujoco.mjtSolver.mjSOL_NEWTON)
 assert meta['enableflags']==0 and meta['disableflags']==0
 assert meta['noslip_iterations']==0 and not meta['is_dual'] and not meta['is_sparse']
 assert meta['nv']==35 and meta['nC']==341
 assert meta['counters']['nisland']==meta['counters']['nidof']==meta['counters']['nA']==0
 assert set(NAMES).issubset(meta['null_arrays'])
 return meta['null_arrays']

def canonical(state,meta):
 excluded=inactive(meta);result=dict(state)
 # Preserve all original bytes on disk; the comparison view masks only proven zero-valid-length arrays.
 result['mjdata_arrays']={k:(v.reshape(-1)[:0] if k in excluded else v) for k,v in state['mjdata_arrays'].items()}
 result['mjdata_validity']=meta
 return result

def exact(a,b):
 import numpy as np
 import torch
 from collections import deque
 if torch.is_tensor(a):
  return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes()
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list,deque)):return type(a)==type(b) and len(a)==len(b) and getattr(a,'maxlen',None)==getattr(b,'maxlen',None) and all(exact(x,y) for x,y in zip(a,b))
 if isinstance(a,float):return isinstance(b,float) and a.hex()==b.hex()
 return a==b

def same(a,b,ma,mb):return exact(canonical(a,ma),canonical(b,mb))


def install_restore_guard():
 """Same original restore sequence; do not copy NumPy-owned NULL-pointer pseudo-arrays."""
 import rolling_snapshot as rs
 import numpy as np
 import torch
 def restore(self,check=True):
  t,e=self.t,self.e;d=e.sim.data
  rs.put(t,self.task);rs.put(e,self.env);rs.put(e.base,self.base);rs.put(e.sim,self.sim);rs.put(e.wrapped,self.wrapped);rs.put(e.tether,self.tether);rs.put(t.h,self.force_history)
  e.base.history_handler.history=rs.clone(self.history)
  if self.origin_cell is not None:self.origin_cell.cell_contents=rs.clone(self.origin)
  for k,v in self.logs.items():setattr(t,k,list(v))
  e.trace=list(self.trace)
  mujoco.mj_setState(self.m,d,self.integration,self.mask);mujoco.mj_forward(self.m,d)
  for key in dir(self.data):
   if key.startswith('_'):continue
   if key in OFFSETS and slot(self.data,key).value is None:
    assert slot(d,key).value is None,('NULL validity changed during restore',key)
    continue
   a=getattr(self.data,key);b=getattr(d,key)
   if isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.shape==b.shape and b.flags.writeable:b[...]=a
  mujoco.mj_setState(self.m,d,self.integration,self.mask)
  torch.cuda.set_rng_state(self.rng);torch.set_rng_state(self.cpu_rng)
  if check:assert self.digest()==self.fingerprint,'complete branch state mismatch'
 rs.BranchState.restore=restore
