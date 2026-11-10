"""Bounded exact update tracing. No arithmetic, optimizer or tolerance changes."""
import copy, functools, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils._pytree import tree_flatten
from diagnostic_fb_resume import tree_digest, fingerprints
from dual_track_task import global_rng,restore_global_rng
from dual_track_fb_online import cpu
from dual_track_high import write

class Fixed:
 def __init__(self,b):self.b=b;self.calls=0
 def sample(self,n):
  self.calls+=1;assert n==1024 and self.calls==1
  return self.b

def leaves(x):return [v for v in tree_flatten(x)[0] if torch.is_tensor(v)]
def signature(x):
 a=x.detach().cpu().contiguous()
 return dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),device=str(x.device),
   hash=hashlib.sha256(a.numpy().tobytes()).hexdigest())

class Trace(TorchDispatchMode):
 def __init__(self,out,expected=None):
  super().__init__();self.out=out;self.rows=[];self.phase='prepare';self.first=None
  self.expected=json.loads(expected.read_text()) if expected else None
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  kwargs=kwargs or {};result=func(*args,**kwargs)
  row=dict(index=len(self.rows),phase=self.phase,op=str(func),outputs=[signature(t) for t in leaves(result)])
  if self.expected is not None and self.first is None:
   ref=self.expected[len(self.rows)] if len(self.rows)<len(self.expected) else None
   if ref!=row:
    self.first=dict(actual=row,reference=ref,inputs=[signature(t) for t in leaves((args,kwargs))])
    torch.save(dict(inputs=cpu((args,kwargs)),outputs=cpu(result)),self.out/'first-difference-tensors.pt')
    write(self.out/'first-difference.json',self.first)
  self.rows.append(row);return result

def instrument(agent,tr,out):
 boundaries=[];handles=[]
 def checkpoint(label):
  with _disable_current_modes():capture(label)
 def capture(label):
  boundaries.append(dict(label=label,op_index=len(tr.rows),model=tree_digest(agent._model.state_dict()),
    optimizers=tree_digest(agent.optimizer_dict),gradients={n:None if p.grad is None else signature(p.grad) for n,p in agent._model.named_parameters()},rng=tree_digest(global_rng())))
 for name in ['update_discriminator','update_fb','update_critic','update_aux_critic','update_actor']:
  fn=getattr(agent,name)
  def wrap(*a,_fn=fn,_name=name,**kw):
   tr.phase=_name+'.forward';checkpoint(_name+'.begin')
   result=_fn(*a,**kw);checkpoint(_name+'.end');return result
  setattr(agent,name,wrap)
 for name in agent.optimizer_dict:
  opt=getattr(agent,name);step=opt.step;zero=opt.zero_grad
  def zw(*a,_fn=zero,_name=name,**kw):tr.phase=_name+'.backward';return _fn(*a,**kw)
  def sw(*a,_fn=step,_name=name,**kw):
   checkpoint(_name+'.pre-step');tr.phase=_name+'.step';r=_fn(*a,**kw);checkpoint(_name+'.post-step');return r
  opt.zero_grad=zw;opt.step=sw
 # Captures exact boundary of all four original target updates.
 import humanoidverse.agents.fb_cpr_aux.agent as module
 soft=module._soft_update_params
 def target(*a,**kw):
  tr.phase='target.'+str(sum(x['label'].startswith('target') for x in boundaries))
  r=soft(*a,**kw);checkpoint(tr.phase);return r
 module._soft_update_params=target
 return boundaries

def trace(agent,c,out,reference=None):
 start=time.monotonic();cache=reference/'fixed-batch.pt' if reference else out/'fixed-batch.pt'
 if reference:
  data=torch.load(cache,map_location='cpu',weights_only=False)
  assert tree_digest(global_rng())==tree_digest(data['rng_before']), 'pre-sampling RNG differs'
  # Verify actual resampling is exactly the same ordered values, not just a seed claim.
  eb=c.sequences.sample(1024);tb=c.replay.sample(1024)
  assert tree_digest(eb)==tree_digest(data['expert']) and tree_digest(tb)==tree_digest(data['train'])
  assert tree_digest(global_rng())==tree_digest(data['rng_after'])
  eb,tb=data['expert'],data['train']
 else:
  before=global_rng();eb=c.sequences.sample(1024);tb=c.replay.sample(1024)
  data=dict(expert=cpu(eb),train=cpu(tb),rng_before=before,rng_after=global_rng())
  torch.save(data,cache)
 restore_global_rng(data['rng_after'])
 write(out/'batch-identity.json',dict(expert=tree_digest(eb),train=tree_digest(tb),rng=tree_digest(global_rng()),
   expert_order=list(eb['observation']),train_order=list(tb['observation']),
   pre_model=tree_digest(agent._model.state_dict()),pre_adam=tree_digest(agent.optimizer_dict)))
 tr=Trace(out,reference/'operators.json' if reference else None)
 boundaries=instrument(agent,tr,out)
 with torch.enable_grad(),tr:
  metrics=agent.update(dict(expert_slicer=Fixed(eb),train=Fixed(tb)),1)
 torch.cuda.synchronize()
 write(out/'operators.json',tr.rows);write(out/'boundaries.json',boundaries)
 result=dict(operators=len(tr.rows),first_difference=tr.first,wall_s=time.monotonic()-start,
    post_model=tree_digest(agent._model.state_dict()),post_adam=tree_digest(agent.optimizer_dict),
    metrics={k:float(v) for k,v in metrics.items() if torch.is_tensor(v) and v.numel()==1})
 write(out/'trace-result.json',result);return result

def run(e,m,contract,out,args):
 from diagnostic_fb_resume import restore
 from dual_track_fb_online import factory,Collector,run_updates
 old=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/dual-track-diagnostics-20261110/runs/d2-control-02')
 if args.mode=='trace-control':
  agent,_=factory(compile=False);c=Collector(e,m,contract,agent,2048);c.bootstrap_sources();c.collect(1024)
  run_updates(agent,c,0,1,False);c.collect(13);c.collect(27)
  ref=json.loads((old/'after-collection-fingerprints.json').read_text())
  actual=fingerprints(agent,c,1);assert actual==ref, ('uninterrupted reconstruction differs',actual,ref)
  result=trace(agent,c,out)
 else:
  agent,c,progress,meta=restore(old/'user-stop-1037',e,m,contract);c.collect(27)
  actual=fingerprints(agent,c,1);ref=json.loads((old/'after-collection-fingerprints.json').read_text());assert actual==ref
  result=trace(agent,c,out,args.reference)
 return dict(status='completed',formal_training=False,engineering_agent_updates=2 if args.mode=='trace-control' else 1,
   new_native_transitions=1064 if args.mode=='trace-control' else 27,trace=result)
