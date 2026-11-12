"""No physics or learning. Verify the exact FP32 image of original action->q_PD.
A failed inverse is reported, never approximated or accepted by tolerance.
"""
import argparse,json,time,hashlib,struct
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();start=time.time()
c=json.loads(a.contract.read_text());s=json.loads(a.states.read_text());device=a.device
torch.set_num_threads(1)
scale=torch.tensor(c['q_PD_scale'],dtype=torch.float32,device=device);offset=torch.tensor(c['q_PD_offset'],dtype=torch.float32,device=device)
assert c['normalize_ratio']==5 and c['clip']==5 and torch.all(scale>0)
qmin=torch.tensor([r['range'][0] for r in s['contract']],dtype=torch.float32,device=device)[None,:];qmax=torch.tensor([r['range'][1] for r in s['contract']],dtype=torch.float32,device=device)[None,:]
margin=torch.maximum(torch.tensor(.02,device=device),.05*(qmax-qmin));lo=qmin+margin;hi=qmax-margin
assert torch.all(lo<=hi)
def g(x):return offset+(x*5).clamp(-c['clip'],c['clip'])*scale/5
base=torch.tensor([r['base_mean'][0] for r in s['controls']],dtype=torch.float32,device=device).clamp(-1,1)
raw=torch.tensor([r['actual'][0] for r in s['controls']],dtype=torch.float32,device=device)
q0=g(base);qr=g(raw);dminus=torch.minimum(torch.tensor(.12,device=device),.25*(q0-lo).clamp_min(0));dplus=torch.minimum(torch.tensor(.12,device=device),.25*(hi-q0).clamp_min(0))
dq=qr-q0;ds=torch.maximum(torch.minimum(dq,dplus),-dminus);wanted=q0+ds
inverse=((wanted-offset)*5/scale)/5
reconstructed=g(inverse)
mismatch=reconstructed!=wanted
# Unmodified action / zero residual are exactly preserved with their known inverse;
# inspect only truly constrained proposals, so cancellation in identity is not a spurious blocker.
clipped=ds!=dq
mask=mismatch&clipped
# Search all ordered float32 actions in [-1,1] for the first constrained mismatch.
def ordered(v):
 u=struct.unpack('I',struct.pack('f',v))[0];return (~u&0xffffffff) if u&0x80000000 else (u^0x80000000)
def from_ordered(k):
 u=(k^0x80000000) if k&0x80000000 else (~k&0xffffffff)
 return struct.unpack('f',struct.pack('I',u))[0]
proof=None
for rc in mask.nonzero().tolist():
 r,i=rc;target=float(wanted[r,i]);sc=scale[0,i];of=offset[0,i]
 def scalar_g(key):
  x=torch.tensor(from_ordered(key),device=device,dtype=torch.float32)
  return float(of+(x*5).clamp(-5,5)*sc/5)
 low,high=ordered(-1.),ordered(1.)
 while low<high:
  mid=(low+high)//2
  if scalar_g(mid)<target:low=mid+1
  else:high=mid
 upper=low;lower=upper-1;yu=scalar_g(upper);yl=scalar_g(lower)
 if yu!=target:
  assert yl<target<yu and upper-lower==1
  proof=dict(frame=r,time=s['controls'][r]['time'],joint=c['dof_names'][i],base_action=float(base[r,i]),raw_action=float(raw[r,i]),q0=float(q0[r,i]),requested_qPD=target,requested_delta=float(ds[r,i]),analytic_inverse=float(inverse[r,i]),analytic_roundtrip=float(reconstructed[r,i]),lower_action=from_ordered(lower),upper_action=from_ordered(upper),lower_qPD=yl,upper_qPD=yu,adjacent_float32_keys=[lower,upper],minimum_unavoidable_error_rad=min(target-yl,yu-target),scope='Monotone exhaustive binary search of ALL float32 normalized actions in [-1,1]; no exact preimage between adjacent actions.')
  break
# Static microbenchmark only, same original operation sequence, with a differentiable analytic inverse.
def step(b,ra):
 bq=g(b);q=g(ra);dm=torch.minimum(torch.tensor(.12,device=device),.25*(bq-lo).clamp_min(0));dp=torch.minimum(torch.tensor(.12,device=device),.25*(hi-bq).clamp_min(0));out=bq+torch.maximum(torch.minimum(q-bq,dp),-dm);return ((out-offset)*5/scale)/5
bench={}
for batch in (1,32):
 b=base[:1].repeat(batch,1).detach().requires_grad_();ra=raw[:1].repeat(batch,1).detach().requires_grad_();ts=[]
 for k in range(110):
  if device=='cuda':torch.cuda.synchronize()
  t=time.perf_counter();z=step(b,ra);z.square().mean().backward()
  if device=='cuda':torch.cuda.synchronize()
  dt=time.perf_counter()-t;b.grad=None;ra.grad=None
  if k>=10:ts.append(dt)
 ts.sort();bench[str(batch)]=dict(p50_s=ts[50],p90_s=ts[90],max_s=max(ts),samples=100,scope='operator only; not full training throughput; inverse qualification pending')
result=dict(status='BLOCKED_NO_EXACT_FP32_PREIMAGE' if proof else 'NO_DISPROOF_FOUND',device=device,frames=len(base),joints=29,constrained_entries=int(clipped.sum()),analytic_inverse_mismatch_on_constrained_entries=int(mask.sum()),proof=proof,mechanical_table=[dict(name=n,qmin=float(qmin[0,i]),qmax=float(qmax[0,i]),margin=float(margin[0,i]),lo=float(lo[0,i]),hi=float(hi[0,i]),scale=float(scale[0,i]),offset=float(offset[0,i])) for i,n in enumerate(c['dof_names'])],source_sha={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.contract,a.states]},microbenchmark=bench,wall_s=time.time()-start,physics_ticks=0,training_updates=0)
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='mechanical_table'},indent=2))
