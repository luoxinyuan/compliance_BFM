"""CPU-only exact arithmetic reconstruction, no optimizer or policy updates."""
import argparse,hashlib,itertools,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args();S=a.stage
torch.set_num_threads(1)
def read(p):return json.loads(p.read_text())
def h(t):return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
C=read(S/'runs/trace-control-01/operators.json');R=read(S/'runs/trace-resume-01/operators.json')
CB=read(S/'runs/trace-control-01/boundaries.json');RB=read(S/'runs/trace-resume-01/boundaries.json')
pre=read(S/'runs/trace-control-01/batch-identity.json');post=read(S/'runs/trace-resume-01/batch-identity.json');assert pre==post
pool={(r['phase'],r['op'],v.get('hash'),str(v['shape'])) for r in C for v in r['outputs'] if 'empty' not in r['op']}
unique=[r for r in R if 'empty' not in r['op'] and any((r['phase'],r['op'],v.get('hash'),str(v['shape'])) not in pool for v in r['outputs'])]
prefix={'discriminator':'_discriminator','forward':'_forward_map','backward':'_backward_map','critic':'_critic','aux_critic':'_aux_critic','actor':'_actor'}
grad=[]
for x,y in zip(CB,RB):
 assert x['label']==y['label']
 if x['label'].endswith('pre-step'):
  pr=prefix[x['label'].split('_optimizer')[0]]+'.'
  dif=[k for k in x['gradients'] if k.startswith(pr) and x['gradients'][k]!=y['gradients'][k]]
  grad.append(dict(optimizer=x['label'].split('.')[0],different_gradient_tensors=len(dif),names=dif))
files=sorted((S/'runs/trace-resume-02').glob('disc-matrix-add-*.pt'))
ops=[torch.load(f,map_location='cpu',weights_only=False) for f in files]
known={h(x['outputs']) for x in ops};leaves={}
for x in ops:
 args,kwargs=x['inputs'];assert kwargs.get('alpha',1)==1
 for t in args:
  if torch.is_tensor(t) and h(t) not in known:leaves[h(t)]=t
assert len(leaves)<=6
targets={v['hash']:r['index'] for r in C for v in r['outputs'] if r['phase']=='discriminator_optimizer.backward' and r['op']=='aten.add.Tensor' and v['shape']==[1024,783]}
found={};vals=list(leaves.values());keys=list(leaves)
def trees(order):
 if len(order)==1:return [(vals[order[0]],str(order[0]))]
 out=[]
 for split in range(1,len(order)):
  for left,ls in trees(order[:split]):
   for right,rs in trees(order[split:]):out.append((left+right,'('+ls+'+'+rs+')'))
 return out
control_grad=next(x for x in CB if x['label']=='discriminator_optimizer.pre-step')['gradients']['_discriminator.trunk.0.weight']['hash']
resume_grad=next(x for x in RB if x['label']=='discriminator_optimizer.pre-step')['gradients']['_discriminator.trunk.0.weight']['hash']
solutions={};outputs={}
for n in range(2,len(vals)+1):
 for order in itertools.permutations(range(len(vals)),n):
  for t,expr in trees(order):
   digest=h(t)
   if digest in targets and digest not in found:found[digest]=dict(control_operator_index=targets[digest],expression=expr)
   if digest in (control_grad,resume_grad) and digest not in solutions:solutions[digest]=expr;outputs[digest]=t
delta=None
if control_grad in outputs and resume_grad in outputs:
 d=(outputs[control_grad]-outputs[resume_grad]).abs();delta=dict(max_abs=float(d.max()),different_elements=int((d!=0).sum()),elements=d.numel())
result=dict(batch_and_rng_pre_state_exact=True,operators_control=len(C),operators_resumed=len(R),
 first_execution_order_difference=read(S/'runs/trace-resume-01/first-difference.json'),
 first_new_intermediate_value=unique[0],optimizer_gradients=grad,
 addition_leaf_hashes=keys,addition_reconstruction=found,final_gradient_solutions=solutions,gradient_difference=delta,
 arithmetic_scope='CPU IEEE FP32 additions of actual saved CUDA backward operands; exact SHA matching to CUDA results, no tolerance',
 exclusions='uninitialized allocator bytes excluded; intermediate tensor schedule/order is separate from final gradient difference; target/Adam unchanged',
 contract_status='localization only, strict uninterrupted/resumed final weights remain unequal; no long-run qualification')
(S/'runs/trace-analysis').mkdir(exist_ok=False)
(S/'runs/trace-analysis/result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
