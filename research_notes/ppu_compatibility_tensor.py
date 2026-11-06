"""Real frozen-BFM tensor parity and derivative check on PPU; no policy training."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
import sys,json,time,hashlib
from pathlib import Path
B=Path('/root/compliance-ppu-20261106');R=B/'repo'
sys.path[:0]=[str(R),str(R/'research_notes')]
import torch
from humanoidverse.agents import load_utils
from frozen_bfm_placement import place_inference_modules
from clean_policy import Expert,direct_command
torch.set_num_threads(1);torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
t=time.monotonic()
m=load_utils.load_model_from_checkpoint_dir(P/'issue19-recovery-20261028T045520Z/model-view/checkpoint',device='cpu')
print('FULL_ORIGINAL_BFM_LOADED',flush=True)
placement=place_inference_modules(m)
print('ORIGINAL_INFERENCE_ON_PPU',flush=True)
data=json.loads((B/'output/nvidia-saved-reference.json').read_text());comparisons=[]
for name,group in data.items():
 for r in group['controls']:
  obs={k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in r['bfm_obs'].items()}
  z=torch.tensor(r['executed_z'],device='cuda',dtype=torch.float32)
  a=m.act(obs,z,mean=True)
  ref=torch.tensor([r['action']],device='cuda',dtype=torch.float32)
  comparisons.append(dict(model=name,time=r['time'],action_max_abs=float((a-ref).abs().max()),action_rms=float((a-ref).square().mean().sqrt())))
r=data['C0']['controls'][3]
obs={k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in r['bfm_obs'].items()}
with torch.enable_grad():
 z=torch.tensor(r['executed_z'],device='cuda',requires_grad=True)
 # Public actor() is intentionally no_grad; LPP uses the same frozen _actor with a differentiable z.
 a=m._actor(m._normalize(obs),m.project_z(z),m.cfg.actor_std).mean.float()
 loss=a.square().mean();g=torch.autograd.grad(loss,z)[0]
 assert torch.isfinite(g).all() and g.norm()>0
 gradient=dict(norm=float(g.norm()),max=float(g.abs().max()),device=str(g.device))
teachers=[]
for name,path,sha in [('I',P/'m39a-demand-20261104/B-ppo-03/job/best.pt','72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a'),('C0',P/'prior-recovery-20261105/correct-C0/C0.pt','ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c')]:
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 c=torch.load(path,map_location='cuda',weights_only=True)
 net=Expert(c['input_dim'],'direct').to('cuda');net.load_state_dict(c['state_dict']);net.eval().requires_grad_(False)
 ds=[]
 for r in data[name]['decisions']:
  raw=net.distribution(torch.tensor([r['obs']],device='cuda',dtype=torch.float32)).mean
  ds.append(float((raw-torch.tensor([r['raw']],device='cuda',dtype=torch.float32)).abs().max()))
 teachers.append(dict(model=name,checkpoint_sha256=sha,input_dim=c['input_dim'],max_raw_difference=max(ds)))
torch.cuda.synchronize()
out=dict(status='completed',placement=placement,comparison=comparisons,gradient=gradient,teachers=teachers,wall_s=time.monotonic()-t,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,bfm_updated=False,rl_decisions=0,supervised_updates=0)
(B/'output/bfm-tensor-results.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='placement'}),flush=True)
