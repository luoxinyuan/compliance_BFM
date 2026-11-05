"""Saved rollout sampling audit, small high-level actors only; no BFM CPU fallback."""
import json,time,math,hashlib
from pathlib import Path
import numpy as np,torch
from transfer_policy import Expert,direct_command
from geometry_policy import expand
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105')
def main():
 started=time.monotonic();torch.set_num_threads(1);torch.set_grad_enabled(False);rows=[]
 for seed in [4601,4602]:
  d=R/('ppo'+str(seed))
  for version in [0,8,16,24]:
   p=d/('initial.pt' if version==0 else f'candidate-{version:04d}.pt');rb=d/f'rollback-{version:04d}.json'
   if rb.exists():
    restored=json.loads(rb.read_text())['restored_iteration'];p=d/('initial.pt' if restored==0 else f'candidate-{restored:04d}.pt')
   if not p.exists():continue
   ck=torch.load(p,map_location='cpu',weights_only=True);net=Expert(1088,'direct');expand(net,ck)
   parts=[]
   for f in sorted(d.glob(f'worker-*/rollout-{version:04d}.npz')):
    st=json.loads(f.with_name(f.stem+'-stats.json').read_text());a=np.load(f);x=torch.tensor(a['obs']);raw=torch.tensor(a['action']);dist=net.distribution(x);err=(dist.log_prob(raw).sum(-1)-torch.tensor(a['logp'])).abs();assert err.max()<.002,(p,float(err.max()))
    mean=dist.mean;noise=raw-mean;z=16*torch.nn.functional.normalize(raw,dim=-1);zm=16*torch.nn.functional.normalize(mean,dim=-1);ang=torch.rad2deg(torch.acos((z*zm).sum(-1).div(256).clamp(-1,1)))
    for family in sorted(set(s['demand_family'] for s in st)):
     ix=torch.tensor([j for j,s in enumerate(st) if s['demand_family']==family]);parts.append(dict(worker=f.parent.name,family=family,n=len(ix),mean_raw_norm=float(mean[ix].norm(dim=-1).mean()),noise_l2=float(noise[ix].norm(dim=-1).mean()),noise_coordinate_std=float(noise[ix].std()),z_noise_angle_mean_deg=float(ang[ix].mean()),z_noise_angle_max_deg=float(ang[ix].max()),logprob_cpu_reproduction_max=float(err[ix].max())))
   if parts:rows.append(dict(seed=seed,sampling_version=version,checkpoint=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),groups=parts))
 result=dict(rows=rows,seconds=time.monotonic()-started,rl_decisions=0,scope='actual sampled batches at saved policy boundaries; only small actor inference on CPU; no new physical samples or optimization')
 (R/'sampled-noise-audit.json').write_text(json.dumps(result,indent=2));print(len(rows),result['seconds'])
if __name__=='__main__':main()
