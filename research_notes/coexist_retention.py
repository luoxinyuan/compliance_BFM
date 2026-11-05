"""Function retention of I on physically validated startup domain only."""
import json,gzip
from pathlib import Path
import torch
class Retention:
 kind='raw-function-reliable-startup'
 def __init__(self,net,out):
  old=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105');rows=[];sources=[]
  for m in json.loads((old/'fit2-B/initialization.json').read_text())['manifest']:
   if m['kind']=='walk':
    rr=json.load(gzip.open(m['source'],'rt'));rows += [r for r in rr if 4.-1e-7<=r['time']<=7.5];sources.append(m['source'])
  rec=old.parent/'prior-recovery-20261105';d=json.loads((rec/'recovery-decision-corrected.json').read_text())
  for p in d['accepted_label_files']:rows+=json.load(gzip.open(p,'rt'));sources.append(p)
  self.x=torch.tensor([r['x'] for r in rows],device='cuda');self.raw=net.distribution(self.x).mean.detach().clone();self.n=len(rows);self.gen=torch.Generator(device='cuda').manual_seed(4600)
  (out/'retention-sources.json').write_text(json.dumps(dict(records=self.n,sources=sources,independent_recovery_starts=3,scope='I queried on validated recovery and actual long-startup states; no local labels overridden'),indent=2))
 def sample(self,n):return torch.randint(self.n,(n,),device='cuda',generator=self.gen)
 def loss(self,net,ix):
  d=net.distribution(self.x[ix]);return .5*((d.mean-self.raw[ix])/d.stddev).square().sum(-1).mean()
 def diagnostics(self,net):
  with torch.no_grad():
   r=net.distribution(self.x).mean;u=r/r.norm(dim=-1,keepdim=True);v=self.raw/self.raw.norm(dim=-1,keepdim=True)
   return dict(raw_rms=float((r-self.raw).square().mean().sqrt()),z_angle_deg=float(torch.rad2deg(torch.acos((u*v).sum(-1).clamp(-1,1))).mean()))
