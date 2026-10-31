"""Training-only calibrated signed-force latent baseline, no anchor or joint override."""
import hashlib,json
from pathlib import Path
import torch
from m6_learning import RUN,done,read

def calibration(dev):
 path=done('train0')/'dataset.pt';d=torch.load(path,map_location=dev,weights_only=True)
 neg=[i for i,m in enumerate(d['manifest']) if m['condition']['delta_xyz'][0]<0]
 mask=torch.zeros(len(d['h']),device=dev,dtype=torch.bool)
 for i in neg:mask |= d['group']==i
 # Last force is observed causal history, not task direction at deployment.
 f=(-5*d['h'][:,-3]).clamp_min(0);mask &= f>0
 z=d['z'][mask];b=d['base'][mask];s=f[mask];assert len(s)>0
 # Invert spherical normalization into the standing tangent plane.
 tangent=z/((z*b).sum(-1,keepdim=True)/256)-b
 template=(s[:,None]*tangent).sum(0,keepdim=True)/s.sum()
 ref=(s.square().sum()/s.sum()).clamp_min(1e-6)
 info=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),samples=len(s),force_reference_n=float(ref),tangent_norm=float(template.norm()),formula='force-weighted mean training-negative projected tangent; coefficient clip(-mean(last3 Fx)/Fref,0,1) times existing residual contact gate; plus unchanged causal forward native path',uses_no_final_samples=True)
 return template,ref,info

def command(model,base,h,gate,template,ref):
 force_x=5*h[:,-9:].reshape(-1,3,3)[:,:,0].mean(-1,keepdim=True)
 coefficient=(-force_x/ref).clamp(0,1)*gate
 tangent=template-(template*base).sum(-1,keepdim=True)/256*base
 z=model.project_z(base+coefficient*tangent)
 return torch.where(coefficient==0,base,z)
