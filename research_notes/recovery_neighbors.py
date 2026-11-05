"""Post-contact nearest-input description, without conflating proximity with ambiguity."""
import argparse,os,json,gzip,hashlib,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--gpu');a,_=p.parse_known_args();a.output.mkdir();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;start=time.monotonic()
sys.path.insert(0,'/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/prior-recovery/repo/research_notes')
import torch,numpy as np
from clean_policy import Expert
from geometry_policy import expand
torch.set_num_threads(1);torch.set_grad_enabled(False);O=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105');ident=json.loads((O/'fit2-B/initialization.json').read_text());I=O.parent/'m39a-demand-20261104/B-ppo-03/job/best.pt';n=Expert(1088,'direct').cuda();expand(n,torch.load(I,map_location='cuda',weights_only=True));rows=[];groups={}
for m in ident['manifest']:
 rr=json.load(gzip.open(m['source'],'rt'));ii=np.linspace(0,len(rr)-1,m['samples']).round().astype(int);rr=[rr[i] for i in ii if 4.3<=rr[i]['time']<=9.0];groups[m['kind']]=np.arange(len(rows),len(rows)+len(rr));rows+=rr
x=n.normalized(torch.tensor([r['x'] for r in rows],device='cuda'));z=torch.tensor([r['z'] for r in rows],device='cuda');results=[]
for mode in ['all','without_own_z']:
 xx=x if mode=='all' else torch.cat([x[:,:565],x[:,821:]],-1)
 for k in ['arm5','arm15','left','right']:
  ii=groups[k];jj=groups['walk'];d=torch.cdist(xx[ii],xx[jj])/xx.shape[-1]**.5;v,ix=torch.topk(d.flatten(),min(5,d.numel()),largest=False)
  for dd,j in zip(v,ix):
   aa=int(ii[int(j)//len(jj)]);bb=int(jj[int(j)%len(jj)]);angle=float(torch.rad2deg(torch.acos((z[aa]*z[bb]).sum().div(256).clamp(-1,1))));results.append(dict(mode=mode,kind=k,times=[rows[aa]['time'],rows[bb]['time']],rms=float(dd),z_angle_deg=angle,privileged_error_xyz=[rows[aa]['x'][465:468],rows[bb]['x'][465:468]]))
(a.output/'neighbors.json').write_text(json.dumps(results,indent=2));(a.output/'result.json').write_text(json.dumps(dict(status='completed',elapsed_s=time.monotonic()-start,updates=0)));print('POST_CONTACT_NEIGHBORS',results[:5])
