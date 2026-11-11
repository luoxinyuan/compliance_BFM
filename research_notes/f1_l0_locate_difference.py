import json,time
from pathlib import Path
import torch,numpy as np
S=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/f1-l0-20261111');O=S/'runs/l0-01';start=time.monotonic()
a=torch.load(O/'P1-1-timing-1.0-0.pt',map_location='cpu',weights_only=False);b=torch.load(O/'P1-1-timing-1.0-1.pt',map_location='cpu',weights_only=False);diff=[]
def walk(a,b,path=''):
 if torch.is_tensor(a) or isinstance(a,np.ndarray):
  aa=a.detach().numpy() if torch.is_tensor(a) else a;bb=b.detach().numpy() if torch.is_tensor(b) else b
  if aa.shape!=bb.shape:diff.append({'path':path,'shapes':[list(aa.shape),list(bb.shape)]});return
  mask=aa!=bb
  if mask.any():
   pos=np.argwhere(mask);entry={'path':path,'shape':list(aa.shape),'dtype':str(aa.dtype),'different_elements':int(mask.sum()),'first_indices':pos[:8].tolist()}
   if aa.dtype.kind in 'fiu':entry.update(max_abs=float(np.max(np.abs(aa.astype(float)-bb.astype(float)))),a_values=[float(aa[tuple(p)]) for p in pos[:8]],b_values=[float(bb[tuple(p)]) for p in pos[:8]])
   diff.append(entry)
 elif isinstance(a,dict):
  if a.keys()!=b.keys():diff.append({'path':path,'keys_differ':True})
  for k in a.keys()&b.keys():walk(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  if len(a)!=len(b):diff.append({'path':path,'length':[len(a),len(b)]})
  for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
 elif a!=b:diff.append({'path':path,'a':str(a)[:160],'b':str(b)[:160]})
walk(a,b)
result={'different_fields':diff,'difference_count':len(diff),'top_level_status':{k:not any(r['path'].startswith('/'+k+'/') or r['path']=='/'+k for r in diff) for k in a},'cpu_seconds':time.monotonic()-start,'analysis':'exact archived pair; no new GPU/physics; no tolerance change'}
(O/'first-difference.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
