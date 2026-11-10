"""CPU-only tensor-level investigation of the failed restart update equality check."""
import json,sys,time
from pathlib import Path
import torch
from dual_track_high import write

def main(a,b,out):
 out.mkdir(exist_ok=False);torch.set_num_threads(2);start=time.monotonic();rows=[]
 for file in ['model.pt','optimizers.pt']:
  left=torch.load(a/file,map_location='cpu',weights_only=False,mmap=True);right=torch.load(b/file,map_location='cpu',weights_only=False,mmap=True)
  def compare(x,y,path):
   if torch.is_tensor(x):
    if not torch.equal(x,y):
     delta=(x.double()-y.double()).abs();rows.append(dict(file=file,path=path,shape=list(x.shape),dtype=str(x.dtype),
       unequal=int((x!=y).sum()),max_abs=float(delta.max()),rms=float(delta.square().mean().sqrt()),reference_max=float(x.abs().max())))
   elif isinstance(x,dict):
    assert x.keys()==y.keys()
    for k in x:compare(x[k],y[k],path+'/'+str(k))
   elif isinstance(x,(list,tuple)):
    assert len(x)==len(y)
    for i,(xx,yy) in enumerate(zip(x,y)):compare(xx,yy,path+'/'+str(i))
   elif x!=y:rows.append(dict(file=file,path=path,non_tensor_difference=[x,y]))
  compare(left,right,'')
  del left,right
 write(out/'differences.json',dict(rows=rows,wall_s=time.monotonic()-start))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
