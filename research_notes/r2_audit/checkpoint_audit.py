import os,sys,time,json,io,zlib,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
from torch import nn
T=time.monotonic();torch.set_num_threads(2)
P=Path(os.environ['COMPLIANCE_PROJECT_ROOT']);S=P/'experiments/adapter-return-pilot-20261113';R=S/'runtime';O=Path(__file__).parent/'data';sys.path[:0]=[str(R),str(R/'research_notes')]
from research.interaction_fb.schema import deploy,privileged
from research.interaction_fb.envelope import Envelope
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
c=load(S/'runs/fit-01/return/adapter-64.pt')['agent'];env=Envelope(c['envelope'],'cpu')
def net(s,key):
 sd={k[len(key)+1:]:v for k,v in s['nets'].items() if k.startswith(key+'.')};m=nn.Sequential(nn.Linear(sd['0.weight'].shape[1],128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,sd['4.weight'].shape[0]));m.load_state_dict(sd);return m.eval().requires_grad_(False)
q=net(c,'q');out=[]
with torch.no_grad():
 for case in [2,3]:
  t=load(S/f'runs/evaluate-case-{case:02d}-01/V1-mean/trace.pt');f=t['actions'][0];o=f['observation'];z=f['z'];b=f['audit']['base'];v=f['audit']['qexec'];x=torch.cat([deploy(o),z,torch.zeros(1,32)],-1)
  for branch in ['control','return']:
   for step in [0,16,32,64]:
    p=S/f'runs/fit-01/{branch}/adapter-{step:02d}.pt';s=load(p)['agent'];raw=net(s,'adapter')(x);a,au=env(b,(b+raw).clamp(-1,1));value=q(torch.cat([privileged(o),a,z,torch.zeros(1,32)],-1))
    out.append(dict(case=case,branch=branch,step=step,SHA=hashlib.sha256(p.read_bytes()).hexdigest(),Q=float(value),raw_rad_rms=float((raw*env.scale).square().mean().sqrt()),qPD_change_vsV1_rms=float((au['qexec']-v).square().mean().sqrt()),qPD_change_vsV1_max=float((au['qexec']-v).abs().max()),trigger_count=int(au['triggered'].sum())))
(O/'checkpoint-drift.json').write_text(json.dumps(out,indent=2))
p=P/'experiments/interaction-pilot-v3-20261112/runs/pilot-B-v3/checkpoints/round-19/core.ptz';ck=torch.load(io.BytesIO(zlib.decompress(p.read_bytes())),map_location='cpu',weights_only=False)
# ONLY actually visited first10 specs; never inspect unopened later or sealed parents.
sp=[dict(index=i,parent=s['parent_id'],family=s['demand_family'],onset=s['onset'],release=s['release'],source=s['source'],end=s['end']) for i,s in enumerate(ck['task_specs'][:ck['spec_id']+1])]
(O/'visited-specs.json').write_text(json.dumps(sp,indent=2))
(O/'checkpoint-receipt.json').write_text(json.dumps(dict(wall_s=time.monotonic()-T,new_physics=0,new_updates=0,cuda_initialized=torch.cuda.is_initialized()),indent=2))
