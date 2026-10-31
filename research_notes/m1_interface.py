"""Candidate continuous native-command surface; qualify before Student execution.
State coordinate u, forward progress a, and return r, plus original bounded local residual.
No discrete skill selector, additional joint actuator, or larger local residual radius.
"""
import json
from pathlib import Path
import torch
from contact_projector import constrained
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m1-state-response-20261031A'
def slerp(a,b,u):
 u=torch.as_tensor(u,device=a.device,dtype=a.dtype).reshape(-1,1)
 theta=torch.acos(((a*b).sum(-1,keepdim=True)/256).clamp(-1+1e-7,1-1e-7))
 return 16*torch.nn.functional.normalize(torch.sin((1-u)*theta)/torch.sin(theta)*a+torch.sin(u*theta)/torch.sin(theta)*b,dim=-1)
def endpoints(env):
 ledger=json.loads((RUN/'compute-ledger.json').read_text());qa=[a for a in ledger['attempts'] if a['stage']=='qualification' and a.get('result_status')=='completed'];assert len(qa)==1
 out=Path(qa[0]['path'])/'outputs';get=lambda n:json.loads((out/n).read_text());standing=env.standing
 raised=torch.tensor(get('raised__goal.json')['control'][160]['executed_z'],device=standing.device,dtype=standing.dtype)
 # t6.2 handoff command is the original raised endpoint; no new goal optimization.
 assert abs(get('raised__goal.json')['control'][160]['time']-6.2)<1e-8
 forward=torch.tensor(get('standing__goal.json')['goal_candidate'].get('z',json.loads((P/'bridge-v2-owned/runs/issue21-goalcalibration-20261030A/attempt-01/outputs/same-state-forward.json').read_text())['states'][-1]['per_mode']['native']['z']),device=standing.device,dtype=standing.dtype)
 raised_goal=torch.tensor(get('raised__goal.json')['goal_candidate']['z'],device=standing.device,dtype=standing.dtype)
 return [standing,raised,forward,raised_goal]
def base_latent(ends,u,a,r):
 standing,raised,forward,raised_goal=ends;pose=slerp(standing,raised,u);goal=slerp(forward,raised_goal,u);command=slerp(pose,goal,a);return slerp(command,standing,r)
def policy_latent(model,ends,raw,gate):
 coords=raw[:,:3].sigmoid();base=base_latent(ends,*coords.unbind(-1));return constrained(model,base,raw[:,3:],gate)[0]
class SharedProjector(torch.nn.Module):
 def __init__(self,h):
  super().__init__();self.register_buffer('center',h.mean(0));self.register_buffer('scale',h.std(0,unbiased=False).clamp_min(.1));self.net=torch.nn.Sequential(torch.nn.Linear(h.shape[1],128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,259));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
 def forward(self,h):return self.net((h-self.center)/self.scale)
