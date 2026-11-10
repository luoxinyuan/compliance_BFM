"""CPU-only alignment and actual-action preservation loss regressions."""
import copy,io,json,types
import torch
from fb_safe_data import SafeNativeSequences
from dual_track_fb_extension import CONTACT_KEY
from fb_safe_action import SafeAction,install,KEY
from fb_safe_update import install_retention

class Dist:
 def __init__(self,mean):self.mean=mean
 def sample(self,clip=None):
  noise=.2*torch.randn_like(self.mean)
  if clip is not None:noise=noise.clamp(-clip,clip)
  return self.mean+noise
class Actor(torch.nn.Module):
 def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(29))
 def forward(self,obs,z,std):return Dist(self.bias[None].expand(len(z),-1))
class Norm(torch.nn.Module):
 def forward(self,obs):return dict(obs)
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self._actor=Actor();self._obs_normalizer=Norm();self.amp_dtype=torch.bfloat16
  self.cfg=types.SimpleNamespace(actor_std=.2,amp=False)
 def _normalize(self,obs):return self._obs_normalizer(obs)
 def _critic(self,obs,z,action):return action.sum(-1)[None,:,None].expand(2,-1,1)
 def _aux_critic(self,obs,z,action):return self._critic(obs,z,action)
 def _forward_map(self,obs,z,action):return action.mean(-1)[None,:,None].expand(2,-1,256)
class Agent:
 def __init__(self,m):
  self._model=m;self.device='cpu';self.cfg=types.SimpleNamespace(model=types.SimpleNamespace(amp=False),
   train=types.SimpleNamespace(stddev_clip=.3,actor_pessimism_penalty=.5,scale_reg=True,reg_coeff=1.,reg_coeff_aux=1.))
  self.actor_optimizer=torch.optim.Adam(m._actor.parameters(),lr=1e-4)
 def update(self,*a,**kw):raise RuntimeError('not a real FB agent update')
 def get_targets_uncertainty(self,q,p):return q.mean(0),torch.zeros_like(q.mean(0)),q.mean(0)

def main():
 torch.set_num_threads(1);torch.manual_seed(5802)
 seq=SafeNativeSequences()
 for j,source in enumerate(seq.parents):
  rows=[]
  for i in range(24):
   code=float(j*100+i);obs={'robot':torch.tensor([[code,code]]),CONTACT_KEY:torch.zeros(1,97)}
   rows.append(dict(time=i*.02,observation=obs,next_observation={**obs,'robot':torch.tensor([[code+1,code+1]])},
     z=torch.full((1,256),code),action=torch.full((1,29),code)))
  seq.add(source,source+'-parent',rows);seq.add(source,source+'-parent',rows)
 assert all(len(x)==1 for x in seq.parents.values())
 data=seq.sample(1024)
 assert torch.equal(data['observation']['robot'][:,0],data['native_z'][:,0])
 assert torch.equal(data['observation']['robot'][:,0],data['native_action'][:,0])
 f=io.BytesIO();torch.save(seq,f);f.seek(0);loaded=torch.load(f,weights_only=False);assert loaded.parents.keys()==seq.parents.keys()
 names=['j'+str(i) for i in range(29)];names[5]='left_ankle_roll_joint';names[11]='right_ankle_roll_joint'
 contract=dict(q_PD_offset=[[0.]*29],q_PD_scale=[[2.]*29],clip=5.,normalize_ratio=5.,dof_names=names)
 transform=SafeAction(contract,[[-.26,.26]]*29,'cpu');model=Model();original=copy.deepcopy(model).requires_grad_(False)
 install(model,original,transform);agent=Agent(model);agent._safe_qpd_scale=transform.scale;agent._safe_action_transform=transform
 install_retention(agent,original)
 n=8;obs={'robot':torch.zeros(n,2),CONTACT_KEY:torch.zeros(n,97),KEY:torch.ones(n,1)};z=torch.ones(n,256)
 agent._native_retention_batch=dict(observation=obs,native_z=z,native_action=torch.zeros(n,29))
 with torch.no_grad():model._actor.bias.fill_(.03)
 before=model._actor.bias.detach().clone();metrics=agent.update_actor(obs,torch.zeros(n,29),z,1.)
 assert not torch.equal(before,model._actor.bias) and float(metrics['native_keep_valid_fraction'])==1
 agent._native_retention_batch['native_action'][:,[5,11]]=1.
 metrics=agent.update_actor(obs,torch.zeros(n,29),z,1.)
 assert float(metrics['native_keep_valid_fraction'])==0 and all(torch.isfinite(x) for x in metrics.values())
 print(json.dumps(dict(status='passed',native_source_z_action_time_alignment=True,source_parent_dedup=True,
  source_class_serialization=True,actor_keep_backward_valid=True,empty_valid_domain_backward_valid=True,
  synthetic_tests_only=True,actual_FB_agent_updates=0)))
if __name__=='__main__':main()
