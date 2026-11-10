"""CPU execution-contract checks, independent of physical qualification."""
import copy,json,types
import torch
from fb_safe_action import SafeAction,install,KEY
class Dist:
 def __init__(self,mean):self.mean=mean
 def sample(self,clip=None):
  noise=torch.randn_like(self.mean)*.2
  if clip is not None:noise=noise.clamp(-clip,clip)
  return self.mean+noise
class Actor(torch.nn.Module):
 def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(29))
 def forward(self,obs,z,std):return Dist(self.bias[None].expand(len(z),-1))
class Normal(torch.nn.Module):
 def forward(self,obs):return {'robot':obs['robot']}
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self._actor=Actor();self._obs_normalizer=Normal()

def main():
 torch.set_num_threads(1)
 names=['j'+str(i) for i in range(29)];names[5]='left_ankle_roll_joint';names[11]='right_ankle_roll_joint'
 c=dict(q_PD_offset=[[0.]*29],q_PD_scale=[[2.]*29],clip=5.,normalize_ratio=5.,dof_names=names)
 t=SafeAction(c,[[-.26,.26]]*29,'cpu');m=Model();original=copy.deepcopy(m).requires_grad_(False)
 install(m,original,t);obs={'robot':torch.ones(3,4),KEY:torch.ones(3,1)};z=torch.zeros(3,256)
 normalized=m._obs_normalizer(obs);assert torch.equal(normalized[KEY],obs[KEY])
 torch.manual_seed(58);raw=m._actor(normalized,z,.2);sample=raw.sample(clip=.3)
 after=torch.get_rng_state();torch.manual_seed(58);proposal=Dist(torch.zeros(3,29)).sample(clip=.3);assert torch.equal(after,torch.get_rng_state())
 expected=proposal.clamp(-1,1);expected[:,[5,11]]=expected[:,[5,11]].clamp(-.105,.105)
 assert torch.allclose(sample,expected,atol=1e-8,rtol=1e-8)
 assert bool((t.last['q_PD'][:,[5,11]].abs()<=.21+1e-7).all())
 assert torch.equal(t.last['raw_proposal'],proposal)
 with torch.no_grad():m._actor.bias.fill_(.03)
 m.zero_grad();m._actor(normalized,z,.2).mean.sum().backward();assert m._actor.bias.grad.abs().sum()>0
 obs[KEY].zero_();assert torch.equal(m._actor(m._obs_normalizer(obs),z,.2).mean,torch.zeros(3,29))
 print(json.dumps(dict(status='passed',single_noise_draw=True,actual_action_qPD_mapping=True,raw_proposal_preserved=True,
  bounded_ankle_targets=True,execution_context_preserved_without_new_neural_inputs=True,differentiable_inside_bounds=True)))
if __name__=='__main__':main()
