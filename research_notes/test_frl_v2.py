"""Issue58 meaningful CPU regressions; no GPU, physics mutation or formal training."""
import copy,json
import numpy as np
import torch
from frl_v2_task import configure,sample,continuation,manifests
from frl_v2_retention import Retention
from frl_v2_ppo import update as changed
from dual_track_ppo import update as original
from fb_ordered_grad import ordered_component_gradients

def main():
 torch.set_num_threads(1)
 configure();r=np.random.default_rng(580100)
 for family in ('near','mid','stop','zero','none'):
  for _ in range(100):
   s=sample(r,family)
   assert s['source']=='standing' and s['total_cap']==10
   if family=='stop':
    assert .16<=s['distance']<=.45
    c=continuation(s)
    assert 'stop_time' not in c and s['parent_id']==c['parent_id']
    assert all(s[k]==c[k] for k in ('source_until','source_profile','onset','distance','ramp'))
 m=manifests();assert len(m['dev'])==32 and len(m['sealed'])==56
 # Analytic episode marginal, not finite rollout frequency.
 macro=np.array([.375,.125,.25,.125,.125]);n=200000;counts=dict.fromkeys(('near','mid','stop','zero','none'),0)
 for f in r.choice(list(counts),size=n,p=macro):
  counts[f]+=1
  if f=='stop':counts['mid']+=1
 empirical=np.array(list(counts.values()))/sum(counts.values())
 assert np.max(abs(empirical-[.3,.3,.2,.1,.1]))<.004
 class Net(torch.nn.Module):
  def __init__(self):
   super().__init__();self.actor=torch.nn.Linear(3,256);self.critic=torch.nn.Linear(3,1)
   self.log_std=torch.nn.Parameter(torch.full((256,),float(np.log(.008))),requires_grad=False)
  def distribution(self,x):return torch.distributions.Normal(self.actor(x),self.log_std.exp())
  def value(self,x):return self.critic(x).squeeze(-1)
 torch.manual_seed(58);net=Net();other=copy.deepcopy(net);x=torch.randn(12,3)
 d=net.distribution(x);raw=d.sample();b=dict(obs=x,action=raw,logp=d.log_prob(raw).sum(-1).detach(),
  advantages=torch.randn(12),returns=torch.randn(12),value=net.value(x).detach(),raw_adv_std=torch.ones(()),pre_mask=torch.ones(12,dtype=torch.bool))
 aopt=torch.optim.Adam([{'params':net.actor.parameters(),'lr':3e-5},{'params':net.critic.parameters(),'lr':3e-4}])
 bopt=torch.optim.Adam([{'params':other.actor.parameters(),'lr':3e-5},{'params':other.critic.parameters(),'lr':3e-4}])
 state=torch.get_rng_state();original(net,aopt,b,epochs=1,minibatch=6);torch.set_rng_state(state)
 ret=Retention(other,False);ret.seed_reference(x,b['pre_mask'])
 changed(other,bopt,b,epochs=1,minibatch=6,retention=ret)
 assert all(torch.equal(v,other.state_dict()[k]) for k,v in net.state_dict().items()),'A changed original PPO'
 hold=Retention(other,True);hold.seed_reference(x,b['pre_mask']);assert float(hold.divergence(other,x).max())==0
 assert torch.equal(hold.reference_mean,hold.initial.distribution(hold.reference).mean)
 assert torch.equal(hold.divergence(other,hold.reference),hold.divergence(other,hold.reference.clone()))
 with torch.no_grad():other.actor.bias.add_(.001)
 other.zero_grad();loss=hold.loss(other,x,b['pre_mask']);loss.backward()
 assert other.actor.bias.grad.norm()>0
 record=hold.after_batch(other,x,b['pre_mask']);assert record['mean_D_pre']>0 and record['coefficient_next']>.1
 # All three effective branches including second-order GP retained.
 torch.manual_seed(58);w=torch.nn.Linear(3,4).double();u=torch.randn(8,3,dtype=torch.float64,requires_grad=True)
 losses=[w(u).square().mean(),torch.nn.functional.softplus(w(u+1)).mean()]
 g=torch.autograd.grad(w(u*2).sum(),u,create_graph=True)[0]
 losses.append(g.square().mean())
 combined=torch.autograd.grad(sum(losses),tuple(w.parameters()),retain_graph=True,allow_unused=True)
 separate=ordered_component_gradients(losses,w.parameters())
 assert all(torch.allclose(a,b,atol=1e-12,rtol=1e-12) for a,b in zip(combined,separate))
 print(json.dumps(dict(status='passed',standing_mid_stop_contract=True,parents=[32,56],
   analytic_episode_probabilities=[.3,.3,.2,.1,.1],empirical_episode_probabilities=empirical.tolist(),
   A_matches_original_PPO_exact=True,retention_gradient_and_dual_rule=True,all_discriminator_loss_gradients_retained=True)))
if __name__=='__main__':main()
