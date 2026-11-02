import unittest,torch
from narrow_policy import Expert,legal_command,update
class Tests(unittest.TestCase):
 def test_legal_latent(self):
  class Model:
   def project_z(self,z):return 16*torch.nn.functional.normalize(z,dim=-1)
  b=16*torch.nn.functional.normalize(torch.randn(7,256),dim=-1)
  z,_=legal_command(Model(),b,torch.zeros_like(b));self.assertTrue(torch.equal(b,z))
  z,_=legal_command(Model(),b,torch.randn_like(b)*5)
  self.assertLess(float((z.norm(dim=-1)-16).abs().max()),1e-4)
  self.assertLessEqual(float(torch.rad2deg(torch.acos(((z*b).sum(-1)/256).clamp(-1,1))).max()),45.01)
 def test_feedback_update(self):
  torch.manual_seed(4);n=Expert(12);n.log_std.requires_grad_(False);x=torch.randn(256,12)
  with torch.no_grad():
   d=n.distribution(x);a=d.sample();lp=d.log_prob(a).sum(-1);v=n.value(x);adv=a[:,0]/.02;adv=(adv-adv.mean())/adv.std();ret=v+adv
  opt=torch.optim.Adam([dict(params=n.actor.parameters(),lr=3e-5),dict(params=n.critic.parameters(),lr=3e-4)])
  s=n.log_std.clone();u=update(n,opt,dict(obs=x,action=a,logp=lp,value=v,advantages=adv,returns=ret,raw_adv_std=adv.std()),epochs=2)
  self.assertEqual(u['pre_logprob_max_error'],0);self.assertTrue(torch.equal(s,n.log_std));self.assertGreater(float(n.actor(x)[:,0].mean()),0)
  self.assertGreater(u['gradients'][0]['actor'],0);self.assertGreater(u['gradients'][0]['critic'],0)
if __name__=='__main__':unittest.main()
