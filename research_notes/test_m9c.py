import unittest,torch,numpy as np
from m9c_rl import Adapter,update,command
class M:
 def project_z(self,z):return 16*torch.nn.functional.normalize(z,dim=-1)
class Test(unittest.TestCase):
 def test_raw_gaussian_update_contract(self):
  torch.manual_seed(23);net=Adapter(8);opt=torch.optim.Adam(net.parameters(),lr=3e-5);episodes=[]
  with torch.no_grad():
   for j in range(2):
    rows=[]
    for i in range(50):
     x=torch.randn(1,8);d=net.distribution(x);a=d.sample();rows.append(dict(x=x,raw=a,logprob=d.log_prob(a).sum(-1),value=net.critic(x).flatten(),mean=d.mean,reward=-float(a.square().mean()),done=i==49))
    episodes.append(rows)
   result=update(net,opt,episodes)
  self.assertLess(result['preupdate_logprob_max_error'],2e-5);self.assertGreater(result['updates'],0);self.assertTrue(all(x['analytic_joint_kl']>=0 for x in result['logs']))
 def test_legal_z_and_zero_contact(self):
  m=M();z=m.project_z(torch.randn(1,256));basis=torch.linalg.qr(torch.randn(256,16)).Q.T;r={'tensor':torch.cat([z,z],0),'phase_offset':0};refs={k:r for k in ['straight','left','right']}
  for alpha in [0.,1.]:
   p,rate=command(m,z,refs,dict(turn=0.,alpha=alpha,phase=.2),torch.randn(1,19),basis);self.assertAlmostEqual(float(p.norm()),16,places=4);self.assertTrue(.77<rate<1.3)
   if alpha==0:self.assertTrue(torch.allclose(p,z,atol=2e-6))
if __name__=='__main__':unittest.main()
