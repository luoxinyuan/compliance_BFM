"""No model/simulator loading: motor ownership, legal coordinates, and PPO checks."""
import sys,types,unittest,torch
from pathlib import Path
# Pure tensor stubs isolate primitive sampling from heavyweight physics imports.
sys.modules['m9_cyclic']=types.SimpleNamespace(sample=lambda ref,phase:ref['tensor'])
def sphere(a,b,u):return torch.nn.functional.normalize(a*(1-u)+b*u,dim=-1)*16
sys.modules['locomotion_rollout']=types.SimpleNamespace(spherical=sphere)
from clean_policy import Expert,direct_command,structured_command,update
class Model:
 def project_z(self,x):return torch.nn.functional.normalize(x,dim=-1)*16
class Tests(unittest.TestCase):
 def test_own_commands_not_force_rules(self):
  s=torch.zeros(1,256);s[0,0]=16;w=s.clone();w[0,0]=12;w[0,1]=torch.sqrt(torch.tensor(112.))
  ref={'tensor':w,'period':1.42};a=torch.zeros(1,258)
  z,b,alpha,phase,rate,n=structured_command(Model(),s,ref,a,0.,0.)
  torch.testing.assert_close(z,s);self.assertEqual(phase,0.)
  a[0,:2]=torch.tensor([1.,1.]);z,b,alpha,phase,rate,n=structured_command(Model(),s,ref,a,0.,0.)
  self.assertGreater(alpha,0);self.assertGreater(phase,0);self.assertAlmostEqual(float(z.norm()),16,places=5)
  a[0,:2]=0;_,_,alpha2,phase2,_,_=structured_command(Model(),s,ref,a,alpha,phase)
  self.assertLess(alpha2,alpha);self.assertEqual(phase2,phase)
 def test_exact_structured_teacher_inverse(self):
  torch.manual_seed(1);b=torch.nn.functional.normalize(torch.randn(1,256),dim=-1)*16;d=torch.randn_like(b);d-=((d*b).sum()/256)*b;d=d/d.norm()*10;z=Model().project_z(b+d)
  recovered=z/((z*b).sum()/256)-b;torch.testing.assert_close(recovered,d,atol=3e-6,rtol=3e-6)
 def test_direct_and_probability(self):
  for kind in ['structured','direct']:
   torch.manual_seed(2);n=Expert(12,kind);x=torch.randn(64,12);dist=n.distribution(x);a=dist.sample();v=n.value(x).detach();ret=v+torch.randn(64)
   batch=dict(obs=x,action=a,logp=dist.log_prob(a).sum(-1).detach(),advantages=(ret-v),returns=ret,value=v,raw_adv_std=(ret-v).std())
   opt=torch.optim.Adam([dict(params=n.actor.parameters(),lr=1e-5),dict(params=n.critic.parameters(),lr=3e-4)])
   r=update(n,opt,batch,epochs=1,minibatch=64);self.assertLess(r['pre_logprob_max_error'],1e-5);self.assertGreater(r['optimizer_updates'],0)
  s=torch.randn(1,256);s=s/s.norm()*16
  torch.testing.assert_close(direct_command(s/16,s),s)
  torch.testing.assert_close(direct_command(torch.zeros_like(s),s),s)
 def test_rejected_update_restores_weights_and_adam(self):
  torch.manual_seed(4);n=Expert(12,'direct');x=torch.randn(64,12);d=n.distribution(x);a=d.sample();v=n.value(x).detach();ret=v+torch.randn(64)
  b=dict(obs=x,action=a,logp=d.log_prob(a).sum(-1).detach(),advantages=ret-v,returns=ret,value=v,raw_adv_std=(ret-v).std())
  before={k:p.clone() for k,p in n.state_dict().items()};opt=torch.optim.Adam([dict(params=n.actor.parameters(),lr=100.),dict(params=n.critic.parameters(),lr=3e-4)])
  r=update(n,opt,b,epochs=1,minibatch=64);self.assertEqual(r['optimizer_updates'],0);self.assertEqual(len(r['rejected_candidates']),7)
  for k,p in n.state_dict().items():torch.testing.assert_close(p,before[k],rtol=0,atol=0)
  self.assertEqual(len(opt.state),0);self.assertLessEqual(r['epochs'][-1]['joint_gaussian_KL'],.02001)
 def test_no_legacy_online_controller(self):
  import ast
  t=ast.parse(Path(__file__).with_name('clean_env.py').read_text())
  cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Task')
  for f in cls.body:
   if isinstance(f,ast.FunctionDef) and f.name in ['reset','prepare','features','step']:
    src=ast.get_source_segment(Path(__file__).with_name('clean_env.py').read_text(),f)
    self.assertNotIn('CyclicFeedback(',src);self.assertNotIn('self.prior=',src);self.assertNotIn('self.nominal.step',src)
if __name__=='__main__':unittest.main()
