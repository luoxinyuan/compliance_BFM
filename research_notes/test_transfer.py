import ast,unittest
from pathlib import Path
import torch
from transfer_policy import Expert,update
class Tests(unittest.TestCase):
 def test_student_input_allowlist(self):
  tree=ast.parse(Path(__file__).with_name('transfer_student.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sinput');attrs={n.attr for n in ast.walk(fn) if isinstance(n,ast.Attribute)}
  self.assertFalse(attrs&{'anchor','active','time','spec','last_support','partner_velocity','teacher_state','qpos','qvel'})
  self.assertEqual({n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str)},set())
 def test_protocol_split(self):
  import json
  p=json.loads(Path(__file__).with_name('transfer_protocol.json').read_text());train={(x['distance'],x['ramp'],x['onset']) for x in p['dev'].values()};final={(x['distance'],x['ramp'],x['onset']) for x in p['final'].values()};self.assertFalse(train&final)
 def test_bc_replay_finite_and_std_fixed(self):
  torch.manual_seed(7);n=Expert(8,'direct');n.actor[-1].weight.data.normal_(0,.002);x=torch.randn(32,8);d=n.distribution(x);a=d.sample();v=n.value(x).detach();b=dict(obs=x,action=a,logp=d.log_prob(a).sum(-1).detach(),advantages=torch.linspace(-1,1,32),returns=v+torch.randn(32),value=v,raw_adv_std=torch.tensor(1.))
  o=torch.optim.Adam([dict(params=n.actor.parameters(),lr=1e-5),dict(params=n.critic.parameters(),lr=3e-4)]);std=n.log_std.clone();u=update(n,o,b,epochs=1,minibatch=32,replay=(x,d.mean.detach()),bc_weight=.02)
  self.assertTrue(torch.equal(std,n.log_std));self.assertLessEqual(u['epochs'][-1]['joint_gaussian_KL'],.02001);self.assertEqual(u['epochs'][-1]['bc_weight'],.02)
if __name__=='__main__':unittest.main()
