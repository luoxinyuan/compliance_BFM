import ast,json,unittest
from pathlib import Path
P=Path(__file__).parent
class Contracts(unittest.TestCase):
 def test_seal_disjoint(self):
  p=json.loads((P/'m9b_protocol.json').read_text());dev=p['execution']['collect0']['conditions'];hold=p['sealed_before_new_training']['conditions'];self.assertFalse(set(dev)&set(hold));self.assertEqual(p['training']['steps'],[2500,2500])
 def test_deploy_inputs(self):
  tree=ast.parse((P/'m9b_learning.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='memory_policy');names={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)};self.assertFalse(names&{'anchor','spec','phase','guide','time','oracle_z'})
 def test_freeze_framework(self):
  s=(P/'m9b_physics.py').read_text();self.assertIn('p.grad is None',s);self.assertIn("torch.cuda.device_count()==1",s);self.assertNotIn('torch.optim.',s)
 def test_counterfactual_partition(self):
  for a,b,c,d in [(3,2,1,4),(-2,4,3,5)]:self.assertEqual(.5*((a-b)+(c-d))+.5*((a-c)+(b-d)),a-d)
 def test_no_command_teacher_memory(self):
  s=(P/'m9b_learning.py').read_text();self.assertIn("prev=np.array(c['executed_z'][0])",s);self.assertNotIn("prev=np.array(c['oracle_z']",s)
if __name__=='__main__':unittest.main()
