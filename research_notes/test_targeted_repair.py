import ast,json,unittest
from pathlib import Path
import torch
from targeted_repair import balanced_ids
HERE=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_fixed_balanced_batches(self):
  groups=torch.tensor([0]*60+[1]*60+[2]*10+[3]*10+[4]*10+[5]*10)
  for _ in range(100):
   ids=balanced_ids({'group':groups});self.assertEqual([(groups[ids]==j).sum().item() for j in range(6)],[4,4,2,2,2,2]);self.assertEqual(len(set(ids.tolist())),16)
 def test_eval_seal(self):
  p=json.loads((HERE/'targeted_repair_protocol.json').read_text());self.assertEqual(p['fit']['updates'],2000)
  self.assertEqual([v['prefix'] for v in p['evaluations'] if v['id'].startswith('new27')],[2.7,2.7])
  s=(HERE/'targeted_repair.py').read_text();self.assertNotIn('new27',s);self.assertNotIn('2.7',s)
 def test_no_teacher_privilege_to_student(self):
  tree=ast.parse((HERE/'targeted_repair.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='deploy_input']
  self.assertEqual(len(calls),1);self.assertEqual(ast.unparse(calls[0]),'deploy_input(model, ob, history, keys)')
  self.assertNotIn('env.step(', (HERE/'targeted_repair.py').read_text())
 def test_original_runtime_hooks(self):
  old=ast.parse((HERE/'contact_direction_teacher.py').read_text());new=ast.parse((HERE/'contact_repair_run.py').read_text())
  for name in ['_physics_step','sample_native_contacts']:
   get=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(get(old)),ast.dump(get(new)))
 def test_eval_has_no_optimizer_and_exact_state_capture(self):
  source=(HERE/'repair_evaluation.py').read_text()
  self.assertNotIn('backward(',source);self.assertNotIn('Adam(',source);self.assertNotIn('fit_repair(',source)
  old=ast.parse((HERE/'contact_direction_teacher.py').read_text().replace('previous_action=env.previous_action.tolist(),','previous_action=None if env.previous_action is None else env.previous_action.tolist(),'));new=ast.parse(source)
  find=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
  self.assertEqual(ast.dump(find(old)),ast.dump(find(new)))
 def test_reset_and_prefix_action_serialization(self):
  for value,expected in [(None,None),(torch.zeros(1,29),[[0.]*29])]:
   actual=None if value is None else value.tolist();self.assertEqual(actual,expected)
  src=(HERE/'contact_repair_run.py').read_text()
  self.assertEqual(src.count('fit_repair(model,'),1)
  self.assertIn("if stage=='repair':",src)
  self.assertIn("REUSE_COMPLETED_NO_RERUN",(HERE/'repair_evaluation.py').read_text())
if __name__=='__main__':unittest.main()
