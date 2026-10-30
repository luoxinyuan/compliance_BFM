import ast,json,unittest
from pathlib import Path
D=Path(__file__).parent
class Contract(unittest.TestCase):
 def test_fixed_design(self):
  p=json.loads((D/'skill_compat_protocol.json').read_text());a,b=p['conditions']
  self.assertEqual({k:v for k,v in a.items() if k not in ['name','delta_x_m']},{k:v for k,v in b.items() if k not in ['name','delta_x_m']})
  self.assertEqual(a['delta_x_m'],.035);self.assertAlmostEqual(b['delta_x_m'],.18681928110688906)
  self.assertEqual(p['resources']['pair_branches'],6)
 def test_no_training_or_direct_joint_control(self):
  tree=ast.parse((D/'skill_compat_stage.py').read_text())
  calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
  self.assertFalse(any('optim' in c or c.endswith('backward') or 'arm_ik' in c for c in calls))
  self.assertIn('original_act',calls);self.assertIn('env.wrapped.step',calls)
 def test_schedule(self):
  tree=ast.parse((D/'skill_compat_stage.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transition_weights')
  def smooth(x):x=max(0,min(1,x));return x*x*(3-2*x)
  ns={'smooth':smooth};exec(compile(ast.Module(body=[node],type_ignores=[]),'schedule','exec'),ns)
  f=ns['transition_weights'];self.assertEqual(f(5.3),(0.,0.));self.assertEqual(f(6.3),(1.,0.));self.assertEqual(f(9.8),(1.,1.))
  self.assertLess(abs(f(5.300001)[0]-f(5.3)[0]),1e-10)
if __name__=='__main__':unittest.main()
