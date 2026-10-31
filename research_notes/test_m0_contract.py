import ast,json,unittest
from pathlib import Path
D=Path(__file__).parent
class Contract(unittest.TestCase):
 def test_half_schedule(self):
  tree=ast.parse((D/'m0_half_stage.py').read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='fixed_progress')
  def smooth(u):
   u=min(1,max(0,u));return u*u*(3-2*u)
  scope={'smooth':smooth};exec(compile(ast.Module(body=[node],type_ignores=[]),'schedule','exec'),scope)
  f=scope['fixed_progress'];self.assertEqual([f(x) for x in [3,3.8,5.8,7.8,8.8,9.55,10.8]],[0,0,.25,.5,.5,0,0])
 def test_heldout_and_budget(self):
  p=json.loads((D/'m0_protocol.json').read_text());b=p['track_B']
  self.assertNotEqual(b['heldout']['onset'],b['development']['onset']);self.assertTrue(b['heldout']['sealed_before_training'])
  self.assertEqual(b['fit']['aggregation_iterations'],1);self.assertEqual(b['fit']['updates_each'],1000)
  self.assertEqual(p['resources']['compute_cumulative_wall_seconds'],3600)
 def test_deployable_inputs(self):
  tree=ast.parse((D/'m0_lpp_stage.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='deploy_input']
  self.assertEqual(len(calls),2)
  for call in calls:
   for arg in call.args:self.assertFalse(any(k in ast.unparse(arg) for k in ['anchor','time','direction','desired','target']))
 def test_release_labels_use_50hz_fade(self):
  tree=ast.parse((D/'m0_lpp_stage.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='query_expert');src=ast.unparse(fn)
  self.assertIn("fade.value(c['tether_time'])",src);self.assertIn("fade.update(now, decision['active'], desired)",src)
if __name__=='__main__':unittest.main()
