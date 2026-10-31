import ast,json,unittest
from pathlib import Path
D=Path(__file__).parent
class QualificationContract(unittest.TestCase):
 def test_fixed_schedule_qualification(self):
  # Extract pure functions without importing model/simulator or CUDA on the Mac.
  scope={};tree=ast.parse((D/'beyond_alpha_stage.py').read_text())
  smooth=ast.parse('def smooth(u):\n u=min(1.,max(0.,u)); return u*u*(3.-2.*u)')
  exec(compile(smooth,'smooth','exec'),scope)
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixed_progress')
  exec(compile(ast.Module(body=[node],type_ignores=[]),'schedule','exec'),scope)
  f=scope['fixed_progress'];self.assertEqual([f(t) for t in [3,3.8,5.8,7.8,8.8,9.55,10.8]],[0,0,.5,1,1,0,0])
  values=[f(3+i*.02) for i in range(391)]
  self.assertTrue(all(0<=v<=1 for v in values));self.assertLess(max(abs(a-b) for a,b in zip(values,values[1:])),.041)
 def test_student_has_no_privileged_input(self):
  tree=ast.parse((D/'beyond_alpha_stage.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='student_input']
  self.assertEqual(len(calls),1)
  self.assertEqual([ast.unparse(a) for a in calls[0].args],['model','obs','[hist]','env.keys'])
  branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="controller == 'A2'")
  text='\n'.join(ast.unparse(n) for n in branch.body)
  self.assertNotIn('fixed_progress',text);self.assertNotIn('anchor',text)
 def test_precommitted_bounded_comparison(self):
  p=json.loads((D/'beyond_alpha_protocol.json').read_text());c=p['conditions'][0]
  self.assertEqual(c['controllers'],['standing','A2','forward_clock','raise_clock'])
  self.assertEqual(c['delta_xyz_m'],[0,0,.30]);self.assertAlmostEqual(c['end']-c['onset']-c['duration'],2.)
  self.assertEqual(len(p['raised']['sha256']),64);self.assertNotEqual(p['raised']['sha256'],'TO_FILL')
if __name__=='__main__':unittest.main()
