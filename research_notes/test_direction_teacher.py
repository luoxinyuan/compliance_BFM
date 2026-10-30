"""Parity tests for the original teacher and the saved-state probe contract."""
import ast,json,unittest
from pathlib import Path
HERE=Path(__file__).parent
ORIGINAL=HERE/'ccl02a_teacher.py'
if not ORIGINAL.exists():ORIGINAL=HERE.parent/'issue21-bidirectional/ccl02a_teacher.py'
SOURCE=HERE/'contact_direction_teacher.py'
def tree(path):return ast.parse(path.read_text())
def named(path,name):return next(n for n in ast.walk(tree(path)) if isinstance(n,ast.FunctionDef) and n.name==name)
def decision(path):
 s=path.read_text();a=s.index('                    desired=np.zeros(7);');b=s.index('                    decisions.append(',a)
 import textwrap
 return ast.dump(ast.parse(textwrap.dedent(s[a:b])))
class Tests(unittest.TestCase):
 def test_teacher_decision_exact_original(self):
  self.assertEqual(decision(ORIGINAL),decision(SOURCE))
  old=named(ORIGINAL,'act');new=named(SOURCE,'act')
  part=lambda n:next(x for x in n.body if isinstance(x,ast.If) and 'release_start' in ast.unparse(x.test))
  self.assertEqual(ast.dump(part(old)),ast.dump(part(new)))
 def test_native_physics_and_snapshot_exact(self):
  old=HERE/'contact_direction_probe.py'
  if not old.exists():old=HERE.parent/'issue21-direction-probe/contact_direction_probe.py'
  for name in ['sample_native_contacts','_physics_step','snapshot']:
   self.assertEqual(ast.dump(named(old,name)),ast.dump(named(SOURCE,name)))
 def test_finite_teacher_only_no_student_or_optimizer(self):
  p=json.loads((HERE/'direction_teacher_protocol.json').read_text())
  self.assertEqual(p['modes'],['teacher']);self.assertEqual(p['directions'],[1,-1]);self.assertEqual(p['branch_seconds'],1.)
  self.assertEqual(set(p['reference_sha256']),{'common-state.json','plus-baseline.json','minus-baseline.json'})
  for n in ast.walk(tree(SOURCE)):
   if isinstance(n,ast.Call):
    name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
    self.assertNotIn(name,['backward','fit','fit_balanced','Adam','SGD','Projector','deploy_input','constrained'])
  self.assertIn('if state!=common:',SOURCE.read_text())
 def test_bounded_teacher_mirror_and_fade(self):
  try:import numpy as np
  except ImportError:self.skipTest('numpy unavailable locally; execute numeric unit test in existing project CPU environment')
  namespace={};module=tree(ORIGINAL)
  for n in module.body:
   if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['arm_ik','ReleaseFade']:
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(ORIGINAL),'exec'),namespace)
  jac=np.array([[.3,0,0,.1,0,0,0],[0,.2,0,0,.1,0,0],[0,0,.25,0,0,.05,0]])
  outputs=[]
  for sign in [1,-1]:
   error=np.array([sign*.035,.002,-.001]);delta=namespace['arm_ik'](jac,error)
   self.assertLessEqual(abs(delta).max(),.04);self.assertLessEqual(np.linalg.norm(delta),.08000001)
   self.assertGreater((jac@delta)@error,0);self.assertGreater(sign*(jac@delta)[0],0);outputs.append(delta)
  self.assertAlmostEqual(outputs[0][0],-outputs[1][0])
  fade=namespace['ReleaseFade']();desired=outputs[0]
  np.testing.assert_array_equal(fade.update(3.,True,desired),desired)
  np.testing.assert_allclose(fade.update(4.5,False),desired)
  np.testing.assert_allclose(fade.value(4.6),desired*.5,atol=1e-12)
  np.testing.assert_array_equal(fade.value(4.7),np.zeros(7))
if __name__=='__main__':unittest.main()
