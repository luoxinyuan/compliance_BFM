import ast,json,unittest
from pathlib import Path
import numpy as np
from humanoidverse.force_compliance.interaction import spring_force
HERE=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_mirrored_partner_same_preload(self):
  origin=np.array([.1,-.2,.85]);v=np.array([.001,.002,-.003]);K,D,cap=80,8,5
  f0=spring_force(origin,origin,np.zeros(3),v,K,D,cap)
  np.testing.assert_allclose(f0,-D*v)
  elapsed=.5;u=elapsed/2;s=3*u*u-2*u**3;ds=6*u*(1-u)
  forces=[]
  for sign in [1,-1]:
   delta=np.array([sign*.035,0,0]);forces.append(spring_force(origin+s*delta,origin,delta*ds/2,v,K,D,cap))
  self.assertGreater(forces[0][0],0);self.assertLess(forces[1][0],0)
  np.testing.assert_allclose((forces[0]+forces[1])/2,-D*v)
 def test_eval_only_and_pinned_checkpoint(self):
  p=json.loads((HERE/'direction_probe_protocol.json').read_text());self.assertEqual(p['branch_seconds'],1.)
  self.assertEqual(len(p['directions'])*len(p['modes']),6)
  self.assertEqual(p['student_sha256'],'a5daf853276d55f5b8c099df4f48917f572a32c48e548d2c90f39e593d4a809d')
  tree=ast.parse((HERE/'contact_direction_probe.py').read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.Call):
    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
    self.assertNotIn(name,['backward','fit','fit_balanced','Adam','SGD','point_state'])
 def test_native_physics_hook_preserved(self):
  old=ast.parse((HERE/'ccl02a_teacher.py').read_text());new=ast.parse((HERE/'contact_direction_probe.py').read_text())
  find=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sample_native_contacts')
  self.assertEqual(ast.dump(find(old)),ast.dump(find(new)))
 def test_explicit_force_ablation_preserves_proprio(self):
  source=(HERE/'contact_direction_probe.py').read_text()
  self.assertIn("sensor_gate.update(float(env.tether.time),env.tether.force)",source)
  self.assertIn("if label=='student-force-zero':fed[:,-9:]=0.",source)
  self.assertIn('if state!=common:',source)
if __name__=='__main__':unittest.main()
