import ast,hashlib,json,unittest
from pathlib import Path
import numpy as np
from wholebody_physics import smooth
from ccl02a_teacher import arm_ik,ReleaseFade
class Tests(unittest.TestCase):
 def test_smooth_cadence(self):
  self.assertEqual([smooth(x) for x in [-1,0,.5,1,2]],[0,0,.5,1,1])
  self.assertAlmostEqual(smooth(.000001),3e-12,places=16)
 def test_original_bound_for_ten_joints(self):
  rng=np.random.default_rng(0)
  for _ in range(100):
   q=arm_ik(rng.normal(size=(3,10)),rng.normal(size=3))
   self.assertLessEqual(abs(q).max(),.04+1e-12);self.assertLessEqual(np.linalg.norm(q),.08+1e-12)
 def test_original_fade_ten_joints(self):
  f=ReleaseFade();f.last=np.zeros(10);self.assertEqual(f.update(3,False).shape,(10,))
  f.update(7.9,True,np.ones(10)*.01);self.assertTrue(np.allclose(f.update(8,False),.01));self.assertTrue(np.allclose(f.value(8.1),.005));self.assertTrue(np.allclose(f.value(8.2),0))
 def test_frozen_protocol(self):
  p=json.loads((Path(__file__).parent/'wholebody_protocol.json').read_text())
  self.assertEqual(p['labels'],['baseline','arm','goal_bfm','waist_dls']);self.assertEqual((p['K'],p['D'],p['max_force_n']),(80,8,5));self.assertEqual(p['policy_updates'],0)
 def test_original_teacher_identity(self):
  self.assertEqual(hashlib.sha256((Path(__file__).parent/'ccl02a_teacher.py').read_bytes()).hexdigest(),'50d44c732550c60000ee9437955896d6267c81d5bb3a028b0761ca05309c7c00')
 def test_no_learning_calls(self):
  for name in ['wholebody_physics.py','wholebody_run.py','wholebody_geometry.py']:
   text=(Path(__file__).parent/name).read_text();ast.parse(text)
   for token in ['torch.optim','.backward(','fit_repair(','fit_balanced(','Projector(']:self.assertNotIn(token,text)
if __name__=='__main__':unittest.main()
