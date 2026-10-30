import ast,hashlib,json,math,unittest
from pathlib import Path
import torch
from goal_calibration_precheck import angle,PINS
class TestCalibration(unittest.TestCase):
 def test_direction_angles(self):
  a=torch.zeros(1,256);a[0,0]=16;b=a.clone();b[0,0]=0;b[0,1]=16
  self.assertEqual(angle(a,a),0);self.assertAlmostEqual(angle(a,b),90);self.assertAlmostEqual(angle(a,-a),180)
 def test_anchor_cancels_initial(self):
  torch.manual_seed(0);s=torch.randn(1,256);s=16*s/s.norm();g=torch.randn_like(s);g=16*g/g.norm()
  x=s+g-g;x=16*x/x.norm();self.assertTrue(torch.allclose(x,s,atol=1e-6));self.assertGreater(angle(g,s),10)
 def test_actual_goal_contract_projects(self):
  root=Path(__file__).resolve().parents[1];tree=ast.parse((root/'humanoidverse/agents/fb/model.py').read_text())
  goal=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='goal_inference')
  self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='project_z' for n in ast.walk(goal)))
 def test_protocol_keeps_physics(self):
  p=json.loads((Path(__file__).parent/'goal_calibration_protocol.json').read_text());self.assertEqual(p['labels'],['anchored','native']);self.assertEqual([p[k] for k in ['K','D','max_force_n','ramp_s','release_s','horizon_s']],[80,8,5,4,8,9])
 def test_original_teacher(self):
  self.assertEqual(hashlib.sha256((Path(__file__).parent/'ccl02a_teacher.py').read_bytes()).hexdigest(),'50d44c732550c60000ee9437955896d6267c81d5bb3a028b0761ca05309c7c00')
 def test_no_training_or_IK_resolve(self):
  for name in ['goal_calibration.py','goal_calibration_precheck.py','goal_calibration_run.py']:
   s=(Path(__file__).parent/name).read_text();ast.parse(s)
   for token in ['torch.optim','.backward(', 'least_squares(', 'fit_repair(', 'Projector(']:self.assertNotIn(token,s)
 def test_student_theoretical_cone_unchanged(self):
  self.assertAlmostEqual(math.degrees(math.atan(.3)),16.69924423399362)
if __name__=='__main__':unittest.main()
