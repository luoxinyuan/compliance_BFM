import ast,json,unittest
from pathlib import Path
class Contract(unittest.TestCase):
 def test_sources_parse(self):
  for p in Path(__file__).parent.glob('m8_*.py'):ast.parse(p.read_text())
 def test_no_live_pose_reset_from_goal(self):
  s=(Path(__file__).parent/'m8_stage.py').read_text();self.assertIn('goal_observation(env,np.asarray(row[',s);self.assertNotIn('env.sim.data.qpos[:]=',s)
 def test_budget_and_physics(self):
  x=json.loads((Path(__file__).parent/'m8_protocol.json').read_text());self.assertEqual(x['contact']['max_force'],5);self.assertEqual(x['qualification']['candidates'],2);self.assertEqual(x['contact']['semantics'],'maintained-contact guidance, bilateral original Tether, not unilateral pressure');self.assertLessEqual(x['resources']['compute_cumulative_wall_seconds'],7200)
class Feedback(unittest.TestCase):
 def test_signed_force_feedback_and_stop(self):
  from m8_controller import BodyFeedback
  c=json.loads((Path(__file__).parent/'m8_protocol.json').read_text())['body_feedback'];f=BodyFeedback(c)
  for _ in range(100):f.step([0,0,-2])
  self.assertGreater(f.alpha,.5);held=f.alpha
  for _ in range(20):f.step([2,0,0])
  self.assertAlmostEqual(f.alpha,held)
  for _ in range(150):f.step([0,0,0])
  self.assertEqual(f.alpha,0)
 def test_negative_control_no_progress(self):
  from m8_controller import BodyFeedback
  c=json.loads((Path(__file__).parent/'m8_protocol.json').read_text())['body_feedback'];f=BodyFeedback(c)
  for _ in range(500):self.assertEqual(f.step([0,0,0]),0)
class LatentInterface(unittest.TestCase):
 def test_endpoints_and_gradients(self):
  import torch
  from m8_learning import blend,policy_latent
  class Model:
   def project_z(self,z):return 16*torch.nn.functional.normalize(z,dim=-1)
  torch.manual_seed(0);m=Model();ends=[m.project_z(torch.randn(1,256)) for _ in range(3)]
  self.assertTrue(torch.equal(blend(ends[0],ends[1],0),ends[0]))
  self.assertTrue(torch.equal(blend(ends[0],ends[1],1),ends[1]))
  raw=torch.zeros(4,258,requires_grad=True)
  with torch.no_grad():raw[:,0]=.3;raw[:,1]=.4
  z=policy_latent(m,ends,raw,torch.ones(4));self.assertLess(float((z.norm(dim=-1)-16).abs().max()),1e-4)
  z[:,0].sum().backward();self.assertTrue(torch.isfinite(raw.grad).all());self.assertGreater(float(raw.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
