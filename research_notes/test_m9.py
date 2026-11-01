import unittest,ast,json
from pathlib import Path
class History(unittest.TestCase):
 def test_link_identity_preserved(self):
  import numpy as np
  from m9_history import ContactHistory
  h=ContactHistory(4);h.append([1,0,0],'right_wrist_yaw_link',[0,0,0],np.eye(3));h.append([0,0,-1],'torso_link',[0,0,.25],np.eye(3));x=h.features(np.eye(3));self.assertEqual(x[-2,3],1);self.assertEqual(x[-2,4],0);self.assertEqual(x[-1,4],1);self.assertEqual(h.rows[0]['link'],'right_wrist_yaw_link')
 def test_rotating_world_coordinates(self):
  import numpy as np
  from m9_history import ContactHistory
  R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
  a=ContactHistory(2);b=ContactHistory(2);a.append([2,1,0],'right_wrist_yaw_link',[0,0,0],np.eye(3));b.append(R@np.array([2,1,0]),'right_wrist_yaw_link',[0,0,0],R);np.testing.assert_allclose(a.features(np.eye(3)),b.features(R))
 def test_empty_invalid_and_no_contact(self):
  import numpy as np
  from m9_history import ContactHistory
  h=ContactHistory(2);self.assertEqual(h.features(np.eye(3)).sum(),0);h.append([0,0,0],'torso_link',[0,0,.25],np.eye(3));self.assertIsNone(h.rows[-1]['link']);h.append([1,0,0],'torso_link',[0,0,.25],np.eye(3),False);self.assertFalse(h.rows[-1]['contact_detected']);np.testing.assert_array_equal(h.features(np.eye(3))[-1,:3],0)
 def test_source_budget(self):
  p=Path(__file__).parent
  for f in p.glob('m9_*.py'):ast.parse(f.read_text())
  x=json.loads((p/'m9_protocol.json').read_text());self.assertEqual(len(x['reward_tasks']),5);self.assertEqual(x['resources']['compute_cumulative_wall_seconds'],7200)
 def test_causal_native_feedback_and_release(self):
  import numpy as np
  from m9_contact import NativeFeedback
  f=NativeFeedback()
  for _ in range(400):s=f.step(np.zeros(3),np.eye(3))
  self.assertEqual(s['alpha'],0);self.assertEqual(s['phase'],0)
  for _ in range(100):s=f.step([3,0,0],np.eye(3))
  self.assertGreater(s['alpha'],.9);last=s['alpha']
  for _ in range(180):
   s=f.step(np.zeros(3),np.eye(3));self.assertLessEqual(abs(s['alpha']-last),.01800001);last=s['alpha']
  self.assertEqual(s['alpha'],0)
 def test_real_curved_partner_force(self):
  import numpy as np
  from types import SimpleNamespace
  from m9_contact import install_partner
  spec=dict(onset=3.8,ramp=7.,distance=.8,curvature=.4363323129985824,release=13,enabled=True)
  tether=SimpleNamespace(time=7.,cfg=SimpleNamespace(stiffness=80,damping=8,max_force=5),point_state=lambda:(np.zeros(3),np.zeros(3)))
  install_partner(tether,spec,np.zeros(3));_,f=tether.measure();self.assertGreater(f[0],0);self.assertGreater(f[1],0);self.assertAlmostEqual(np.linalg.norm(f),5)
  anchor=tether.anchor.copy();vel=tether.partner_velocity.copy();tether.time+=1e-5;tether.measure();np.testing.assert_allclose((tether.anchor-anchor)/1e-5,vel,rtol=1e-4,atol=1e-7)
  tether.time=14;_,f=tether.measure();np.testing.assert_array_equal(f,0)
if __name__=='__main__':unittest.main()
