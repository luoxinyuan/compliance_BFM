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
  h=ContactHistory(2);self.assertEqual(h.features(np.eye(3)).sum(),0);h.append([0,0,0],'torso_link',[0,0,.25],np.eye(3));self.assertIsNone(h.rows[-1]['link']);h.append([1,0,0],'torso_link',[0,0,.25],np.eye(3),False);self.assertFalse(h.rows[-1]['contact_detected'])
 def test_source_budget(self):
  p=Path(__file__).parent
  for f in p.glob('m9_*.py'):ast.parse(f.read_text())
  x=json.loads((p/'m9_protocol.json').read_text());self.assertEqual(len(x['reward_tasks']),5);self.assertEqual(x['resources']['compute_cumulative_wall_seconds'],7200)
if __name__=='__main__':unittest.main()
