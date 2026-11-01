import ast,json,unittest
from pathlib import Path
class Contract(unittest.TestCase):
 def test_sources_parse(self):
  for p in Path(__file__).parent.glob('m8_*.py'):ast.parse(p.read_text())
 def test_no_live_pose_reset_from_goal(self):
  s=(Path(__file__).parent/'m8_stage.py').read_text();self.assertIn('goal_observation(env,np.asarray(row[',s);self.assertNotIn('env.sim.data.qpos[:]=',s)
 def test_budget_and_physics(self):
  x=json.loads((Path(__file__).parent/'m8_protocol.json').read_text());self.assertEqual(x['contact']['max_force'],5);self.assertEqual(x['qualification']['candidates'],2);self.assertEqual(x['contact']['semantics'],'maintained-contact guidance, bilateral original Tether, not unilateral pressure');self.assertLessEqual(x['resources']['compute_cumulative_wall_seconds'],7200)
if __name__=='__main__':unittest.main()
