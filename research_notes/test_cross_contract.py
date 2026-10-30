import json,unittest
from pathlib import Path
class CrossContract(unittest.TestCase):
 def setUp(self):self.p=json.loads(Path(__file__).with_name('alpha_cross_protocol.json').read_text())
 def test_cross_and_seal(self):
  m=self.p['matrix'];self.assertEqual({(c['disconnect_start'],round(c['disconnect_end']-c['disconnect_start'],2)) for c in m},{(s,d) for s in [7.4,8.2] for d in [.3,.7]})
  final=[c for c in self.p['conditions'] if c['split']=='sealed_final']
  self.assertEqual(len(final),4)
  self.assertTrue(all(c['disconnect_start'] not in [7.4,7.8,8.2] for c in final))
 def test_limits_and_branch_counts(self):
  self.assertEqual(sum(len(c['controllers']) for c in self.p['diagnostic_conditions']),8)
  self.assertEqual(sum(len(c['controllers']) for c in self.p['conditions']),40)
  self.assertEqual(self.p['optimization']['updates'],2000)
  self.assertEqual(self.p['qualification']['alpha_shadow_rms_gap_and_first_1s_after_resume_max'],.05)
 def test_original_initialization_and_input(self):
  self.assertEqual(self.p['initialization']['sha256'],self.p['frozen_checkpoints']['A'])
  self.assertEqual(self.p['memory']['student_dim'],1425)
  self.assertEqual(self.p['memory']['force_samples'],320)
if __name__=='__main__':unittest.main()
