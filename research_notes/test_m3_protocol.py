import ast,json,unittest
from pathlib import Path
P=Path(__file__).parent
class ProtocolTests(unittest.TestCase):
 def test_sealed_body_and_timing(self):
  p=json.loads((P/'m3_protocol.json').read_text());h=p['held_out'];self.assertNotIn(h['u'],p['development_configurations']);self.assertNotEqual(h['onset'],p['contact']['onset']);self.assertIn(.5,p['development_configurations']);self.assertGreater(h['end']-h['release'],p['preparation']['real_force_history_s'])
 def test_independent_ready_and_real_history(self):
  p=json.loads((P/'m3_protocol.json').read_text());x=p['preparation'];self.assertGreater(x['control_ready_s']-x['prefix_s']-x['entry_s'],x['real_force_history_s']);self.assertGreaterEqual(p['contact']['onset']-x['control_ready_s']-x['handoff_s'],1.25)
 def test_no_fit_or_teacher_parameter_changes_in_qualification(self):
  s=(P/'m3_stage.py').read_text();ast.parse(s);self.assertNotIn('torch.optim',s);self.assertNotIn('.backward(',s);self.assertIn("assert stage=='final' or u!=protocol['held_out']['u']",s);self.assertIn('arm_ik(jac[:,vadr[arm]],env.tether.anchor-point)',s);self.assertIn('ReleaseFade()',s);self.assertIn('handoff_progress(t,ready)',s)
 def test_fixed_training_and_unseen_test(self):
  x=json.loads((P/'m3_training_plan.json').read_text());self.assertEqual(x['updates'],1500);self.assertEqual(len(x['teacher_sources']),3);self.assertNotIn('u625',' '.join(x['teacher_sources']));self.assertNotIn('u500__teacher.json.gz',x['teacher_sources']);self.assertEqual(x['heldout']['actual_end_control_tick'],26.16)
if __name__=='__main__':unittest.main()
