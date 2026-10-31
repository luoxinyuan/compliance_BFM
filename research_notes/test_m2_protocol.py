import ast,json,unittest
from pathlib import Path
from m2_rule import infer_coordinate
P=Path(__file__).parent
class ContractTests(unittest.TestCase):
 def test_no_contact_onset_drives_initialization(self):
  x=json.loads((P/'m2_protocol.json').read_text());p=x['preparation'];c=x['contact']
  self.assertGreaterEqual(p['control_ready_s']-(p['prefix_s']+p['entry_s']),p['real_force_history_s'])
  self.assertGreaterEqual(c['onset']-(p['control_ready_s']+p['handoff_s']),1.25-1e-8)
  self.assertAlmostEqual(c['end']-c['release'],6)
  self.assertTrue(x['no_training'])
 def test_continuous_proprio_coordinate(self):
  refs=[[0.]*7,[1.]*7,[2.]*7]
  for q,u in [(0,0),(.5,.25),(1,.5),(1.5,.75),(2,1)]:self.assertAlmostEqual(infer_coordinate([q]*7,refs)['u'],u)
  self.assertAlmostEqual(infer_coordinate([.999]*7,refs)['u'],.4995)
 def test_frozen_methods_and_real_contacts(self):
  x=json.loads((P/'m2_protocol.json').read_text());self.assertEqual(x['checkpoints']['distill'],'c9036c46a0559c57fc2897419f06c66ef6113cb0fe4d57762df9a59c8b9e1ac7');self.assertEqual(x['configuration']['u'],.5);self.assertEqual(x['contact']['cap_n'],5)
 def test_no_hidden_input_or_fitting(self):
  s=(P/'m2_stage.py').read_text();ast.parse(s)
  self.assertIn('student_raw=nets[controller](deployable_h)',s)
  self.assertNotIn('torch.optim',s);self.assertNotIn('.backward(',s)
  self.assertIn('handoff_progress(t,ready)',s)
if __name__=='__main__':unittest.main()
