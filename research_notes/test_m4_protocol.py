import ast,json,unittest
from pathlib import Path
P=Path(__file__).parent
class Protocol(unittest.TestCase):
 def test_syntax(self):
  for f in P.glob('m4_*.py'):ast.parse(f.read_text())
 def test_schedule(self):
  p=json.loads((P/'m4_protocol.json').read_text());self.assertEqual(len(p['schedule']['continuous']),10);self.assertEqual(len(p['schedule']['recovery']),6);self.assertEqual(p['resources']['compute_cumulative_wall_seconds'],3600)
  self.assertEqual(p['conditional_training']['rounds']*p['conditional_training']['updates_per_round'],1500)
 def test_no_shared_changes(self):
  s=(P/'m4_stage.py').read_text();self.assertIn("configuration=='u500'",s);self.assertIn("assert max(parity.values())==0",s);self.assertIn('project(model,obs,oracle_command',s);self.assertNotIn('opt.step()',s)
 def test_source_inheritance(self):
  s=(P/'m4_physics.py').read_text();self.assertIn("from humanoidverse.force_compliance.environment import ComplianceEnv",s);self.assertIn("assert torch.cuda.device_count()==1",s);self.assertIn('original_bfm_unchanged=True',s)
if __name__=='__main__':unittest.main()
