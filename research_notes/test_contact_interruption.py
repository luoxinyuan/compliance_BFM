import json,unittest
from pathlib import Path
from types import SimpleNamespace
from contact_interruption import disconnected,apply_connection

class ConnectionTests(unittest.TestCase):
 def test_exact_ticks(self):
  for end,count in [(8.,40),(8.6,160)]:
   self.assertEqual(sum(disconnected(i*.005,(7.8,end)) for i in range(2480)),count)
   self.assertTrue(disconnected(7.8-1e-12,(7.8,end)));self.assertFalse(disconnected(end-1e-12,(7.8,end)))
 def test_original_force_enable_contract(self):
  tether=SimpleNamespace(cfg=SimpleNamespace(enabled=True),force=[1,2,3])
  self.assertFalse(apply_connection(tether,7.9,(7.8,8.)));self.assertFalse(tether.cfg.enabled)
  self.assertEqual(tether.force,[1,2,3]) # Does not fake a sensor value; physics recomputes it.
  self.assertTrue(apply_connection(tether,8.01,(7.8,8.)))
 def test_frozen_two_conditions(self):
  p=json.loads(Path(__file__).with_name('interruption_protocol.json').read_text())
  self.assertEqual(len(p['conditions'])*len(p['controllers']),8)
  self.assertEqual([round(c['disconnect_end']-c['disconnect_start'],1) for c in p['conditions']],[.2,.8])
  self.assertEqual(p['policy_updates'],0)
  self.assertEqual(len(set((c['onset'],c['duration'],c['end']) for c in p['conditions'])),1)

if __name__=='__main__':unittest.main()
