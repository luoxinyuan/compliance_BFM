import json,unittest
from pathlib import Path
from repair_data_contract import target_fields,phase_indices

class RepairContract(unittest.TestCase):
 def test_visited_label_is_teacher_not_failed_student(self):
  row=dict(alpha=.9,action=[9],q_PD=[9],shadow_alpha=.2,shadow_action=[2],shadow_q_PD=[2])
  self.assertEqual(target_fields(row,'shadow'),(.2,[2],[2]))
  self.assertEqual(target_fields(row,'teacher'),(.9,[9],[9]))
 def test_missing_shadow_fails_instead_of_imitation(self):
  with self.assertRaises(KeyError):target_fields(dict(alpha=.9,action=[9],q_PD=[9]),'shadow')
 def test_phase_boundaries_partition_without_future(self):
  cond=dict(onset=5.4,disconnect_start=7.8,disconnect_end=8.)
  controls=[dict(time=t) for t in [3.,5.4,7.78,7.8,7.98,8.,8.98,9.,12.38]]
  g=phase_indices(controls,cond)
  self.assertEqual(g,dict(active_before=[1,2],gap=[3,4],reconnect1s=[5,6],rest=[0,7,8]))
 def test_sealed_conditions_not_fit_sources_and_balanced(self):
  p=json.loads((Path(__file__).parent/'alpha_repair_protocol.json').read_text())
  self.assertEqual(sum(s['batch_count'] for s in p['training_sources']),128)
  self.assertEqual(sum(s['batch_count'] for s in p['training_sources'] if s['kind']=='normal'),64)
  self.assertEqual(sum(len(c['controllers']) for c in p['conditions']),24)
  for cond in p['conditions']:
   if cond['split']=='sealed_new_gap':
    self.assertTrue(all(cond['name'] not in s['path'] for s in p['training_sources']))
  self.assertEqual(p['optimization']['updates'],2000)
  self.assertEqual(p['initialization']['sha256'],p['old_checkpoint_sha256']['A'])

if __name__=='__main__':unittest.main()
