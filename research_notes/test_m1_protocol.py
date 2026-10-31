import ast,json,unittest
from pathlib import Path
D=Path(__file__).parent
class Contract(unittest.TestCase):
 def test_timing_and_heldout(self):
  p=json.loads((D/'m1_protocol.json').read_text());c=p['contact'];self.assertAlmostEqual(c['release']-c['onset'],5);self.assertAlmostEqual(c['end']-c['release'],2)
  for k in ['onset','release','end']:self.assertAlmostEqual(c[k]*50,round(c[k]*50))
  self.assertNotEqual(c['onset'],p['learning']['heldout']['onset']);self.assertNotEqual(c['delta_xyz'],p['learning']['heldout']['delta_xyz'])
 def test_causal_network_inputs(self):
  t=ast.parse((D/'m1_stage.py').read_text());calls=[n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['deploy_input','student_input']]
  self.assertEqual(len(calls),2)
  for call in calls:self.assertFalse(any(x in ast.unparse(call) for x in ['anchor','target_z','configuration','onset']))
 def test_pairing_ignores_only_unused_reward(self):
  import copy,hashlib
  t=ast.parse((D/'m1_stage.py').read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='physical_state_digest')
  scope=dict(copy=copy,hashlib=hashlib,json=json);exec(compile(ast.Module(body=[f],type_ignores=[]),'test','exec'),scope);digest=scope['physical_state_digest']
  a={'base_tensors':{'rew_buf':[0.0],'actions':[0.]},'qpos':[1.]};b=copy.deepcopy(a);b['base_tensors']['rew_buf']=[1e-9]
  self.assertEqual(digest(a),digest(b));self.assertIn('rew_buf',a['base_tensors'])
  b['qpos']=[1.01];self.assertNotEqual(digest(a),digest(b));b=copy.deepcopy(a);b['base_tensors']['actions']=[.01];self.assertNotEqual(digest(a),digest(b))
 def test_measurements_do_not_forward_live_physics(self):
  t=ast.parse((D/'m1_stage.py').read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='cached_point')
  self.assertNotIn('mj_forward',ast.unparse(f));self.assertNotIn('point_state',ast.unparse(f))
  calls=[n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='point_state']
  self.assertEqual(len(calls),1) # Original once-only anchor initialization; no logging side effect.
 def test_heldout_physics_onset_is_not_rounded(self):
  p=json.loads((D/'m1_protocol.json').read_text())['learning']['heldout'];self.assertEqual(p['onset'],6.55)
  self.assertAlmostEqual(p['onset']*200,round(p['onset']*200));self.assertNotEqual(p['onset']*50,round(p['onset']*50))
  t=ast.parse((D/'m1_physics.py').read_text());f=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='_physics_step');body=ast.unparse(f)
  self.assertLess(body.index('self.m1_anchor_hook(force_time)'),body.index('super()._physics_step()'))
 def test_causal_ready_independent_of_sealed_onset(self):
  p=json.loads((D/'m1_final_protocol.json').read_text())['causal_execution_correction'];self.assertEqual(p['fixed_ready_s'],6.2);self.assertNotEqual(p['fixed_ready_s'],6.55)
  t=ast.parse((D/'m1_stage.py').read_text());calls=[n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='command_handoff']
  self.assertTrue(all(isinstance(n.args[-1],ast.Name) and n.args[-1].id=='control_ready' for n in calls))
 def test_entry_is_bumpless(self):
  t=ast.parse((D/'m1_stage.py').read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='enter_progress')
  def smooth(x):x=min(1,max(0,x));return x*x*(3-2*x)
  s={'smooth':smooth};exec(compile(ast.Module(body=[f],type_ignores=[]),'test','exec'),s)
  self.assertEqual([s['enter_progress'](x) for x in [0,3,3.375,3.75,6.2]],[0,0,.5,1,1])
if __name__=='__main__':unittest.main()
