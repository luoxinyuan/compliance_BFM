import ast,json,unittest
from pathlib import Path
from m7_partner import segment,install,uninstall
import numpy as np
from types import SimpleNamespace
P=json.loads(Path(__file__).with_name('m7_protocol.json').read_text())
class Test(unittest.TestCase):
 def test_segments(self):
  s=P['conditions']['dev_short'];tr=s['transition']
  self.assertTrue(segment(s,9.9)[4]);self.assertFalse(segment(s,10.1)[4]);self.assertTrue(segment(s,10.2)[4])
  self.assertEqual(segment(s,10.2)[:2],(.15,-.185))
 def test_original_force_implementation_preserved(self):
  class Fake:
   def measure(self):
    c=self.cfg;u=np.clip((self.time-c.activation_time)/c.ramp_duration,0,1);self.anchor=self.origin+(3*u*u-2*u**3)*self.delta;self.active=c.enabled and c.activation_time<=self.time<c.release_time
    return self.anchor,self.active
  s=P['conditions']['dev_short'];x=Fake();x.origin=np.zeros(3);x.delta=np.ones(3);x.cfg=SimpleNamespace(activation_time=5.,ramp_duration=4.,enabled=True,release_time=s['release']);x.time=0
  original=x.measure;install(x,s,np.zeros(3))
  for t,expected in [(5,0),(9,.15),(10.2,.15),(14.2,-.035)]:
   x.time=t;p,_=x.measure();self.assertAlmostEqual(p[0],expected);self.assertEqual(x.cfg.activation_time,5.)
  x.time=10.1;self.assertFalse(x.measure()[1]);x.time=s['release'];self.assertFalse(x.measure()[1]);uninstall(x);self.assertEqual(x.measure,original)
 def test_sealed(self):
  for phase,jobs in P['schedule'].items():
   if phase!='final':self.assertFalse(any(c.startswith('sealed') for c,_ in jobs))
 def test_signed_rule(self):
  from m7_gate import DirectionalGate
  from contact_gate import ContactGate
  cfg=dict(dt=.02,filter_samples=3,entry_forward_force_n=.5,entry_samples=4,exit_force_norm_n=.2,exit_samples=20,entry_transition_s=4.,return_transition_s=.75)
  a,b=DirectionalGate(cfg),ContactGate(cfg)
  for _ in range(230):self.assertEqual(a.step([2.,0,0])['alpha'],b.step([2.,0,0])['alpha'])
  values=[a.step([-2.,0,0])['alpha'] for _ in range(230)]
  self.assertEqual(values[-1],0.);self.assertTrue(all(y<=x+1e-10 for x,y in zip(values[6:],values[7:])))
  for _ in range(100):self.assertEqual(a.step([0.,0,0])['alpha'],0.)
 def test_atomic_ledger_concurrent_reader(self):
  import tempfile,threading
  from m7_io import write
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'ledger.json';write(p,dict(n=0,payload='x'*10000));errors=[];stop=threading.Event()
   def reader():
    while not stop.is_set():
     try:
      x=json.loads(p.read_text());assert x['payload']=='x'*10000
     except Exception as e:errors.append(str(e))
   t=threading.Thread(target=reader);t.start()
   try:
    for i in range(100):write(p,dict(n=i,payload='x'*10000))
   finally:stop.set();t.join()
   self.assertEqual(errors,[]);self.assertEqual(json.loads(p.read_text())['n'],99)
 def test_counterfactual_only_changes_future_partner(self):
  import copy
  a=copy.deepcopy(P['conditions']['sealed_mid']);b=P['conditions']['posthoc_mid_hold']
  a['transition']['reverse_target']=a['delta_xyz'][0]
  self.assertEqual(a,b);self.assertEqual(segment(b,b['transition']['restart'])[1],0.)
  self.assertEqual(P['schedule']['counterfactual'],[['posthoc_mid_hold','distill']])
 def test_syntax(self):
  for f in Path(__file__).parent.glob('m7_*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
