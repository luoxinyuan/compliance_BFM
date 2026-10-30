import inspect,json,unittest
from pathlib import Path
from contact_gate import ContactGate,smooth

C=json.loads(Path(__file__).with_name('contact_gate_protocol.json').read_text())

class GateTests(unittest.TestCase):
    def test_allowed_input_and_no_contact(self):
        self.assertEqual(list(inspect.signature(ContactGate.step).parameters),['self','measured_force_xyz'])
        g=ContactGate(C['gate'])
        for _ in range(1000):self.assertEqual(g.step([0,0,0])['alpha'],0)
        self.assertEqual(g.entries,0)
    def test_short_spike_and_wrong_direction(self):
        for f in ([5,0,0],[-5,0,0],[0,5,0]):
            g=ContactGate(C['gate']);g.step(f)
            for _ in range(100):g.step([0,0,0])
            self.assertEqual(g.entries,0)
    def test_drop_and_resume(self):
        g=ContactGate(C['gate'])
        for _ in range(80):g.step([2,0,0])
        for _ in range(3):g.step([0,0,0])
        before=g.alpha
        for _ in range(10):g.step([0,0,0])
        self.assertEqual(g.alpha,before);self.assertEqual(g.state,'active')
        for _ in range(30):g.step([2,0,0])
        self.assertGreater(g.alpha,before);self.assertEqual(g.entries,1)
    def test_release_and_bounds(self):
        g=ContactGate(C['gate']);values=[]
        for _ in range(300):values.append(g.step([3,1,0])['alpha'])
        self.assertEqual(g.alpha,1)
        for _ in range(100):values.append(g.step([0,0,0])['alpha'])
        self.assertEqual(g.alpha,0);self.assertEqual(g.exits,1)
        self.assertTrue(all(0<=a<=1 for a in values))
        self.assertLessEqual(max(abs(a-b) for a,b in zip(values,values[1:])),1.5*.02/.75+1e-9)
    def test_time_translation_invariance(self):
        stream=[[0,0,0]]*10+[[1,0,0]]*120+[[0,0,0]]*100
        a=ContactGate(C['gate']);b=ContactGate(C['gate'])
        for _ in range(137):b.step([0,0,0])
        self.assertEqual([a.step(f) for f in stream],[b.step(f) for f in stream])
    def test_protocol_fixed(self):
        self.assertEqual(len(C['conditions'])*len(C['controllers'])+1,16)
        self.assertEqual(C['conditions'][-1]['split'],'sealed_test')
        self.assertEqual(C['conditions'][-1]['onset'],4.6)
        self.assertTrue(C['no_training'] and C['no_parameter_tuning'])
        self.assertAlmostEqual(smooth(.5),.5)

if __name__=='__main__':unittest.main()
