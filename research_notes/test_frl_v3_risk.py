import unittest
from pathlib import Path
from frl_v3_risk import event,h
class RiskContract(unittest.TestCase):
    def test_zero_excess_exact_zero(self):
        self.assertEqual(event([{'stage_loaded_slip_mps':0.},{'stage_loaded_slip_mps':.45}]),0.)
    def test_event_not_time_integrated(self):
        self.assertAlmostEqual(event([{'stage_loaded_slip_mps':.675}]),.0825)
        self.assertAlmostEqual(event([{'stage_loaded_slip_mps':.9}]),.33)
        self.assertAlmostEqual(event([{'stage_loaded_slip_mps':1.35}]),.99)
    def test_step_max_and_zero_lambda(self):
        rows=[{'stage_loaded_slip_mps':v} for v in [.2,.3,.9,.1]]
        self.assertAlmostEqual(event(rows),.33);self.assertEqual(event(rows,0.),0.)
    def test_continuous_quadratic_linear_join(self):
        self.assertEqual(h(0),0);self.assertEqual(h(1),1);self.assertEqual(h(2),3)
        self.assertAlmostEqual((h(1+1e-6)-h(1-1e-6))/(2e-6),2,places=5)
    def test_archive_path(self):
        root=Path('/test/archive');owner='B64';parent='123456789012abcdefghijklmnopqrstuvwxyz'
        p=root/f'runs/frl-formal-B-01/dev-{dict(BC500=0,B64=64,B128=128)[owner]}'/(parent[:12]+'-main.json.gz')
        self.assertEqual(str(p),'/test/archive/runs/frl-formal-B-01/dev-64/123456789012-main.json.gz')
if __name__=='__main__':unittest.main()
