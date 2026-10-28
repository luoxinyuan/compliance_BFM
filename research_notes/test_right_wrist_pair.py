"""Dependency-free tests; no model, simulator, network, or server imports."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import right_wrist_pair as pilot


def fixture():
    records=[]
    for t in [.5,1.5,2.5,3.75,4.25,5.,6.]:
        records.append(dict(time=t,force_time=t-.005,force=[0.,0.,0.],qpos=[0.,0.,.8],
            wrist=[0.,-.2,.6],root=[0.,0.,.8],wrist_relative_root=[0.,-.2,-.2],
            wrist_pelvis_frame=[0.,-.2,-.2],anchor_error=0.,tilt_deg=0.,
            root_rpy_deg=[0.,0.,0.],support=[True,True],foot_displacement=0.,
            foot_normal_forces=[150.,150.],torque_saturation=0.))
    return dict(initial={'qpos':[0.,0.,.8]},physics=records,
        actor=[dict(time=r['time']-.02,action=[0.]*29) for r in records],
        physics_dt=.005,metrics={'fall_rate':0.})


class PilotTests(unittest.TestCase):
    def test_metric_pair_signs_and_feedback(self):
        baseline=fixture();tether=copy.deepcopy(baseline)
        for p,a in zip(tether['physics'],tether['actor']):
            if 1.<=p['force_time']<4.5:
                p['force']=[2.,0.,0.];p['wrist'][0]+=.01;p['wrist_relative_root'][0]+=.01;a['action'][0]=.1
        result=pilot.analyze(baseline,tether)
        self.assertTrue(result['pair_valid'])
        delta=result['paired_phase_deltas']['hold_late']
        self.assertAlmostEqual(delta['wrist_paired_mean_delta_m'][0],.01)
        self.assertAlmostEqual(delta['action_rms_delta'],.1/(29**.5))
        self.assertEqual(result['episodes']['tether']['force_peak_n'],2.)
        self.assertEqual(result['paired_phase_deltas']['released']['action_rms_delta'],0.)
    def test_mismatched_initial_and_precontact_are_rejected(self):
        b=fixture();t=copy.deepcopy(b);t['initial']['qpos'][0]=1.
        self.assertFalse(pilot.analyze(b,t)['pair_valid'])
        t=copy.deepcopy(b);t['physics'][0]['qpos'][0]=.0001
        self.assertFalse(pilot.analyze(b,t)['pair_valid'])
        t=copy.deepcopy(b);t['actor'][0]['action'][0]=.001
        self.assertFalse(pilot.analyze(b,t)['pair_valid'])
    def test_fall_and_truncation_are_not_hidden(self):
        b=fixture();t=copy.deepcopy(b);t['metrics']['fall_rate']=1.;t['physics'].pop()
        r=pilot.analyze(b,t)
        self.assertFalse(r['pair_valid']);self.assertTrue(r['episodes']['tether']['failed'])
    def test_protocol_is_frozen_small_cpu_pair(self):
        self.assertEqual(pilot.OVERRIDES,dict(direction=[1.,0.,0.],displacement_min=.03,
            displacement_max=.03,max_force=5.,activation_time=1.,anchor_delay=0.,release_time=4.5))
        tree=ast.parse((HERE/'right_wrist_pair.py').read_text())
        evaluate=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='evaluate']
        self.assertEqual(len(evaluate),1);self.assertIs(evaluate[0].args[2].value,True)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('mujoco',sys.modules)
    def test_all_delivered_python_parses(self):
        for name in ['right_wrist_pair.py','right_wrist_supervisor.py','test_right_wrist_pair.py',
                     'verify_right_wrist_pair.py','plot_right_wrist_pair.py']:
            ast.parse((HERE/name).read_text())

if __name__=='__main__':unittest.main(verbosity=2)
