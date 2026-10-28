"""Replay gate and direction instrumentation tests; no model or simulator."""
import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import right_wrist_replay_minusx as p


def fixture():
    physics=[dict(time=(i+1)*.005,force_time=i*.005,qpos=[0.,0.,.8],qvel=[0.,0.,0.],
        torque=[0.]*29,force=[0.]*3,active=False) for i in range(1200)]
    actor=[dict(time=i*.02,action=[0.]*29,obs={'x':[[0.]]},z=[[1.]],
        bfm_evaluated=True,source_index=i) for i in range(300)]
    baseline=dict(initial={'qpos':[0.,0.,.8]},physics=physics,actor=actor,physics_dt=.005)
    replay=copy.deepcopy(baseline)
    for row in replay['actor']:row['bfm_evaluated']=False
    return baseline,replay


class ReplayMinusXTests(unittest.TestCase):
    def test_full_exact_gate(self):
        a,b=fixture();gate=p.require_replay_gate(a,b)
        self.assertTrue(gate['passed']);self.assertEqual(gate['max_abs_delta'],dict(qpos=0.,qvel=0.,torque=0.,action=0.))
    def test_gate_rejects_truncated_matching_prefix(self):
        a,b=fixture();b['physics'].pop();b['actor'].pop()
        with self.assertRaises(ValueError):p.require_replay_gate(a,b)
    def test_gate_rejects_late_state_and_pd_divergence(self):
        for key in ['qpos','qvel','torque']:
            a,b=fixture();b['physics'][1199][key][0]=1e-15
            gate=p.trace_match(a,b);self.assertFalse(gate['passed'])
            self.assertIn('/physics/1199/'+key,gate['first_difference'])
    def test_gate_rejects_action_observation_timing_changes(self):
        for key in ['time','action','obs']:
            a,b=fixture()
            if key=='time':b['actor'][150][key]+=.02
            elif key=='action':b['actor'][150][key][0]=.0001
            else:b['actor'][150][key]['x'][0][0]=.0001
            with self.assertRaises(ValueError):p.require_replay_gate(a,b)
    def test_gate_rejects_live_actor_or_repeated_source_index(self):
        a,b=fixture();b['actor'][-1]['bfm_evaluated']=True
        with self.assertRaises(ValueError):p.require_replay_gate(a,b)
        a,b=fixture();b['actor'][-1]['source_index']=298
        with self.assertRaises(ValueError):p.require_replay_gate(a,b)
    def test_gate_rejects_tether_or_nonzero_force(self):
        for key,value in [('active',True),('force',[1e-15,0.,0.])]:
            a,b=fixture();b['physics'][10][key]=value
            with self.assertRaises(ValueError):p.require_replay_gate(a,b)
    def test_minus_x_velocity_sign_and_endpoints(self):
        for t in [0.,1.,3.,4.5,6.]:self.assertEqual(p.audit_anchor_velocity(t,[-.03,0,0]),[0,0,0])
        self.assertAlmostEqual(p.audit_anchor_velocity(2.,[-.03,0,0])[0],-.0225)
        for t in [1.1,1.5,2.8]:
            self.assertEqual(p.audit_anchor_velocity(t,[-.03,0,0])[0],-p.audit_anchor_velocity(t,[.03,0,0])[0])
    def test_replay_original_lookup_rejects_wrong_time_or_exhaustion(self):
        a,_=fixture()
        self.assertEqual(p.replay_row(a['actor'],299,5.98),[0.]*29)
        with self.assertRaises(IndexError):p.replay_row(a['actor'],300,6.)
        with self.assertRaises(ValueError):p.replay_row(a['actor'],10,.22)
    def test_no_numerical_imports(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('mujoco',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
