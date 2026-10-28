"""No model/simulator imports; test anchor selection, replay contract and metrics."""
import ast,copy,json,math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import right_wrist_anchor as p


def episode(name,live):
    times=[.5,1.,1.005,2.,3.75,4.25,5.,6.]
    rows=[]
    for time in times:
        active=name!='no_force' and 1.<=time-.005<4.5
        force=[1.,2.,-2.] if active else [0.,0.,0.]
        rows.append(dict(time=time,force_time=time-.005,force=force,audit_force=force,active=active,
            wrist=[.2,-.1,.8],root=[0.,0.,.8],wrist_relative_root=[.2,-.1,0.],wrist_pelvis_frame=[.2,-.1,0.],
            anchor_error=.01,tilt_deg=2.,root_rpy_deg=[1.,1.,0.],qpos=[0.,0.,.8],support=[True,True],
            foot_displacement=.001,foot_normal_forces=[150.,150.],torque_saturation=0.,torque=[0.]*29,failed=False,
            spring_force=force,damping_force=[0.]*3,point_velocity_before=[0.]*3,position_error_before=[0.]*3,
            position_error_after=[.01,0.,0.],wrist_from_anchor_origin=[.02,0.,0.]))
    actors=[dict(time=i*.02,action=[0.]*29,bfm_evaluated=live) for i in range(len(times))]
    return dict(name=name,initial={'qpos':[0,0,.8]},physics=rows,actor=actors,metrics={'fall_rate':0},physics_dt=.005,trajectory=[[.2,-.1,.8],[.03,0,0]])


class AnchorReplayTests(unittest.TestCase):
    def test_exact_activation_state_not_next_sample(self):
        e=episode('no_force',True);e['physics'][2]['wrist']=[99,99,99]
        self.assertEqual(p.select_anchor(e),[.2,-.1,.8])
    def test_anchor_rejects_missing_duplicate_or_forced_sample(self):
        e=episode('no_force',True);e['physics'][1]['force']=[1,0,0]
        with self.assertRaises(ValueError):p.select_anchor(e)
        e=episode('no_force',True);e['physics'].pop(1)
        with self.assertRaises(ValueError):p.select_anchor(e)
        e=episode('no_force',True);e['physics'].append(e['physics'][1])
        with self.assertRaises(ValueError):p.select_anchor(e)
    def test_replay_exact_values_index_time_and_copy(self):
        rows=[{'time':0.,'action':[i/100 for i in range(29)]}]
        output=p.replay_row(rows,0,0.);self.assertEqual(output,rows[0]['action'])
        output[0]=999;self.assertNotEqual(output,rows[0]['action'])
        with self.assertRaises(IndexError):p.replay_row(rows,1,.02)
        with self.assertRaises(ValueError):p.replay_row(rows,0,.02)
    def test_replay_rejects_nonfinite_or_wrong_dimension(self):
        for action in [[0.]*28,[float('nan')]*29]:
            with self.assertRaises(ValueError):p.replay_row([{'time':0.,'action':action}],0,0.)
    def test_anchor_derivative_for_damping_audit(self):
        for t in [0.,1.,3.,4.5]:self.assertEqual(p.audit_anchor_velocity(t),[0.,0.,0.])
        self.assertAlmostEqual(p.audit_anchor_velocity(2.)[0],.0225)
    def test_direction_and_replay_inference_count(self):
        e=episode('replay',False);s=p.condition_summary(e)
        self.assertEqual(s['live_bfm_calls'],0);self.assertEqual(s['control_decisions'],8)
        self.assertEqual(s['force_abs_peak_xyz_n'],[1.,2.,2.])
        self.assertAlmostEqual(s['force_direction_angle_to_plus_x_deg']['median'],math.degrees(math.acos(1/3)))
        self.assertEqual(s['force_law_reconstruction_max_error_n'],0.)
    def test_comparison_flags_tampered_replay(self):
        a,b,c=episode('no_force',True),episode('closed_loop',True),episode('replay',False)
        result=p.analyze_all({'no_force':a,'closed_loop':b,'replay':c})
        self.assertTrue(result['replay_actions_exactly_match_no_force'])
        self.assertTrue(result['same_absolute_tether_trajectory'])
        c['actor'][-1]['action'][0]=.001
        self.assertFalse(p.analyze_all({'no_force':a,'closed_loop':b,'replay':c})['replay_actions_exactly_match_no_force'])
    def test_truncated_failure_uses_common_time_not_different_terminals(self):
        base,failed=episode('no_force',True),episode('replay',False)
        failed['physics']=failed['physics'][:6];failed['metrics']['fall_rate']=1.
        base['physics'][-1]['wrist'][0]=99.
        result=p.comparison(base,failed)
        self.assertFalse(result['full_horizon_pair_valid'])
        self.assertEqual(result['common_end_time_s'],4.25)
        self.assertEqual(result['paired_common_horizon_delta_m']['wrist'],[0.,0.,0.])
        self.assertNotIn('paired_final_delta_m',result)
    def test_numpy_failure_scalar_is_preserved_not_suppressed(self):
        class Scalar:
            __module__='numpy'
            def item(self):return True
        self.assertEqual(json.loads(json.dumps({'failed':Scalar()},default=p.json_scalar)),{'failed':True})
        with self.assertRaises(TypeError):p.json_scalar(object())
    def test_protocol_and_source_parse(self):
        self.assertEqual(p.OVERRIDES,dict(direction=[1.,0.,0.],displacement_min=.03,displacement_max=.03,max_force=5.,activation_time=1.,anchor_delay=0.,release_time=4.5))
        for n in ['right_wrist_anchor.py','right_wrist_anchor_supervisor.py','test_right_wrist_anchor.py',
                  'verify_right_wrist_anchor.py','plot_right_wrist_anchor.py']:
            ast.parse((Path(__file__).parent/n).read_text())
        self.assertNotIn('torch',sys.modules);self.assertNotIn('mujoco',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
