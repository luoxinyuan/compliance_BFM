import unittest
import numpy as np
from ccl02a_teacher import arm_ik,ReleaseFade,phase,TASKS,audit_episode

class TeacherTests(unittest.TestCase):
    def test_jacobian_direction_bound_and_singular_case(self):
        j=np.array([[.2,0,0,0,0,0,0],[0,.1,0,0,0,0,0],[0,0,.05,0,0,0,0]])
        for sign in [-1,1]:
            error=np.array([sign*.03,.01,.02]);dq=arm_ik(j,error)
            self.assertGreater(float((j@dq)@error),0)
            self.assertLessEqual(abs(dq).max(),.04);self.assertLessEqual(np.linalg.norm(dq),.08+1e-15)
        np.testing.assert_array_equal(arm_ik(j,np.zeros(3)),np.zeros(7))
        np.testing.assert_array_equal(arm_ik(np.zeros_like(j),np.ones(3)),np.zeros(7))
    def test_release_is_causal_zero_before_contact_and_bounded(self):
        f=ReleaseFade();d=np.array([.04,-.03,.02,0,0,0,0])
        np.testing.assert_array_equal(f.update(.8,False),np.zeros(7))
        np.testing.assert_array_equal(f.update(1.,True,d),d)
        np.testing.assert_array_equal(f.update(4.6,False),d)
        np.testing.assert_allclose(f.update(4.7,False),.5*d,atol=1e-14)
        np.testing.assert_array_equal(f.update(4.8,False),np.zeros(7))
        np.testing.assert_array_equal(f.update(5.2,False),np.zeros(7))
        np.testing.assert_array_equal(f.update(5.3,True,-d),-d)
    def test_release_interpolation_uses_low_level_clock_without_new_teacher_query(self):
        f=ReleaseFade();d=np.array([.04,-.03,.02,0,0,0,0]);f.update(4.5,True,d);f.update(4.6,False)
        values=[f.value(4.6+.02*i) for i in range(11)]
        for i,v in enumerate(values):np.testing.assert_allclose(v,max(0.,1.-i/10.)*d,atol=1e-14)
        self.assertLessEqual(max(np.linalg.norm(b-a) for a,b in zip(values,values[1:])),np.linalg.norm(d)/10.+1e-14)
        np.testing.assert_array_equal(f.last,d)
    def test_pd_audit_uses_actual_post_reset_refresh_state(self):
        def pose(q):return [0.]*7+[q]*29
        vel=[0.]*35
        control=[dict(q_PD=[1.]*29,action=[.2]*29,pre_qpos=pose(q),pre_qvel=vel) for q in [0.,.5]]
        rows=[dict(qpos=pose(q),qvel=vel,torque=[1.-q]*29) for q in [0.]*4+[.5]*4]
        episode=dict(physics=rows,control=control)
        contract=dict(q_PD_scale=[[5.]*29],q_PD_offset=[[0.]*29])
        state=dict(base_tensors=dict(p_gains=[1.]*29,d_gains=[0.]*29,_kp_scale=[[1.]*29],_kd_scale=[[1.]*29],torque_limits=[10.]*29))
        self.assertEqual(audit_episode(episode,contract,state)['torque_max_error'],0.)
        control[1]['pre_qpos']=pose(0.)
        with self.assertRaises(AssertionError):audit_episode(episode,contract,state)
    def test_fixed_task_and_stage_boundaries(self):
        self.assertEqual([(t['direction'],t['distance_m'],t['onset_s']) for t in TASKS],[(1,.025,.9),(-1,.025,.9),(1,.035,1.1),(-1,.035,1.1)])
        self.assertEqual([phase(t,.9) for t in [.899,.9,2.899,2.9,4.499,4.5,5.9]],['pre_contact','ramp','ramp','hold','hold','release','release'])

if __name__=='__main__':unittest.main()
