import unittest
import numpy as np
from ccl02a_teacher import arm_ik,ReleaseFade,phase,TASKS

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
    def test_fixed_task_and_stage_boundaries(self):
        self.assertEqual([(t['direction'],t['distance_m'],t['onset_s']) for t in TASKS],[(1,.025,.9),(-1,.025,.9),(1,.035,1.1),(-1,.035,1.1)])
        self.assertEqual([phase(t,.9) for t in [.899,.9,2.899,2.9,4.499,4.5,5.9]],['pre_contact','ramp','ramp','hold','hold','release','release'])

if __name__=='__main__':unittest.main()
