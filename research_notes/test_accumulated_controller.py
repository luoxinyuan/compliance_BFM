import unittest
import numpy as np
from accumulated_controller import AccumulatedCorrection,executable_bounds


class ControllerTests(unittest.TestCase):
    def tick(self, c, t, active=True):
        return c.tick(t, active, np.full(7,-2.), np.full(7,2.))

    def test_no_contact_no_control(self):
        c=AccumulatedCorrection(np.full(7,.2))
        self.assertTrue(np.array_equal(self.tick(c,0,False)[0],np.zeros(7)))

    def test_saturation_no_windup_and_immediate_reversal(self):
        c=AccumulatedCorrection(np.full(7,.2));t=0
        for high in range(100):
            c.propose(np.array([.04,0,0,0,0,0,0]),True,-np.ones(7),np.ones(7))
            for _ in range(5): self.tick(c,t);t+=.02
        self.assertAlmostEqual(c.applied[0],.2)
        self.assertAlmostEqual(c.target[0],.2)
        c.propose(np.array([-.04,0,0,0,0,0,0]),True,-np.ones(7),np.ones(7))
        self.assertLess(self.tick(c,t)[0][0],.2)

    def test_bounds_and_velocity_all_joints(self):
        c=AccumulatedCorrection(np.full(7,.2));t=0
        inc=np.full(7,.08/np.sqrt(7))
        for high in range(50):
            c.propose(inc,True,-np.ones(7),np.ones(7))
            for _ in range(5):
                a,log=self.tick(c,t);t+=.02
                self.assertLessEqual(np.linalg.norm(a),.4+1e-10)
                self.assertLessEqual(max(abs(a)),.2+1e-10)
                self.assertLessEqual(np.linalg.norm(log['velocity']),.8+1e-8)
                self.assertLessEqual(max(abs(np.array(log['velocity']))),.4+1e-8)

    def test_release_continuous_monotone_zero(self):
        c=AccumulatedCorrection(np.full(7,.2));t=0
        for h in range(15):
            c.propose(np.full(7,.08/np.sqrt(7)),True,-np.ones(7),np.ones(7))
            for _ in range(5):self.tick(c,t);t+=.02
        before=c.applied.copy();a,_=self.tick(c,t,False);t+=.02
        np.testing.assert_allclose(a,before)
        for _ in range(45):
            previous=a.copy();a,log=self.tick(c,t,False);t+=.02
            self.assertLessEqual(np.linalg.norm(a),np.linalg.norm(previous)+1e-12)
            self.assertLessEqual(np.linalg.norm(log['velocity']),.8+1e-8)
        np.testing.assert_array_equal(a,np.zeros(7))

    def test_joint_margin_blocks_outward_integration(self):
        c=AccumulatedCorrection(np.full(7,.2));t=0
        for h in range(20):
            c.propose(np.full(7,.02),True,-np.ones(7),np.full(7,.05))
            for _ in range(5):
                a,_=c.tick(t,True,-np.ones(7),np.full(7,.05));t+=.02
                self.assertLessEqual(max(a),.05+1e-10)
        np.testing.assert_allclose(c.target,.05)

    def test_invalid_base_or_unachievable_envelope_stops(self):
        c=AccumulatedCorrection(np.full(7,.2))
        with self.assertRaisesRegex(ValueError,'Original BFM'):
            c.tick(0,True,np.ones(7),np.ones(7)*2)
        c=AccumulatedCorrection(np.full(7,.2));c.applied[:]=.1
        with self.assertRaisesRegex(ValueError,'velocity constraints'):
            c.tick(0,True,-np.ones(7),np.full(7,.01))

    def test_action_range_is_tighter_than_joint_range(self):
        q=np.full(7,.05);scale=np.full(7,.128);offset=np.zeros(7)
        lower,upper=executable_bounds(q,np.tile([-2.,2.],(7,1)),scale,offset,5.)
        self.assertLess(max(upper),.08)
        c=AccumulatedCorrection(np.full(7,.2));t=0
        for h in range(30):
            c.propose(np.full(7,.02),True,lower,upper)
            for _ in range(5):
                a,_=c.tick(t,True,lower,upper);t+=.02
                self.assertLess(max(abs((q+a)/scale)),1.)
        np.testing.assert_allclose(c.applied,upper)

if __name__=='__main__':unittest.main()
