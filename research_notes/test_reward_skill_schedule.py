"""Synthetic command scheduling checks; no real model or physics."""
import unittest
import torch
from reward_skill_rollout import coefficient,spherical


class ScheduleTests(unittest.TestCase):
    def test_fixed_entry_execute_exit(self):
        self.assertEqual([coefficient(t) for t in [0,.375,.75,2,3.25,3.625,4,6]],
                         [0,.5,1,1,1,.5,0,0])
        for t in [0,.75,3.25,4]:
            self.assertLess(abs(coefficient(t+1e-6)-coefficient(t-1e-6)),1e-9)

    def test_norm_endpoint_angle_monotonic(self):
        a=torch.zeros(1,256);a[0,0]=16
        b=torch.zeros(1,256);b[0,1]=16
        angles=[]
        for x in [0,.1,.25,.5,.75,.9,1]:
            z=spherical(a,b,x)
            self.assertAlmostEqual(float(z.norm()),16,places=5)
            angles.append(float(torch.acos((z*a).sum()/256)))
        self.assertEqual(angles,sorted(angles))
        self.assertTrue(torch.equal(spherical(a,b,0),a))
        self.assertTrue(torch.equal(spherical(a,b,1),b))
        self.assertTrue(torch.equal(spherical(a,a,.5),a))
        with self.assertRaises(AssertionError):spherical(a,-a,.5)


if __name__=='__main__':unittest.main()
