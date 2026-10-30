"""Metric units and fixed phase durations on stationary synthetic data."""
import copy
import unittest
from reward_skill_summary import summarize


class MetricTests(unittest.TestCase):
    def test_stationary_reference(self):
        row=dict(force_time=3.,root=[0,0,.8],qpos=[0,0,.8,1,0,0,0]+[0]*29,
                 torso_matrix=[[1,0,0],[0,1,0],[0,0,1]],feet_xyz=[[0,.1,0],[0,-.1,0]],
                 com_xyz=[0,0,.7],body_positions={'left_wrist_roll_link':[0,.3,.7],
                    'right_wrist_roll_link':[0,-.3,1.1],'left_shoulder_pitch_link':[0,.1,1.1],
                    'right_shoulder_pitch_link':[0,-.1,1.1]},support=[True,True],height=.8,
                 tilt_deg=0.,force=[0,0,0],joint_limit_excess_max_rad=0.,torque_saturation=0.,undesired_ground_contacts=0)
        rows=[];control=[]
        for i in range(1200):
            r=copy.deepcopy(row);r['force_time']=3+i*.005;rows.append(r)
        for i in range(300):
            control.append(dict(relative_time=i*.02,action_saturation_fraction=0.,q_PD_step_max_rad=0.,z_angle_deg=0.))
        s=summarize(dict(label='synthetic',initial=row,physics=rows,control=control,boundary=None,common_state_sha256='synthetic'))
        for phase,duration in [('enter',.75),('execute',2.5),('exit',.75),('settle',2.)]:
            p=s['phases'][phase]
            self.assertEqual(p['duration_seconds'],duration)
            self.assertEqual(p['root_min_height_cm'],80.)
            self.assertEqual(p['ego_com_forward_mean_mps'],0.)
            self.assertEqual(p['root_displacement_peak_norm_cm'],0.)
            self.assertEqual(p['arm_reward_height_bounds_fraction'],1.)


if __name__=='__main__':unittest.main()
