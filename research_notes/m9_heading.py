"""Rigid yaw transform of an existing stable state, never a joint-pose edit."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from m8_stage import advance
def rotate_ready(env,model,degrees):
 d=env.sim.data;G=Rotation.from_euler('z',degrees,degrees=True);R=G.as_matrix();pivot=d.qpos[:3].copy();q=Rotation.from_quat(d.qpos[[4,5,6,3]]);new=(G*q).as_quat();d.qpos[3:7]=new[[3,0,1,2]];d.qvel[:3]=R@d.qvel[:3];d.qacc_warmstart[:3]=R@d.qacc_warmstart[:3];d.qfrc_applied[:3]=R@d.qfrc_applied[:3];d.xfrc_applied[:,:3]=d.xfrc_applied[:,:3]@R.T;d.xfrc_applied[:,3:]=d.xfrc_applied[:,3:]@R.T
 # Free-joint angular velocity is body-local, so it is intentionally unchanged.
 # Proprio/action histories are yaw-invariant; refresh them with .4s actual
 # standing controls before any contact. No reference pose or hidden mode reset.
 mujoco.mj_forward(env.sim.model,d)
 for _ in range(20):
  a=model.act(env.obs,env.standing,mean=True);assert not advance(env,a)
 assert not env.failed
 return R,pivot,dict(rigid_heading_offset_deg=degrees,ready_time=float(env.tether.time),refresh_controls=20,joint_pose_manually_changed=False,root=d.qpos[:3].tolist())
