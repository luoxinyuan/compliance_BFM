"""Maintained bilateral tether points on ONE rigid body; add/remove exactly one summed COM wrench."""
import math,types
from collections import deque
import numpy as np

def rotation(a):
 c,s=math.cos(a),math.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def smooth(t,start,ramp):
 u=float(np.clip((t-start)/ramp,0,1));return 3*u*u-2*u*u*u,(6*u*(1-u)/ramp if 0<u<1 else 0.)
def frame(spec,t):
 from heading_env import pose
 if spec.get('geometry','wrist')=='wrist':
  if 'stop_time' in spec and t>=spec['stop_time']:
   from stop_env import braking
   x,xd=braking(spec,t);return np.array([x,0.,0.]),np.array([xd,0.,0.]),0.,0.
  p,v=pose(spec,t);s,_=smooth(t,spec['onset'],spec['ramp']);angle=spec['curvature']*spec['distance']*s
  if 'change_time' in spec:
   uc,_=smooth(spec['change_time'],spec['onset'],spec['ramp']);angle=spec['curvature']*spec['distance']*min(s,uc)+spec['after_curvature']*spec['distance']*max(0,s-uc)
  return p,v,angle,0.
 s,ds=smooth(t,spec['onset'],spec['ramp']);a=math.radians(spec.get('rotation_deg',0));delta=np.array([spec['distance'],0,0.]);return s*delta,ds*delta,a*s,a*ds

def install(task):
 import mujoco
 from humanoidverse.force_compliance.interaction import spring_force
 e=task.e;t=e.tether;m=t.model;d=t.data;spec=task.spec;pair=spec.get('geometry')=='torso_pair'
 t.body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'torso_link' if pair else 'right_wrist_yaw_link')
 assert t.body>0
 locals_=np.array([[0.,.13,.15],[0.,-.13,.15]]) if pair else np.zeros((1,3));t.cfg.local_point=locals_.mean(0).tolist()
 base=d.xpos[t.body]+d.xmat[t.body].reshape(3,3)@locals_.mean(0);R0=d.xmat[t.body].reshape(3,3).copy();offsets=(R0@(locals_-locals_.mean(0)).T).T
 if not pair:base=task.anchor.copy()
 task.geometry_origin=base.copy();task.geometry_yaw=math.atan2(R0[1,0],R0[0,0]) if pair else 0.;task.contact_history=deque(maxlen=100);task.contact_rows=[];task.geometry_enabled=True
 cap=2.5 if pair else 5.;slots=[1,2] if pair else [0]
 def measure(self):
  mujoco.mj_forward(m,d);p,v,angle,omega=frame(spec,self.time);Q=rotation(angle);anchors=base+p+(Q@offsets.T).T;av=v+np.cross(np.array([0.,0.,omega]),(Q@offsets.T).T)
  self.active=bool(spec['enabled'] and spec['onset']<=self.time<spec['release']);rows=[];forces=[];points=[];vels=[];wrenches=[]
  for i,local in enumerate(locals_):
   point=d.xpos[self.body]+d.xmat[self.body].reshape(3,3)@local;jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,self.body);velocity=jac@d.qvel
   raw=80*(anchors[i]-point)+8*(av[i]-velocity) if self.active else np.zeros(3);f=spring_force(anchors[i],point,av[i],velocity,80,8,cap) if self.active else np.zeros(3)
   tau=np.cross(point-d.xipos[self.body],f);wrenches.append(np.r_[f,tau]);forces.append(f);points.append(point);vels.append(velocity)
   rows.append(dict(slot=slots[i],link='torso_link' if pair else 'right_wrist_yaw_link',body=int(self.body),local=local.tolist(),point=point.tolist(),velocity=velocity.tolist(),anchor=anchors[i].tolist(),anchor_velocity=av[i].tolist(),force=f.tolist(),raw_force=raw.tolist(),cap=cap,active=self.active,reference_com=d.subtree_com[1].tolist(),yaw_moment_com=float(np.cross(point-d.subtree_com[1],f)[2]),power=float(f@velocity)))
  self.force=np.sum(forces,axis=0);self.anchor=anchors.mean(0);self.partner_velocity=av.mean(0);self.geometry_rows=rows;self.geometry_wrench=np.sum(wrenches,axis=0);self.geometry_target_yaw=task.geometry_yaw+angle;return np.mean(points,axis=0),self.force.copy()
 def apply(self):
  self.measure();before=d.xfrc_applied[self.body].copy();self.geometry_before=before.copy();d.xfrc_applied[self.body]+=self.geometry_wrench
  assert np.allclose(d.xfrc_applied[self.body]-before,self.geometry_wrench,atol=1e-12)
  return self.geometry_wrench.copy()
 t.measure=types.MethodType(measure,t);t.apply=types.MethodType(apply,t);t.measure()
 # Existing ComplianceEnv finally subtracts this exact returned aggregate once.
 if True: # ComplianceEnv.reset replaces both tether and callback; install each reset.
  original=e._physics_step
  def wrapped():
   try:return original()
   finally:
    if getattr(task,'geometry_enabled',False) and e.trace:
     rows=getattr(t,'geometry_rows',[]);row=e.trace[-1];row['contacts']=rows;row['target_yaw']=t.geometry_target_yaw
     row['sum_force_norm']=sum(np.linalg.norm(r['force']) for r in rows);row['net_yaw_moment_com']=sum(r['yaw_moment_com'] for r in rows);row['external_power']=sum(r['power'] for r in rows)
     assert row['sum_force_norm']<=5.000001
     # Wrench must have been removed, not carried into the next physics tick.
     assert np.max(np.abs(d.xfrc_applied[t.body]-t.geometry_before))<1e-8
  e.sim.simulate_at_each_physics_step=wrapped;e.geometry_logged=True
 task.geometry_contract=dict(link='torso_link' if pair else 'right_wrist_yaw_link',body=int(t.body),locals=locals_.tolist(),parent=int(m.body_parentid[t.body]),body_joints=int(m.body_jntnum[t.body]),cap_each=cap,total_norm_budget=5.,wrench_reference='body inertial COM xipos; diagnostics subtree COM',origin=base.tolist(),R0=R0.tolist(),semantics='maintained bilateral tether, not unilateral press')
