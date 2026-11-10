"""Finite original native sources, real BFM/PD execution; no contact or training."""
import copy,gzip,json,time
from pathlib import Path
import numpy as np
import torch
from native_env import Task,spec
from locomotion_rollout import spherical
from m9_cyclic import sample
from wholebody_physics import goal_observation
from dual_track_high import write,sha

CASES=['standing','forward_motion','left_motion','right_motion','left_waist_goal','right_waist_goal',
 'lower_body_goal','arm_ready_goal','arm_lift_goal','backward_reward','forward_reward','backward_velocity_goal']

def run(e,m,c,out,args):
 start=time.monotonic();task=Task(e,m,c,5802);rows=[]
 reward=json.loads((args.reference/'b-mean-results.json').read_text());assert reward['original_model_sha256']==c['model_sha256']
 rz={r['name']:torch.tensor(r['z'],device='cuda',dtype=torch.float32) for r in reward['rows']}
 import mujoco
 ids=[mujoco.mj_name2id(e.sim.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in c['dof_names']]
 limits=e.sim.model.jnt_range[ids];qadr=e.sim.model.jnt_qposadr[ids]
 for kind in CASES:
  s=spec(0.,4.,source='standing',until=4.,enabled=False);s.update(end=17.,release=13.,onset=4.)
  task.reset(s,record=True,label=kind);q=e.sim.data.qpos.copy();v=np.zeros(e.sim.model.nv)
  target=e.standing;source={}
  if kind.endswith('_motion'):
   ref={'forward_motion':'straight','left_motion':'left','right_motion':'right'}[kind]
   source=dict(interface='native motion/goal latent sequence',reference=ref,variant='M10 task',period=task.refs[ref]['period'])
  elif kind in ['left_waist_goal','right_waist_goal']:
   q[7+12]+=np.deg2rad(15 if kind.startswith('left') else -15);target=m.goal_inference(goal_observation(e,q,v));source=dict(interface='goal_inference',waist_target_deg=15 if kind.startswith('left') else -15)
  elif kind=='arm_lift_goal':
   q[7+22]-=.6;target=m.goal_inference(goal_observation(e,q,v));source=dict(interface='goal_inference',right_shoulder_pitch_offset_rad=-.6)
  elif kind=='backward_velocity_goal':
   v[0]=-.15;target=m.goal_inference(goal_observation(e,q,v));source=dict(interface='goal_inference',root_world_vx_target=-.15)
  elif kind=='lower_body_goal':
   from m10_reference import load_variants
   target=load_variants(e,m)[1]['neutral'];source=dict(interface='goal_inference archived actual squat pose',reference='M10 neutral-arm body goal, originating dance1_subject2/frame1690')
  elif kind=='arm_ready_goal':target=task.native_z['goal_arm5'];source=dict(interface='goal_inference archived',reference=task.assets['goal_reference'])
  elif kind.endswith('_reward'):
   target=rz['backward015' if kind.startswith('backward') else 'forward015'];source=dict(interface='original reward inference',source_audit_sha=sha(args.reference/'b-mean-results.json'),reward='body-frame +/-0.15 vx, sigma .12, lateral .10, yaw rate .25; robot-only, ordinary uncentered')
  else:source=dict(interface='archived standing latent')
  targets=[]
  while not task.done:
   t=float(e.tether.time);u=np.clip((t-4.)/1.5,0,1);u=u*u*(3-2*u)
   if t>=11.5:
    a=np.clip((t-11.5)/1.5,0,1);u*=1-a*a*(3-2*a)
   if kind.endswith('_motion'):target=sample(task.refs[ref],max(0,t-5.5)/task.refs[ref]['period'])
   z=spherical(e.standing,target,float(u));targets.append(dict(time=t,z=z.cpu().tolist()))
   task.step(z/16)
  p=task.physics;cc=task.controls
  assert all(not r['active'] and np.array_equal(np.asarray(r['force']),np.zeros(3)) for r in p)
  with gzip.open(out/(kind+'-actual.json.gz'),'wt') as f:json.dump(dict(initial=task.initial,source_prefix=task.prefix_data,physics=p,controls=cc,targets=targets,source=source,spec=s,boundary=task.boundary),f)
  def measure(part,controls):
   if not part:return None
   root=np.array([r['root'] for r in part]);qs=np.array([r['qpos'] for r in part]);q=qs[:,qadr]
   yaw=lambda key:np.unwrap([np.arctan2(np.array(r[key])[1,0],np.array(r[key])[0,0]) for r in part])
   ty,py=yaw('torso_matrix'),yaw('pelvis_matrix');f=np.array([r['feet_xyz'] for r in part])
   margin=np.minimum(q-limits[:,0],limits[:,1]-q);bad=np.unravel_index(margin.argmin(),margin.shape)
   rel=np.array([np.asarray(r['wrist'])-r['shoulder_xyz'] for r in part]);qd=np.array([r['q_PD'] for r in controls])
   slip=np.array([r['contact_tangent_speed_max_mps'] if sum(r['foot_normal_force_n'])>1e-6 else 0. for r in part])
   return dict(duration_s=len(part)*.005,root_delta_m=(root[-1]-root[0]).tolist(),root_vxyz_mean=np.mean([r['qvel'][:3] for r in part],0).tolist(),
    root_height_mean=float(root[:,2].mean()),root_height_min=float(root[:,2].min()),
    torso_yaw_start_end_deg=np.degrees([ty[0],ty[-1]]).tolist(),pelvis_yaw_start_end_deg=np.degrees([py[0],py[-1]]).tolist(),
    waist_relative_start_end_deg=np.degrees([ty[0]-py[0],ty[-1]-py[-1]]).tolist(),
    wrist_relative_shoulder_start_end_m=rel[[0,-1]].tolist(),foot_path_m=np.linalg.norm(np.diff(f,axis=0),axis=2).sum(0).tolist(),
    support_counts={str(k):sum(tuple(r['support'])==k for r in part) for k in sorted({tuple(r['support']) for r in part})},
    slip_peak_mps=float(slip.max()),slip_p95_mps=float(np.quantile(slip,.95)),
    min_actual_joint_margin_rad=float(margin.min()),min_margin_joint=c['dof_names'][bad[1]],
    q_range_deg=np.degrees(q.max(0)-q.min(0)).tolist(),qdot_abs_peak=float(max(np.max(np.abs(r['qvel'][6:])) for r in part)),
    qPD_max_step_deg=float(np.degrees(abs(np.diff(qd,axis=0))).max()) if len(qd)>1 else 0.,
    actual_q_max_5ms_step_deg=float(np.degrees(abs(np.diff(q,axis=0))).max()) if len(q)>1 else 0.)
  result=dict(name=kind,source=source,boundary=task.boundary,complete=task.boundary is None and e.tether.time>=17.-1e-6,
   whole=measure(p,cc),entry=measure([r for r in p if r['time']<5.5],[r for r in cc if r['time']<5.5]),
   sustain=measure([r for r in p if 5.5<=r['time']<11.5],[r for r in cc if 5.5<=r['time']<11.5]),
   recovery=measure([r for r in p if r['time']>=13.],[r for r in cc if r['time']>=13.]),actual_external_force_max=0.)
  rows.append(result);write(out/'inventory.json',dict(rows=rows,model_sha256=c['model_sha256'],scope='no-contact native original model, not learned high policy',formal_updates=0))
  print('NATIVE_INVENTORY',kind,result['complete'],result['boundary'] and result['boundary']['kind'],flush=True)
 return dict(status='completed',episodes=len(rows),physics_ticks=sum(round(r['whole']['duration_s']/.005) for r in rows),formal_updates=0,wall_s=time.monotonic()-start)
