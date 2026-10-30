"""CPU-only real-model reach bound and unconstrained-by-teacher geometric IK.
No physical stepping, actor loading or policy update. All geometry uses private MjData.
"""
import json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
ROOT=Path(__file__).resolve().parents[1]
RUNS=P/'bridge-v2-owned/runs'
REF=RUNS/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json'

def analyze(root=ROOT):
 m=mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
 ref=json.loads(REF.read_text());q=np.asarray(ref['env_bookkeeping']['last_qpos']);d.qpos[:]=q;mujoco.mj_forward(m,d)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');shoulder=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_shoulder_pitch_link')
 arm=[i for i in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or '').startswith('right_') and any(x in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or '') for x in ['shoulder','elbow','wrist'])]
 waist=[i for i in range(m.njnt) if 'waist' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or '')]
 chain=[];b=body
 while b!=shoulder:
  chain.append(dict(body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b),offset=m.body_pos[b].tolist(),length=float(np.linalg.norm(m.body_pos[b]))));b=int(m.body_parentid[b])
 length=sum(x['length'] for x in chain);origin=d.xpos[body].copy();center=d.xpos[shoulder].copy();maxx=center[0]+np.sqrt(max(0.,length**2-np.square((origin-center)[1:]).sum()))
 # Target just 2 cm beyond even the loose chain-length sphere, no artificial joint restriction.
 delta=maxx-origin[0]+.02;target=origin+np.array([delta,0,0]);rng=np.random.default_rng(0);sol=[]
 for label,ids in [('arm_only',arm),('arm_and_waist',waist+arm)]:
  adr=m.jnt_qposadr[ids];lo=m.jnt_range[ids,0];hi=m.jnt_range[ids,1];best=None
  def residual(x):
   d.qpos[:]=q;d.qpos[adr]=x;mujoco.mj_forward(m,d);return d.xpos[body]-target
  for i in range(16):
   start=np.clip(q[adr],lo+1e-8,hi-1e-8) if i==0 else rng.uniform(lo,hi)
   fit=least_squares(residual,start,bounds=(lo,hi),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   error=float(np.linalg.norm(residual(fit.x)))
   if best is None or error<best['error_m']:
    best=dict(error_m=error,qpos=d.qpos.tolist(),wrist=d.xpos[body].tolist(),shoulder=d.xpos[shoulder].tolist(),joint_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in ids],joint_positions=fit.x.tolist())
  sol.append(dict(label=label,**best))
 return dict(source=str(REF),qpos=q.tolist(),wrist=origin.tolist(),shoulder=center.tolist(),chain=chain,arm_radius_upper_bound_m=length,arm_fixed_yz_max_world_x_upper_bound_m=float(maxx),target=target.tolist(),delta_x_m=float(delta),sphere_lower_bound_error_m=float(np.linalg.norm(target-center)-length),solutions=sol,interpretation='Sphere bound is rigorous for fixed shoulder, including all original joint ranges. IK minima are examples, not global impossibility proof. No collisions, balance or dynamics enforced in this geometric check.')
if __name__=='__main__':
 result=analyze(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT);print(json.dumps(result,indent=2))
