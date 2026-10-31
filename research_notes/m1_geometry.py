"""Read saved physical states; FK and bounded geometric feasibility only. No stepping or BFM."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from scipy.optimize import least_squares
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');ROOT=P/'bridge-v2-owned/trees/m1-state-response/repo';R=P/'bridge-v2-owned/runs'
def geometry(m,q,delta):
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');shoulder=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_shoulder_pitch_link')
 ids=[i for i in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or '').startswith('right_') and any(x in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or '') for x in ['shoulder','elbow','wrist'])]
 b=body;L=0.
 while b!=shoulder:L+=np.linalg.norm(m.body_pos[b]);b=int(m.body_parentid[b])
 wrist=d.xpos[body].copy();center=d.xpos[shoulder].copy();rel=wrist-center;target=wrist+np.array([delta,0,0]);adr=m.jnt_qposadr[ids];lo,hi=m.jnt_range[ids].T
 result=dict(wrist=wrist.tolist(),shoulder=center.tolist(),arm_chain_upper_bound_m=L,sphere_radial_slack_m=float(L-np.linalg.norm(rel)),fixed_yz_forward_upper_bound_m=float(np.sqrt(max(0,L**2-sum(rel[1:]**2)))-rel[0]),target=target.tolist(),sphere_error_lower_bound_m=float(max(0,np.linalg.norm(target-center)-L)),arm_joint_positions_rad=np.asarray(q)[adr].tolist(),arm_joint_margin_rad=np.minimum(np.asarray(q)[adr]-lo,hi-np.asarray(q)[adr]).tolist(),joint_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in ids])
 def residual(x):
  d.qpos[:]=q;d.qpos[adr]=x;mujoco.mj_forward(m,d);return d.xpos[body]-target
 rng=np.random.default_rng(0);solutions=[]
 for i in range(4):
  x=np.clip(np.asarray(q)[adr],lo+1e-8,hi-1e-8) if i==0 else rng.uniform(lo,hi)
  fit=least_squares(residual,x,bounds=(lo,hi),max_nfev=150)
  solutions.append(dict(seed_index=i,error_m=float(np.linalg.norm(residual(fit.x))),q_arm=fit.x.tolist(),nfev=fit.nfev))
 result['geometric_solutions']=solutions;result['best_example_error_m']=min(x['error_m'] for x in solutions)
 result['meaning']='Original arm ranges; no physical locking. Sphere is a loose rigorous bound for fixed shoulder; IK is feasibility/example only, no collision/balance/dynamics proof.'
 return result
if __name__=='__main__':
 read=lambda p:json.loads(p.read_text());m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
 sources={'standing':R/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json','raised':R/'issue21-reward-stage-20261030A/physics-attempt-01/outputs/raisearms-l-m.json','forward':R/'issue21-skill-source-20261030A/physics-attempt-01/outputs/goal_free.json'}
 delta=read(R/'issue21-wholebody-20261030A/attempt-01/outputs/geometry.json')['delta_x_m'];res={}
 for name,path in sources.items():
  e=read(path);q=e['env_bookkeeping']['last_qpos'] if name=='standing' else min(e['physics'],key=lambda r:abs(r['time']-6))['qpos']
  res[name]=geometry(m,q,delta);res[name].update(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 out=R/'m1-state-response-20261031A';(out/'saved-state-geometry.json').write_text(json.dumps(dict(delta_x_m=delta,states=res),indent=2));print(json.dumps(res,indent=2))
