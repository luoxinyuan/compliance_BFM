import json,numpy as np,mujoco
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');W=P/'bridge-v2-owned/trees/arm-to-locomotion/repo';R=P/'bridge-v2-owned/runs/arm-to-locomotion-20261105';O=R/'physical-01'
m=mujoco.MjModel.from_xml_path(str(W/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));c=json.loads((O/'joint-contract.json').read_text());r=json.loads((O/'analysis.json').read_text())[0]['initial_arm'];b=c['bodies']['wrist'];stop=c['bodies']['shoulder'];chain=[];length=0.
while b!=stop:
 length+=float(np.linalg.norm(m.body_pos[b]));chain.append(dict(body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b),offset=m.body_pos[b].tolist()));b=int(m.body_parentid[b]);assert b>0
v=np.array(r['wrist_shoulder']);bound=float(np.sqrt(max(0,length**2-v[1]**2-v[2]**2))-v[0]);out=dict(max_chain_length=length,initial_wrist_shoulder=v.tolist(),max_extra_world_x_fixed_shoulder_y_z_m=bound,chain=chain,scope='necessary geometric upper bound only, permits unconstrained orientation at every actual arm joint; not a controller or proof BFM can attain it');(R/'geometry-bound.json').write_text(json.dumps(out,indent=2));print(out)
