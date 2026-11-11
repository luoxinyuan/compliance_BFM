"""CPU reduction of saved real qpos. No dynamics, reward or old record mutation."""
import time
START=time.monotonic()
import argparse,copy,json,sys,math
from pathlib import Path
import numpy as np,torch,mujoco
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--run',action='append',required=True);p.add_argument('--name',required=True);args=p.parse_args()
S=args.stage;R=S/'runtime-l2';sys.path[:0]=[str(R/'research_notes'),str(R)]
from f1_sequence_search import semantics
from frl_v3_risk import metrics
m=mujoco.MjModel.from_xml_path(str(R/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mujoco.MjData(m)
ids={name:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name) for name in ['left_ankle_roll_link','right_ankle_roll_link']}
result=[]
P=S.parents[1];L0=P/'experiments/f1-l0-array-audit-20261111'
manifest=json.loads((L0/'runs/l0-complete-02/state-manifest.json').read_text());specs={}
for row in manifest:
 if row.get('role')=='source_positive':continue
 snap=torch.load(L0/row['state_snapshot'],map_location='cpu',weights_only=False);specs[row['state_id']]=snap['spec']

try:
 for run in args.run:
  rows=json.loads((S/'runs'/run/'candidate-scores.json').read_text())
  for r in rows:
   if r['layer']=='L2-prediction':continue
   x=torch.load(S/'runs'/run/r['artifact'],map_location='cpu',weights_only=False);ph=x['physics'];poses=[]
   for row in ph:
    d.qpos[:]=row['qpos'];mujoco.mj_kinematics(m,d)
    for name,pos in row['body_positions'].items():assert np.array_equal(d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)],pos),'saved kinematics mismatch'
    poses.append({name:(d.xpos[i].copy(),d.xmat[i].reshape(3,3).copy()) for name,i in ids.items()})
   travel={k:0. for k in ids};intervals=0
   for i,row in enumerate(ph[:-1]):
    increments={}
    for contact,detail in zip(row['stage_foot_contacts'],row['stage_foot_contact_details']):
     if contact['normal_N']<=1e-6:continue
     name=contact['body'];local=np.array(detail['body_local_point']);p0,R0=poses[i][name];p1,R1=poses[i+1][name]
     delta=(p1+R1@local)-(p0+R0@local)
     increments[name]=max(increments.get(name,0.),float(np.linalg.norm(delta[:2])))
    for k,v in increments.items():travel[k]+=v
    intervals+=1
   spec=specs[r['state_id'].removesuffix('-continue')].copy()
   if r['state_id'].endswith('-continue'):
    from frl_v2_task import continuation
    spec=continuation(spec)
   spec.setdefault('demand_family','mid')
   first=ph[:min(200,len(ph))];co=[c for c in x['controls'] if c['time']<first[0]['force_time']+1-1e-8];early=metrics(first,co,spec)
   shoulder=lambda row:np.asarray(row['body_positions']['right_shoulder_pitch_link'])
   rel=lambda row:np.asarray(row['torso_matrix']).T@(np.asarray(row['wrist'])-shoulder(row))
   margin=[]
   for row in ph:
    for ji in range(m.njnt):
     if m.jnt_limited[ji]:
      q=row['qpos'][m.jnt_qposadr[ji]];margin.append(min(q-m.jnt_range[ji,0],m.jnt_range[ji,1]-q))
   body=dict(wrist_relative_shoulder_torso_delta_cm=(100*(rel(ph[-1])-rel(ph[0]))).tolist(),minimum_joint_margin_rad=float(min(margin)),actual_joint_dq_max=float(np.abs(np.asarray([p['qvel'][6:] for p in ph])).max()),q_PD_slew_max_deg=math.degrees(max(c['q_PD_step_max_rad'] for c in x['controls'])),support_mean=np.asarray([p['support'] for p in ph]).mean(0).tolist())
   result.append(dict(first_second_metrics=early,body=body,run=run,artifact=r['artifact'],layer=r['layer'],state_id=r['state_id'],candidate=r.get('candidate'),semantics=semantics(ph),loaded_material_point_xy_path_m=travel,intervals=intervals,kinematics_all_body_positions_exact=True,displacement_definition='track each prior loaded contact material point through actual next5ms body transform; max per foot/tick then integrate, including unloading interval; no point-count multiplication'))
  (S/'runs'/(args.name+'.json')).write_text(json.dumps(result,indent=2))
 status='completed'
except BaseException as e:
 status='failed_preserved';raise
finally:
 (S/'jobs'/(args.name+'.json')).write_text(json.dumps(dict(layer='L1' if all(x.startswith('l1') for x in args.run) else 'L2',status=locals().get('status','failed_preserved'),elapsed_s=time.monotonic()-START,physics_steps=0,arrays_masked_new=0,rows=len(result)),indent=2))
