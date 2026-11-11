"""Pure kinematic audit of BEFORE/AFTER integration tangent velocities, no mj_step."""
import argparse,gzip,json,math,time
from pathlib import Path
from collections import Counter
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();R=Path(__file__).resolve().parents[1];model_path=R/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml';model=mujoco.MjModel.from_xml_path(str(model_path));data=mujoco.MjData(model);nv=model.nv
rows=[];total_ticks=0;max_reproduce=0.;point_error=0.;load=Counter();kind=Counter()
for file in sorted(a.output.glob('state*.json.gz')):
 ep=json.load(gzip.open(file,'rt'));ph=ep['physics'];cs=ep['control'];prevq=np.asarray(cs[0]['pre_qpos']);prevv=np.asarray(cs[0]['pre_qvel']);maxrun=0;run=0
 for tick in ph:
  total_ticks+=1
  if tick['stage_loaded_slip_mps']>.45:
   data.qpos[:]=prevq;data.qvel[:]=prevv;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
   contacts=tick['stage_foot_contacts'];details=tick['stage_foot_contact_details'];assert len(contacts)==len(details)
   j=max((j for j,c in enumerate(contacts) if c['normal_N']>1e-6),key=lambda j:contacts[j]['tangent_speed_mps']);con=contacts[j];detail=details[j];body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,detail['body']);point=np.asarray(detail['world_point']);reconstructed=data.xpos[body]+data.xmat[body].reshape(3,3)@detail['body_local_point'];pe=float(np.max(abs(point-reconstructed)));point_error=max(point_error,pe)
   jac=np.zeros((3,nv));mujoco.mj_jac(model,data,jac,None,point,body);before=float(np.linalg.norm((jac@prevv)[:2]));after=float(np.linalg.norm((jac@np.asarray(tick['qvel']))[:2]));err=abs(before-con['tangent_speed_mps']);max_reproduce=max(max_reproduce,err)
   assert pe<1e-8 and err<1e-8,('kinematic reconstruction mismatch',file.name,tick['force_time'],pe,err)
   n=con['normal_N'];band='<=1N' if n<=1 else '1-10N' if n<=10 else '10-100N' if n<=100 else '>100N';load[band]+=1;category='poststep_still_over045' if after>.45 else 'resolved_within5ms';kind[category]+=1;run=run+1 if after>.45 else 0;maxrun=max(maxrun,run)
   rows.append(dict(state_id=ep['summary']['state_id'],parent_id=ep['summary']['parent_id'],candidate=ep['summary']['candidate'],time=tick['force_time'],before_mps=before,after_mps=after,normal_N=n,body=detail['body'],local_point=detail['body_local_point'],category=category))
  else:run=0
  prevq=np.asarray(tick['qpos']);prevv=np.asarray(tick['qvel'])
 if len(rows):rows[-1]['branch_longest_consecutive_prestep_and_poststep_over045_ticks']=maxrun
result=dict(status='completed',elapsed_s=time.monotonic()-start,total_physics_ticks=total_ticks,overlimit_ticks=len(rows),old_velocity_reproduction_max_error=max_reproduce,contact_local_point_reconstruction_max_error_m=point_error,before_after_counts=dict(kind),normal_bands_N=dict(load),rows=rows,no_physics_steps=True,unchanged_risk_contract=True,interpretation='original metric uses pre-integration tangent velocity and force solved during that tick; post-integration resolution is descriptive only, not grounds to delete old overlimits. Transient-resolved does not prove zero5ms sliding distance; persistent/poststep motion stronger evidence of ongoing slip.')
(a.output/'postcontact-velocity-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
