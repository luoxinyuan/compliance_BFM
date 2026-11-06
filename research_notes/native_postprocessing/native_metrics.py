"""Offline physical metrics, actual200Hz q/qdot and50Hz targets kept distinct."""
import argparse,json,gzip,math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
import sys
sys.path.insert(0,'/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/native-contact-learning/repo/research_notes')
from transition_env import phase
out=[]
for file in sorted((a.root/'final-01').glob('*.json.gz')):
 if 'source-prefix' in file.name:continue
 ep=json.loads(gzip.decompress(file.read_bytes()));rs=ep['physics'];controls=ep['control'];groups={}
 for r in rs:groups.setdefault(phase(ep['condition'],r['force_time']),[]).append(r)
 row={'label':ep['label'],'phases':{},'support_metric':'actual foot normal load>5N; ankle-link paths are not exact discrete step counts','power_metric':'sum abs(tau*dq), mechanical proxy, not electrical energy'}
 for label,rr in groups.items():
  if not rr:continue
  q=np.array([r['qpos'][7:] for r in rr]);dq=np.array([r['qvel'][6:] for r in rr]);tau=np.array([r['torque'] for r in rr]);feet=np.array([r['feet_xyz'] for r in rr]);root=np.array([r['root'] for r in rr]);w=np.array([r['wrist'] for r in rr]);shoulder=np.array([r['shoulder_xyz'] for r in rr]);torso=np.array([r['body_positions']['torso_link'] for r in rr]);forces=np.array([r.get('foot_normal_force_n',[0,0]) for r in rr]);support=forces>5
  relative=w-shoulder;T=np.array([r['torso_matrix'] for r in rr]);torso_w=np.einsum('nji,nj->ni',T,w-torso)
  v={'seconds':len(rr)*.005,'root_delta_cm':(100*(root[-1]-root[0])).tolist(),'wrist_delta_cm':(100*(w[-1]-w[0])).tolist(),'shoulder_delta_cm':(100*(shoulder[-1]-shoulder[0])).tolist(),'torso_delta_cm':(100*(torso[-1]-torso[0])).tolist(),'wrist_relative_shoulder_delta_cm':(100*(relative[-1]-relative[0])).tolist(),'wrist_relative_torso_body_delta_cm':(100*(torso_w[-1]-torso_w[0])).tolist(),'ankle_link_paths_cm':(100*np.linalg.norm(np.diff(feet,axis=0),axis=-1).sum(0)).tolist(),'actual_q_range_deg':np.rad2deg(q.max(0)-q.min(0)).tolist(),'actual_dq_max_radps':np.abs(dq).max(0).tolist(),'actual_q5ms_step_max_deg':float(np.rad2deg(np.abs(np.diff(q,axis=0))).max()) if len(q)>1 else 0.,'mechanical_abs_work_proxy_j':float((np.abs(tau*dq)).sum()*.005),'actual_torque_saturation_peak_fraction':max(r['torque_saturation'] for r in rr),'tilt_max_deg':max(r['tilt_deg'] for r in rr),'root_height_min_m':min(r['height'] for r in rr),'both_feet_below5N_ticks':int((~support.any(-1)).sum()),'foot_unloaded_ticks':(~support).sum(0).tolist(),'slip_contact_speed_max_mps':max(r['contact_tangent_speed_max_mps'] for r in rr)}
  active=[r for r in rr if r['active']]
  if active:
   f=np.array([r['force'] for r in active]);err=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);v.update(force_xyz_mean_n=f.mean(0).tolist(),force_xyz_rms_n=np.sqrt((f*f).mean(0)).tolist(),force_peak_n=float(np.linalg.norm(f,axis=-1).max()),error_signed_xyz_mean_cm=(100*err.mean(0)).tolist(),error_xyz_rms_cm=(100*np.sqrt((err*err).mean(0))).tolist())
  row['phases'][label]=v
 # Identify per-joint command jumps and actual PD tracking at matching control times.
 qq=np.array([c['q_PD'] for c in controls]);actual=np.array([c['pre_qpos'][7:] for c in controls]);prefix=json.loads(gzip.decompress(file.with_name(file.name.replace('.json.gz','-source-prefix.json.gz')).read_bytes()))
 previous=np.array(prefix['control'][-1]['q_PD']) if prefix['control'] else qq[0]
 jumps=np.abs(np.diff(np.concatenate([previous[None],qq],axis=0),axis=0));ix=np.unravel_index(int(np.argmax(jumps)),jumps.shape)
 row['q_PD']={'max_step_deg':float(np.rad2deg(jumps[ix])),'joint_index':int(ix[1]),'time':controls[ix[0]]['time'],'phase':phase(ep['condition'],controls[ix[0]]['time']),'per_joint_p99_step_deg':np.rad2deg(np.quantile(jumps,.99,axis=0)).tolist(),'PD_target_vs_actual_rms_deg':np.rad2deg(np.sqrt(((qq-actual)**2).mean(0))).tolist(),'scope':'q_PD target20ms; actual q5ms measured separately, no claim robot rotated by target jump in20ms'}
 out.append(row)
(a.root/'public').mkdir(exist_ok=True);(a.root/'public/physical-details.json').write_text(json.dumps(out,indent=2,allow_nan=False));print('PHYSICAL_DETAILS',len(out))
