"""Saved physical data only: target jumps are not measured joint motion."""
from pathlib import Path
import json,gzip,xml.etree.ElementTree as ET
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/t30-transfer-20261103';R.mkdir(exist_ok=True);out=R/'quality';out.mkdir(exist_ok=True)
old=P/'bridge-v2-owned/runs/stop-sensing-20261103';contract=json.loads((old/'A-prepare-03/job/action-contract.json').read_text());names=contract['dof_names'];xml=P/'bridge-v2-owned/trees/t30-transfer/repo/humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'
# Explicit mapping check using the existing MuJoCo model, no stepping or BFM inference.
import mujoco
m=mujoco.MjModel.from_xml_path(str(xml));qids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];qadr=m.jnt_qposadr[qids];vadr=m.jnt_dofadr[qids]
files=[old/'E-final-08/job'/('selected_'+n+'.json.gz') for n in ['stop_unseen_early','stop_unseen_late','normal_new','local']];results=[]
for f in files:
 with gzip.open(f,'rt') as fp:e=json.load(fp)
 c=e['control'];p=e['physics'];t=np.array([x['time'] for x in c]);qt=np.array([x['q_PD'] for x in c]);q=np.array([x['pre_qpos'] for x in c])[:,qadr];dq=np.array([x['pre_qvel'] for x in c])[:,vadr];z=np.array([x['executed_z'][0] for x in c]);delta=np.rad2deg(np.diff(qt,axis=0));actual=np.rad2deg(np.diff(q,axis=0));pe=np.rad2deg(qt-q);angle=np.rad2deg(np.arccos(np.clip(np.sum(z[1:]*z[:-1],axis=1)/256,-1,1)))
 joint=[dict(name=n,max_target_step=float(np.abs(delta[:,j]).max()),p95=float(np.quantile(np.abs(delta[:,j]),.95)),p99=float(np.quantile(np.abs(delta[:,j]),.99)),p999=float(np.quantile(np.abs(delta[:,j]),.999)),max_actual_20ms_step=float(np.abs(actual[:,j]).max()),max_target_error=float(np.abs(pe[:,j]).max()),max_speed=float(np.abs(dq[:,j]).max())) for j,n in enumerate(names)]
 events=[]
 for flat in np.argsort(np.abs(delta).ravel())[-10:][::-1]:
  i,j=np.unravel_index(flat,delta.shape);i+=1;phase=next((n for n,(a,b) in e['phases'].items() if a<=t[i]<b),'other');near=min(p,key=lambda x:abs(x['time']-t[i]));events.append(dict(time=float(t[i]),phase=phase,joint=names[j],target_step_deg=float(delta[i-1,j]),actual_preceding_20ms_deg=float(actual[i-1,j]),actual_velocity_rad_s=float(dq[i,j]),target_error_deg=float(pe[i,j]),latent_step_deg=float(angle[i-1]),force_xyz=near['force'],torque_saturation=near['torque_saturation']))
 phases={}
 for name,(a,b) in e['phases'].items():
  pp=[x for x in p if a<=x['time']<b];ii=np.where((t>=a)&(t<b))[0];ii=ii[ii>0]
  if not pp:continue
  tt=np.array([x['time'] for x in pp]);root=np.array([x['root'] for x in pp]);wr=np.array([x['wrist'] for x in pp]);dt=np.diff(tt);rv=np.diff(root,axis=0)/dt[:,None];wv=np.diff(wr,axis=0)/dt[:,None];ra=np.diff(rv,axis=0)/dt[1:,None];wa=np.diff(wv,axis=0)/dt[1:,None];tor=np.array([x['torque'] for x in pp])
  phases[name]=dict(root_speed_rms=float(np.sqrt((rv*rv).sum(1).mean())),wrist_speed_rms=float(np.sqrt((wv*wv).sum(1).mean())),root_acc_p99=float(np.quantile(np.linalg.norm(ra,axis=1),.99)),wrist_acc_p99=float(np.quantile(np.linalg.norm(wa,axis=1),.99)),root_path=float(np.linalg.norm(np.diff(root,axis=0),axis=1).sum()),root_delta=(root[-1]-root[0]).tolist(),torque_abs_peak=float(np.abs(tor).max()),torque_saturation_peak=max(x['torque_saturation'] for x in pp),slip_peak=max(x['contact_tangent_speed_max_mps'] for x in pp),no_support_fraction=float(np.mean([not any(x['support']) for x in pp])),tilt_max=max(x['tilt_deg'] for x in pp),target_step_max_deg=float(np.abs(delta[ii-1]).max()) if len(ii) else 0,actual_step_max_deg=float(np.abs(actual[ii-1]).max()) if len(ii) else 0)
 result=dict(path=str(f),label=e['label'],joint_mapping=names,joints=joint,top_events=events,phases=phases,interpretation='actual_preceding_20ms is prior interval, not causal effect of current command; derivative accelerations are finite-difference diagnostics');results.append(result)
 print(e['label'],events[0],flush=True)
(out/'teacher30-quality.json').write_text(json.dumps(results,indent=2))
