"""Actual per-contact, physical and economic diagnostics; no new reward term."""
import json,gzip,sys,math,hashlib
from pathlib import Path
import numpy as np
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def wrap(a):return np.arctan2(np.sin(a),np.cos(a))
def summarize(p):
 e=json.load(gzip.open(p,'rt'))
 if not isinstance(e,dict) or not e.get('physics'):return None
 rr=e['physics'];s=e['condition'];a=[r for r in rr if r['active']];initial=min(rr,key=lambda r:abs(r['force_time']-s['onset']));end=a[-1] if a else rr[-1]
 def measures(rows,tracking):
  if not rows:return None
  root=np.array([r['root'] for r in rows]);feet=np.array([r['feet_xyz'] for r in rows]);q=np.array([r['qpos'][7:] for r in rows]);dq=np.array([r['qvel'][6:] for r in rows]);torque=np.array([r['torque'] for r in rows]);T=np.array([r['torso_matrix'] for r in rows]);P=np.array([r['pelvis_matrix'] for r in rows]);w=np.array([r['wrist'] for r in rows]);shoulder=np.array([r['shoulder_xyz'] for r in rows]);f=np.array([[c['force'] for c in r['contacts']] for r in rows]);err=np.array([[np.array(c['anchor'])-c['point'] for c in r['contacts']] for r in rows]);ty=np.arctan2(T[:,1,0],T[:,0,0]);py=np.arctan2(P[:,1,0],P[:,0,0]);dt=.005
  r=dict(seconds=len(rows)*dt,force_rms=rms(np.linalg.norm(f,axis=-1)),sum_force_norm_rms=rms(np.linalg.norm(f,axis=-1).sum(-1)),force_xyz_rms=np.sqrt((f*f).mean(0)).tolist(),cap_fraction=float(np.mean([any(np.linalg.norm(c['force'])>=c['cap']-.001 for c in r['contacts']) for r in rows])),error_cm=100*rms(np.linalg.norm(err,axis=-1)) if tracking else None,signed_error_xyz_cm=(100*err.mean(0)).tolist() if tracking else None,root_delta_cm=(100*(root[-1]-root[0])).tolist(),root_xy_path_cm=float(100*np.linalg.norm(np.diff(root[:,:2],axis=0),axis=-1).sum()),foot_xy_path_cm=(100*np.linalg.norm(np.diff(feet[:,:,:2],axis=0),axis=-1).sum(0)).tolist(),wrist_delta_cm=(100*(w[-1]-w[0])).tolist(),relative_wrist_shoulder_delta_cm=(100*((w-shoulder)[-1]-(w-shoulder)[0])).tolist(),shoulder_elbow_q_delta_deg=np.rad2deg(q[-1,22:26]-q[0,22:26]).tolist(),joint_travel_rad=float(abs(np.diff(q,axis=0)).sum()),absolute_mechanical_work_proxy_J=float(np.abs(torque*dq).sum()*dt),max_tilt=max(r['tilt_deg'] for r in rows),torso_pitch_deg_max=float(np.rad2deg(np.arcsin(np.clip(-T[:,2,0],-1,1))).max()),pelvis_pitch_deg_max=float(np.rad2deg(np.arcsin(np.clip(-P[:,2,0],-1,1))).max()),torso_yaw_delta_deg=float(np.rad2deg(wrap(ty[-1]-ty[0]))),pelvis_yaw_delta_deg=float(np.rad2deg(wrap(py[-1]-py[0]))),waist_final_deg=float(np.rad2deg(wrap(ty[-1]-py[-1]))),slip_peak=max(r['contact_tangent_speed_max_mps'] for r in rows),support_loss_ticks=sum(not any(r['support']) for r in rows),torque_saturation_peak=max(r['torque_saturation'] for r in rows),actual_q_max_step_deg=float(np.rad2deg(abs(np.diff(q,axis=0))).max()) if len(rows)>1 else 0.,root_final_speed=float(np.linalg.norm(rows[-1]['qvel'][:3])))
  return r
 phase={}
 for name,lo,hi in [('pre',0,s['onset']),('move',s['onset'],s['onset']+s['ramp']),('hold',s['onset']+s['ramp'],s['release']),('recover',s['release'],s['end']+.01)]:phase[name]=measures([r for r in rr if lo<=r['force_time']<hi],name in ['move','hold'] and s['enabled'])
 qpd=np.array([c['q_PD'] for c in e['control']]);terms={}
 for st in e.get('rl_stats',[]):
  for k,v in st['terms'].items():terms[k]=terms.get(k,0)+v*.02*st['control_steps']
 root0=np.asarray(initial['root']);feet0=np.asarray(initial['feet_xyz']);ev=next((r['force_time'] for r in rr if r['force_time']>=s['onset'] and np.max(np.linalg.norm((np.asarray(r['feet_xyz'])-feet0)[:,:2],axis=-1))>.02),None)
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),label=e['label'],condition=s,contact=measures(a,True),phase=phase,boundary=e['boundary'],reward_integrated_terms=terms,return_value=sum(st['reward'] for st in e.get('rl_stats',[])),root_contact_delta_cm=(100*(np.asarray(end['root'])-root0)).tolist(),foot_xy2cm_time=ev,qpd_max_step_deg=float(np.rad2deg(abs(np.diff(qpd,axis=0))).max()),qpd_p99_step_deg=float(np.percentile(np.rad2deg(abs(np.diff(qpd,axis=0))),99)),physics_steps=len(rr),control_steps=len(e['control']))
if __name__=='__main__':
 root=Path(sys.argv[1]);dest=Path(sys.argv[2]);rows=[]
 for p in sorted(root.glob('*.json.gz')):
  if any(z in p.name for z in ['labels','queries','prefix','timelines']):continue
  r=summarize(p)
  if r:rows.append(r)
 dest.write_text(json.dumps(rows,indent=2));print('METRICS',len(rows))
