"""Actual saved physics, active-contact metrics and full release separately. CPU only."""
import argparse,gzip,json,hashlib,math,time
from pathlib import Path
import numpy as np

def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def stats(ep,lo,hi,names):
 r=[x for x in ep['physics'] if lo<=x['force_time']<hi]
 c=[x for x in ep['control'] if lo<=x['time']<hi]
 if not r:return None
 ts=np.array([x['force_time'] for x in r]);q=np.array([x['qpos'][7:] for x in r]);dq=np.array([x['qvel'][6:] for x in r]);root=np.array([x['root'] for x in r]);v=np.array([x['qvel'][:3] for x in r]);foot=np.array([x['feet_xyz'] for x in r]);w=np.array([x['wrist'] for x in r]);sh=np.array([x['shoulder_xyz'] for x in r]);P=np.array([x['pelvis_matrix'] for x in r]);T=np.array([x['torso_matrix'] for x in r]);torque=np.array([x['torque'] for x in r]);load=np.array([x['foot_normal_force_n'] for x in r]);active=[x for x in r if x['active']]
 ans=dict(seconds=len(r)*.005,steps=len(r),active_steps=len(active),root_delta_cm=(100*(root[-1]-root[0])).tolist(),root_path_cm=float(100*np.linalg.norm(np.diff(root,axis=0),axis=1).sum()),root_speed_mean=float(np.linalg.norm(v[:,:2],axis=1).mean()),root_final_speed=float(np.linalg.norm(v[-1])),root_forward_speed_mean=float(v[:,0].mean()),root_body_forward_speed_mean=float(np.einsum('nji,nj->ni',P,v)[:,0].mean()),wrist_delta_cm=(100*(w[-1]-w[0])).tolist(),wrist_relative_shoulder_delta_cm=(100*((w-sh)[-1]-(w-sh)[0])).tolist(),wrist_relative_torso_body_delta_cm=(100*(T[-1].T@(w[-1]-r[-1]['body_positions']['torso_link'])-T[0].T@(w[0]-r[0]['body_positions']['torso_link']))).tolist(),foot_paths_cm=(100*np.linalg.norm(np.diff(foot,axis=0),axis=2).sum(0)).tolist(),foot_net_cm=(100*(foot[-1]-foot[0])).tolist(),unloaded_fraction=(load<5).mean(0).tolist(),both_unloaded_fraction=float(np.mean((load<5).all(1))),slip_proxy_mean=float(np.mean([x['contact_tangent_speed_max_mps'] for x in r])),slip_proxy_p95=float(np.percentile([x['contact_tangent_speed_max_mps'] for x in r],95)),max_tilt_deg=float(max(x['tilt_deg'] for x in r)),min_height=float(min(x['height'] for x in r)),torque_saturation_max=float(max(x['torque_saturation'] for x in r)),actuator_abs_work_proxy_J=float(np.abs(torque*dq).sum()*.005),actuator_signed_work_proxy_J=float((torque*dq).sum()*.005),external_work_J=float(sum(x['external_power']*.005 for x in r)),max_joint_limit_excess_rad=float(max(x['joint_limit_excess_max_rad'] for x in r)))
 for key,A in [('pelvis',P),('torso',T)]:
  yaw=np.unwrap(np.arctan2(A[:,1,0],A[:,0,0]));pitch=np.arctan2(-A[:,2,0],np.sqrt(A[:,2,1]**2+A[:,2,2]**2))
  ans[key+'_yaw_delta_deg']=float(np.degrees(yaw[-1]-yaw[0]));ans[key+'_yaw_mean_abs_deg']=float(np.abs(np.degrees(yaw)).mean());ans[key+'_pitch_mean_deg']=float(np.degrees(pitch).mean());ans[key+'_pitch_max_abs_deg']=float(np.abs(np.degrees(pitch)).max())
 if active:
  F=np.array([x['force'] for x in active]);E=np.array([np.array(x['anchor'])-x['wrist'] for x in active]);raw=np.array([x['contacts'][0]['raw_force'] for x in active]);fm=np.linalg.norm(F,axis=1)
  ans.update(error_cm=float(100*np.sqrt(np.mean((E*E).sum(1)))),error_xyz_rms_cm=(100*np.sqrt(np.mean(E*E,axis=0))).tolist(),signed_error_cm=(100*E.mean(0)).tolist(),force_rms=float(np.sqrt(np.mean(fm*fm))),force_xyz_rms=np.sqrt(np.mean(F*F,axis=0)).tolist(),force_xyz_mean=F.mean(0).tolist(),cap_fraction=float(np.mean(fm>=9.999)),force_peak=float(fm.max()),force_squared_time_average_N2=float(np.mean(fm*fm)),force_absolute_impulse_Ns=float(fm.sum()*.005),raw_demand_rms_N=float(np.sqrt(np.mean((raw*raw).sum(1)))))
 else:ans.update(error_cm=None,error_xyz_rms_cm=None,signed_error_cm=None,force_rms=0.,cap_fraction=None,force_squared_time_average_N2=0.)
 if len(c)>1:
  qp=np.array([x['q_PD'] for x in c]);dd=np.abs(np.diff(qp,axis=0));ix=np.unravel_index(np.argmax(dd),dd.shape);ans['qpd_peak']=dict(deg=float(np.degrees(dd[ix])),joint=names[ix[1]],time=c[ix[0]+1]['time'],sampling_s=.02);ans['qpd_p95_deg']=float(np.degrees(np.percentile(dd,95)));ans['qpd_p99_deg']=float(np.degrees(np.percentile(dd,99)))
 if len(q)>1:
  dd=np.abs(np.diff(q,axis=0));ix=np.unravel_index(np.argmax(dd),dd.shape);ans['actual_joint_peak']=dict(deg=float(np.degrees(dd[ix])),joint=names[ix[1]],time=float(ts[ix[0]+1]),sampling_s=.005);ans['actual_joint_speed_max_radps']=float(np.abs(dq).max())
 return ans

def events(ep):
 r=ep['physics'];onset=ep['condition']['onset'];t=np.array([x['force_time'] for x in r]);v=np.array([x['qvel'][0] for x in r]);foot=np.array([x['feet_xyz'] for x in r]);load=np.array([x['foot_normal_force_n'] for x in r]);start=next(i for i,x in enumerate(t) if x>=onset)
 root_event=None;step_event=None
 for i in range(start,len(r)-40):
  if root_event is None and np.all(v[i:i+40]>.05):root_event=float(t[i])
  if step_event is None:
   for side in [0,1]:
    if np.all(load[i:i+20,side]<5) and np.max(foot[i:i+100,side,0]-foot[start,side,0])>.02:
     step_event=dict(time=float(t[i]),side=['left','right'][side],definition='load<5N>=100ms and forwardfoot displacement>2cm within next500ms, descriptor only');break
 return dict(root_progress_time=root_event,foot_support_rearrangement=step_event,root_definition='vx>.05m/s>=200ms, not a universal gait success gate')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True);rows=[]
 names=json.loads((a.root/'runs/qualification-03/action-contract.json').read_text())['dof_names']
 for folder in ['qualification-03','reward-free-01','rolling-02']:
  d=a.root/'runs'/folder
  if not (d/'summaries.json').exists():continue
  for row in json.loads((d/'summaries.json').read_text()):
   path=d/(row['label']+'.json.gz');ep=read(path);s=ep['condition'];end=s.get('stop_time',s['onset']+s['ramp']);hold_start=end+s.get('deceleration',0.)
   out=dict(label=row['label'],run=folder,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),condition=s,boundary=ep.get('boundary'),events=events(ep),phases={})
   for key,lo,hi in [('pre',4,s['onset']),('moving',s['onset'],end),('decelerating',end,hold_start),('holding',hold_start,s['release']),('released',s['release'],s['end']),('active',s['onset'],s['release'])]:
    out['phases'][key]=stats(ep,lo,hi,names)
   rows.append(out)
 save(a.output/'physical-details.json',rows)
 # Real paired prefix verification, no condition or walltime metadata in equality.
 d=a.root/'runs/rolling-02';a1=d/'rolling_continue.json.gz';a2=d/'rolling_stop.json.gz'
 if a1.exists() and a2.exists():
  e1,e2=read(a1),read(a2);limit=7.8
  p1=[x for x in e1['physics'] if x['force_time']<limit-1e-8];p2=[x for x in e2['physics'] if x['force_time']<limit-1e-8]
  c1=[x for x in e1['control'] if x['time']<limit-1e-8];c2=[x for x in e2['control'] if x['time']<limit-1e-8]
  qs=json.loads((d/'searches.json').read_text());q1=[x for x in qs if x['label']=='rolling_continue' and x['time']<limit-1e-8];q2=[x for x in qs if x['label']=='rolling_stop' and x['time']<limit-1e-8]
  keys=['state_fingerprint','phase','privileged_input','restricted_ideal_force_input','previous_executed_z','selected_first_z','current_partner_position','current_partner_velocity']
  checks={k:all(a[k]==b[k] for a,b in zip(q1,q2)) for k in keys}
  res=dict(physical_prefix_equal=p1==p2,control_prefix_equal=c1==c2,planning_prefix_input_equal=checks,same_lengths=len(p1)==len(p2) and len(c1)==len(c2) and len(q1)==len(q2),physical_ticks=len(p1),control_steps=len(c1),planning_decisions=len(q1),fork=limit)
  save(a.output/'common-prefix.json',res)
 print('CPU_ANALYSIS',len(rows),flush=True)
if __name__=='__main__':main()
