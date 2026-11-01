"""Saved-trajectory metrics only; never loads BFM or steps physics."""
import json,gzip,sys
from pathlib import Path
import numpy as np
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m8-contact-body-20261101A')
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if str(p).endswith('.gz') else json.loads(p.read_text())
def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def analyze(e):
 rows=e['physics'];cs=e['control'];r={k:np.asarray([x[k] for x in rows]) for k in ['time','force_time','force','wrist','anchor','root','qpos','qvel','tilt_deg','foot_normal_force_n','feet_xyz','joint_limit_excess_max_rad','torque_saturation']};q=np.asarray([c['q_PD'] for c in cs]);t=r['force_time'];active=np.asarray([x['active'] for x in rows]);s=e['condition'];on=s.get('onset',3.8);release=s.get('release',9.);ramp=s.get('ramp',3.);phases={'all_contact':(t>=on)&(t<release)&active,'move':(t>=on)&(t<on+ramp)&active,'hold':(t>=on+ramp)&(t<release)&active,'release':t>=release,'native_hold':(t>=7)&(t<9)};metrics={}
 if s.get('transition'):
  tr=s['transition'];phases.update(forward=(t>=on)&(t<tr['unload'])&active,reverse=(t>=tr['restart'])&(t<release)&active,reverse_hold=(t>=tr['reverse_end'])&(t<release)&active)
 for name,mask in phases.items():
  if not mask.any():continue
  err=r['anchor'][mask]-r['wrist'][mask];f=r['force'][mask];pos=r['wrist'][mask];root=r['root'][mask];first=e['initial'];qpos=r['qpos'][mask]
  metrics[name]=dict(samples=int(mask.sum()),tracking_3d_rms_cm=100*rms(np.linalg.norm(err,axis=1)),error_xyz_rms_cm=(100*np.sqrt(np.mean(err**2,0))).tolist(),force_rms_n=rms(np.linalg.norm(f,axis=1)),force_peak_n=float(np.linalg.norm(f,axis=1).max()),force_xyz_mean_n=f.mean(0).tolist(),force_xyz_rms_n=np.sqrt(np.mean(f*f,0)).tolist(),cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=4.999)),point_delta_xyz_cm=(100*(pos[-1]-first['wrist'])).tolist(),root_delta_xyz_cm=(100*(root[-1]-first['root'])).tolist(),knee_mean_deg=np.degrees(qpos[:,[10,16]].mean(0)).tolist(),hip_mean_deg=np.degrees(qpos[:,[7,13]].mean(0)).tolist(),root_height_min_cm=float(100*root[:,2].min()),tilt_max_deg=float(r['tilt_deg'][mask].max()),feet_force_min_n=r['foot_normal_force_n'][mask].min(0).tolist(),root_speed_peak_mps=float(np.linalg.norm(r['qvel'][mask,:3],axis=1).max()),point_relative_root_delta_xyz_cm=(100*((pos[-1]-root[-1])-(np.asarray(first['wrist'])-first['root']))).tolist(),waist_mean_deg=np.degrees(qpos[:,19:22].mean(0)).tolist(),torso_rotation_peak_deg=float(np.degrees(np.arccos(np.clip((np.einsum('nij,ij->n',np.asarray([x['torso_matrix'] for x in rows])[mask],np.asarray(first['torso_matrix']))-1)/2,-1,1))).max()))
 result=dict(label=e['label'],boundary=e.get('boundary'),phases=metrics,seconds=len(rows)/200,root_height_range_cm=(100*np.array([r['root'][:,2].min(),r['root'][:,2].max()])).tolist(),tilt_max_deg=float(r['tilt_deg'].max()),qpd_step_max_deg=float(np.degrees(max(c.get('q_PD_step_max_rad',0) for c in cs))),actual_joint_20ms_max_deg=float(np.degrees(np.abs(np.diff(np.vstack([np.asarray([c['pre_qpos'] for c in cs]),r['qpos'][-1]]),axis=0)[:,7:]).max())) if len(cs)>1 else 0,final_root_speed_mps=float(np.linalg.norm(r['qvel'][-1,:3])),final_root_delta_cm=(100*(r['root'][-1]-e['initial']['root'])).tolist(),feet_force_min_n=r['foot_normal_force_n'].min(0).tolist(),joint_limit_excess_max_rad=float(r['joint_limit_excess_max_rad'].max()),torque_saturation=float(r['torque_saturation'].max()),action_saturation=max(c.get('action_saturation_fraction',0) for c in cs),no_foot_support_fraction=float(np.mean((r['foot_normal_force_n']<1).all(1))))
 result['release_root_speed_2_4_6_8_mps']=[float(np.linalg.norm(r['qvel'][np.argmin(abs(r['time']-(release+dt))),:3])) for dt in [2,4,6,8]]
 result['tracking_scope']='Only active real contact contributes all_contact/move/hold/reverse; release/native_hold geometry not active following.'
 return result
if __name__=='__main__':
 stage=sys.argv[1];attempts=[a for a in read(RUN/'compute-ledger.json')['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(attempts)==1
 out=Path(attempts[0]['path'])/'outputs';results=[]
 for p in sorted(out.glob('*.json.gz')):
  e=read(p)
  if e['physics']:results.append(analyze(e))
 (RUN/'analysis').mkdir(exist_ok=True);(RUN/'analysis'/(stage+'.json')).write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
