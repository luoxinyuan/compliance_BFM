"""Read saved real physics, never invoke a model or alter a run."""
import json,gzip,math
from pathlib import Path
import numpy as np

def summarize(p):
 e=json.loads(gzip.decompress(Path(p).read_bytes()));rs=e['physics'];cs=e['control'];active=[r for r in rs if r['active']];s=e['condition'];q0=np.array(e['initial']['qpos']);root=np.array([r['root'] for r in rs]);rr=np.array([r['wrist'] for r in rs]);torso=np.array([r['torso_matrix'] for r in rs]);pelvis=np.array([r['pelvis_matrix'] for r in rs]);pitch=lambda R:np.degrees(np.arctan2(-R[:,2,0],np.sqrt(R[:,2,1]**2+R[:,2,2]**2)))
 out=dict(path=str(p),label=e['label'],condition=s,complete=e['boundary'] is None,boundary=e['boundary'],root_displacement_xyz=(root[-1]-q0[:3]).tolist(),root_max_displacement=float(np.linalg.norm(root-q0[:3],axis=1).max()),height_min=float(root[:,2].min()),tilt_max=max(r['tilt_deg'] for r in rs),qPD_peak_deg=float(np.degrees(max(c['q_PD_step_max_rad'] for c in cs))),qPD_p95_deg=float(np.degrees(np.percentile([c['q_PD_step_max_rad'] for c in cs],95))),joint20ms_peak_deg=float(np.degrees(np.abs(np.diff(np.array([c['pre_qpos'][7:] for c in cs]),axis=0)).max())),world_wrist_drop_cm=100*max(0,e['initial']['wrist'][2]-rr[:,2].min()),torso_pitch_mean=float(pitch(torso).mean()),pelvis_pitch_mean=float(pitch(pelvis).mean()),action_clip=max(c['action_saturation_fraction'] for c in cs),torque_saturation=max(r['torque_saturation'] for r in rs),joint_limit_excess=max(r['joint_limit_excess_max_rad'] for r in rs),single_support=float(np.mean([sum(r['support'])==1 for r in rs])),unsupported=float(np.mean([sum(r['support'])==0 for r in rs])),slip_p95=float(np.percentile([r['contact_tangent_speed_max_mps'] for r in rs],95)))
 if active:
  F=np.array([r['force'] for r in active]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);out.update(force_rms=float(np.sqrt(np.sum(F*F,1).mean())),force_xyz_rms=np.sqrt(np.mean(F*F,0)).tolist(),error_cm=float(100*np.sqrt(np.sum(E*E,1).mean())),error_xyz_cm=(100*np.sqrt(np.mean(E*E,0))).tolist(),cap_fraction=float((np.linalg.norm(F,axis=1)>=4.999).mean()))
 else:out.update(force_rms=0,error_cm=None,cap_fraction=0)
 move=[r for r in rs if r['active'] and np.linalg.norm(r['anchor_velocity'][:2])>.02 and np.linalg.norm(r['qvel'][:2])>.05]
 if move:
  R=np.array([r['pelvis_matrix'] for r in move]);v=np.array([r['qvel'][:2] for r in move]);vp=np.array([r['anchor_velocity'][:2] for r in move]);yaw=np.arctan2(R[:,1,0],R[:,0,0]);target=np.arctan2(vp[:,1],vp[:,0]);diff=np.arctan2(np.sin(yaw-target),np.cos(yaw-target));out.update(heading_rms_deg=float(np.degrees(np.sqrt(np.mean(diff**2)))),moving_speed_mean=float(np.linalg.norm(v,axis=1).mean()),partner_speed_same_times=float(np.linalg.norm(vp,axis=1).mean()),moving_eval_seconds=len(move)/200)
 phases={}
 for name,lo,hi in [('pre',3,s['onset']),('move',s['onset'],s['onset']+s['ramp']),('hold',s['onset']+s['ramp'],s['release']),('recover',s['release'],s['end'])]:
  r=[r for r in rs if lo<=r['force_time']<hi]
  if not r:continue
  F=np.array([v['force'] for v in r]);vel=np.array([v['qvel'][:3] for v in r]);phases[name]=dict(seconds=len(r)/200,force_rms=float(np.sqrt(np.mean(np.sum(F*F,1)))),root_speed_peak=float(np.linalg.norm(vel,axis=1).max()),root_speed_last_second=float(np.linalg.norm(vel[-200:],axis=1).mean()))
 if s.get('prefix_until'):
  tail=[r for r in active if r['force_time']>=s['prefix_until']]
  if tail:
   F=np.array([r['force'] for r in tail]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in tail]);out['post_prefix']=dict(start=s['prefix_until'],seconds=len(tail)/200,force_rms=float(np.sqrt(np.sum(F*F,1).mean())),error_cm=float(100*np.sqrt(np.sum(E*E,1).mean())))
 out['residual_norm_max']=max(c.get('residual_uncapped_norm',0) for c in cs)
 out['phases']=phases;return out

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();rows=[summarize(f) for f in sorted(a.folder.glob('*.json.gz'))];(a.folder/'physical-metrics.json').write_text(json.dumps(rows,indent=2));print(json.dumps([{k:r.get(k) for k in ['label','complete','error_cm','force_rms','cap_fraction','root_max_displacement','world_wrist_drop_cm','qPD_peak_deg','heading_rms_deg']} for r in rows],indent=2))
