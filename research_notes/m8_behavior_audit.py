"""Post-test saved-trace diagnosis of active control and release; no new physics."""
import json
import numpy as np
from m8_analysis import RUN,read

def main():
 out=RUN/'final-attempt-01/outputs';names=read(out/'action-contract.json')['dof_names'];results={}
 for name in ['down_test','down_position','wrist_early','wrist_neg_early','wrist_test_pos_small','wrist_sealed_mid']:
  for ctl in ['distill','alignment']+(['m7'] if name=='wrist_sealed_mid' else []):
   e=read(out/(name+'__'+ctl+'.json.gz'));cs=e['control'];r=e['physics'];s=e['condition'];t=np.array([c['time'] for c in cs]);pt=np.array([x['time'] for x in r]);release=pt>=s['release'];active=np.array([x['active'] for x in r]);q=np.array([c['q_PD'] for c in cs]);nom=np.array([c['standing_q_PD'] for c in cs]);pre=np.array([c['pre_qpos'] for c in cs]);j=max(range(len(cs)),key=lambda i:cs[i]['q_PD_step_max_rad']);k=int(np.argmax(abs(q[j]-q[max(0,j-1)])));rootv=np.linalg.norm(np.array([x['qvel'][:3] for x in r]),axis=1)
   raw=None if cs[0]['student_raw'] is None else np.array([c['student_raw'] for c in cs]);progress=None if raw is None else np.clip(raw[:,:2],0,1)
   extra=q-nom;ca=(t>=s['onset'])&(t<s['release']);feet=np.array([x['feet_xyz'] for x in r]);foot0=np.array(e['initial']['feet_xyz']);peak=int(np.flatnonzero(release)[np.argmax(rootv[release])]);after2=t>=s['release']+2
   jump=dict(time=float(t[j]),joint=names[k],qpd_step_deg=float(np.degrees(cs[j]['q_PD_step_max_rad'])),gate_before=cs[max(0,j-1)].get('local_gate'),gate_after=cs[j].get('local_gate'),z_step_deg=cs[j]['z_step_deg'],actual_same_joint_next20ms_deg=float(np.degrees(pre[min(j+1,len(cs)-1),7+k]-pre[j,7+k])),actual_all_joints_next20ms_max_deg=float(np.degrees(abs(pre[min(j+1,len(cs)-1),7:]-pre[j,7:]).max())))
   if progress is not None:jump.update(progress_before=progress[max(0,j-1)].tolist(),progress_after=progress[j].tolist())
   results[e['label']]=dict(largest_qpd_jump=jump,release_root_speed_peak_mps=float(rootv[peak]),release_peak_time=float(pt[peak]),release_max_speed_after2_mps=float(rootv[pt>=s['release']+2].max()),foot_xy_displacement_max_cm=float(100*np.linalg.norm(feet[:,:,:2]-foot0[None,:,:2],axis=2).max()),foot_contact_tangent_speed_max_mps=max(x['contact_tangent_speed_max_mps'] for x in r),same_state_qpd_extra_vs_standing_rms_deg=float(np.degrees(np.sqrt(np.mean(extra[ca]**2)))),same_state_knee_target_extra_mean_deg=np.degrees(extra[ca][:,[3,9]].mean(0)).tolist(),precontact_progress_max=None if progress is None else progress[t<s['onset']].max(0).tolist(),release_progress_peak_after2=None if progress is None else progress[after2].max(0).tolist(),final_progress=None if progress is None else progress[-1].tolist())
 (RUN/'analysis/behavior-audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
