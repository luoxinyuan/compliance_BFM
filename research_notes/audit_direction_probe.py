"""Pure recorded-data audit of paired real-contact direction branches."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);O=R/'outputs';read=lambda p:json.loads(p.read_text());arr=lambda rows,k:np.asarray([r[k] for r in rows],dtype=float)
result=read(O/'result.json');assert result['status']=='completed' and result['policy_updates']==0
protocol=read(O/'protocol.json');state=read(O/'common-state.json');t0=3.
rows={};checks={};comparison={}
for sign in ['plus','minus']:
 rows[sign]={};comparison[sign]={}
 for label in protocol['modes']:
  name=sign+'-'+label;ep=read(O/(name+'.json'));rows[sign][label]=ep;p=ep['physics'];c=ep['control'];d=ep['decisions']
  assert len(p)==200 and len(c)==50 and len(d)==10 and not ep['failed']
  assert read(O/(name+'-prefix-parity.json'))==dict(qpos=0.,qvel=0.,wrist=0.,root=0.,force=0.)
  assert d[0]['force_history']==state['force_history'];assert c[0]['bfm_obs']==state['bfm_obs']
  q=arr(c,'q_PD');dq=arr(c,'student_delta_q');f=arr(p,'force');z=arr(c,'executed_z').reshape(50,256);g=arr(c,'gate')
  active=g>0
  check=dict(initial=read(O/(name+'-branch-initial.json')),same_state_hash=ep['common_state_sha256'],
   force_mean_xyz_n=f.mean(0).tolist(),force_rms_xyz_n=np.sqrt(np.mean(f*f,0)).tolist(),force_rms_n=float(np.sqrt(np.mean(np.sum(f*f,axis=-1)))),force_peak_n=float(np.linalg.norm(f,axis=-1).max()),
   force_x_positive_fraction=float(np.mean(f[:,0]>0)),force_x_negative_fraction=float(np.mean(f[:,0]<0)),force_x_signed_impulse_ns=float(f[:,0].sum()*.005),
   wrist_delta_xyz_mm=(1000*(arr(p,'wrist')[-1]-ep['initial']['wrist'])).tolist(),root_delta_xyz_mm=(1000*(arr(p,'root')[-1]-ep['initial']['root'])).tolist(),
   root_min_m=min(r['height'] for r in p),tilt_max_deg=max(r['tilt_deg'] for r in p),both_feet_loaded_fraction=float(np.mean([all(x>1e-6 for x in r['foot_normal_force_n']) for r in p])),
   foot_speed_peak_mps=max(r['contact_tangent_speed_max_mps'] for r in p),joint_limit_excess_max=max(r['joint_limit_excess_max_rad'] for r in p),
   action_saturation=max(r['action_saturation'] for r in c),torque_saturation=max(r['torque_saturation'] for r in p),unwanted_ground_frames=sum(r['undesired_ground_contacts']>0 for r in p),
   q_PD_step_max_joint_rad=float(abs(np.diff(q,axis=0)).max()),q_PD_step_max_l2_rad=float(np.linalg.norm(np.diff(q,axis=0),axis=-1).max()),
   arm_delta_q_rms_rad=float(np.sqrt(np.mean(dq[active,22:]**2))),nonarm_delta_q_rms_rad=float(np.sqrt(np.mean(dq[active,:22]**2))),
   arm_cartesian_q_delta_mean_mm=(1000*arr(c,'rightarm_cartesian_target_delta')[active].mean(0)).tolist(),z_norm_error_max=float(abs(np.linalg.norm(z,axis=-1)-16).max()))
  check['relative_delta_xyz_mm']=(np.array(check['wrist_delta_xyz_mm'])-check['root_delta_xyz_mm']).tolist()
  standing=np.asarray(state['previous_z']);check['angle_max_deg']=float(np.rad2deg(np.arccos(np.clip((z*standing).sum(-1)/256,-1,1))).max())
  check['zero_gate_exact']=bool(np.array_equal(z[g==0],np.broadcast_to(standing.reshape(256),z[g==0].shape)))
  checks[name]=check
 for label in protocol['modes']:
  ep=rows[sign][label];b=rows[sign]['baseline'];p=ep['physics'];bp=b['physics'];c=ep['control']
  assert ep['initial']==b['initial'];assert ep['decisions'][0]['student_h']==b['decisions'][0]['student_h']
  assert np.array_equal(arr(p,'anchor'),arr(bp,'anchor'))
  # Additional movement relative to baseline; project each increment onto SAME baseline current error.
  extra_w=arr(p,'wrist')-arr(bp,'wrist');extra_r=arr(p,'root')-arr(bp,'root');extra_rel=extra_w-extra_r
  inc=np.diff(extra_w,axis=0,prepend=np.zeros((1,3)));inc_rel=np.diff(extra_rel,axis=0,prepend=np.zeros((1,3)))
  err=arr(bp,'anchor')-arr(bp,'wrist');norm=np.linalg.norm(err,axis=-1);unit=np.divide(err,norm[:,None],out=np.zeros_like(err),where=norm[:,None]>1e-6)
  values=[]
  for lo,hi in protocol['windows_relative_s']:
   mask=(arr(p,'force_time')>=t0+lo-1e-9)&(arr(p,'force_time')<t0+hi-1e-9)
   ids=np.flatnonzero(mask);f=arr(p,'force')[mask];e=arr(p,'error')[mask]
   cmask=(arr(c,'time')>=t0+lo-1e-9)&(arr(c,'time')<t0+hi-1e-9)&(arr(c,'gate')>0)
   cd=arr(c,'rightarm_cartesian_target_delta')[cmask];ce=arr(c,'pre_error')[cmask];cn=np.linalg.norm(ce,axis=-1)
   cu=np.divide(ce,cn[:,None],out=np.zeros_like(ce),where=cn[:,None]>1e-6)
   values.append(dict(start_relative_s=lo,end_relative_s=hi,error_rms_mm=float(np.sqrt(np.mean(e**2))*1000),baseline_error_rms_mm=float(np.sqrt(np.mean(arr(bp,'error')[mask]**2))*1000),
    force_mean_xyz_n=f.mean(0).tolist(),force_rms_n=float(np.sqrt(np.mean(np.sum(f*f,axis=-1)))),
    extra_wrist_end_xyz_mm=(1000*extra_w[ids[-1]]).tolist(),extra_relative_end_xyz_mm=(1000*extra_rel[ids[-1]]).tolist(),
    extra_wrist_increment_toward_baseline_error_mm=float((inc[mask]*unit[mask]).sum()*1000),extra_relative_increment_toward_baseline_error_mm=float((inc_rel[mask]*unit[mask]).sum()*1000),
    arm_cartesian_q_delta_mean_mm=None if not len(cd) else (1000*cd.mean(0)).tolist(),
    arm_cartesian_q_delta_toward_own_error_mean_mm=None if not len(cd) else float((cd*cu).sum(-1).mean()*1000)))
  comparison[sign][label]=values
# Mechanical and observable branch start identical ACROSS signs too.
for label in protocol['modes']:
 assert rows['plus'][label]['initial']==rows['minus'][label]['initial']
 assert rows['plus'][label]['decisions'][0]['student_h']==rows['minus'][label]['decisions'][0]['student_h']
# Runtime compares full numerical snapshots; JSON signed-zero hashes alone are not equality proof.
engineering_parity={}
prior=R.parent/'attempt-01/outputs/plus-baseline.json'
if prior.exists():
 old=read(prior);new=rows['plus']['baseline']
 for key in ['qpos','qvel','force','wrist','root','torque']:
  engineering_parity[key]=float(abs(arr(old['physics'],key)-arr(new['physics'],key)).max())
 assert max(engineering_parity.values())==0
manifest=read(R/'manifest.json');sup=read(R/'supervisor-result.json')
analysis=dict(engineering_retry_baseline_parity=engineering_parity,checks=checks,comparison=comparison,protocol=protocol,runtime_commit=manifest['worktree_head'],script_sha256=manifest['launcher_sha256'],
 resources={k:sup[k] for k in ['pid','pgid','wall_seconds','peak_rss_bytes','peak_gpu_mib','remaining_processes','shared_source_unchanged','resource_boundary']})
(O/'direction-analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
for sign,v in comparison.items():
 for label,metrics in v.items():print(sign,label,json.dumps(metrics[-1]))
print('CHECKS',json.dumps(checks));print('RESOURCES',json.dumps(analysis['resources']))
