"""Read recorded traces only; no model execution, GPU, fitting or simulation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUNS=P/'bridge-v2-owned/runs';R=Path(sys.argv[1]);O=R/'outputs'
read=lambda p:json.loads(p.read_text());arr=lambda rs,k:np.asarray([r[k] for r in rs],dtype=float)
protocol=read(O/'protocol.json');result=read(O/'result.json');assert result['status']=='completed' and result['projector_updates']==2000
S=RUNS/'issue21-bidirectional-20261030T012400Z/attempt-01/outputs';A=RUNS/'issue21-projector-20261030T004000Z/attempt-01/outputs'
D=RUNS/'issue21-direction-20261030T020000Z/attempt-02/outputs';T=RUNS/'issue21-directionteacher-20261030T031000Z/attempt-01/outputs';C=RUNS/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs'
checks={};comparisons={};episodes={};safety_keys=['root_min_m','tilt_max_deg','both_feet_loaded','joint_limit_excess_rad','action_saturation','torque_saturation','nonfoot_ground_frames']
standing=np.asarray(read(RUNS/'issue19-wrist-anchor-20261028T062503Z/attempt-02/outputs/standing_latent.json')).reshape(256)
for task in protocol['evaluations']:
 key=task['id'];short='prefix'in task;onset=task.get('prefix',task.get('onset'));sign='plus' if task['direction']==1 else 'minus'
 if key.startswith('dev3'):paths={'baseline':D/(sign+'-baseline.json'),'teacher':T/(sign+'-teacher.json'),'old-student':D/(sign+'-student.json'),'student':O/(key+'-student.json')}
 elif short:paths={mode:O/(key+'-'+mode+'.json') for mode in ['baseline','teacher','student']}
 else:paths={'baseline':(A if key=='dev-2' else C if key=='dev-3' else S)/(key+'-baseline.json'),'teacher':(A if key=='dev-2' else S)/(key+'-teacher.json'),'old-student':S/(key+'-student.json'),'student':O/(key+'-student.json')}
 eps={k:read(v) for k,v in paths.items()};episodes[key]={k:str(v) for k,v in paths.items()};b=eps['baseline'];bp=b['physics'];comparisons[key]={};checks[key]={}
 for mode,ep in eps.items():
  p=ep['physics'];c=ep['control'];assert ep['initial']==b['initial']
  assert np.array_equal(arr(p,'anchor'),arr(bp[:len(p)],'anchor'))
  assert len(p)==round(task['seconds']*200) and len(c)==round(task['seconds']*50)
  q=arr(c,'q_PD');nom=arr(c,'nominal_q_PD');dq=q-nom;f=arr(p,'force');w=arr(p,'wrist');root=arr(p,'root');initial=ep['initial']
  check=dict(failed=ep['failed'],root_min_m=float(arr(p,'height').min()),tilt_max_deg=float(arr(p,'tilt_deg').max()),
   both_feet_loaded=float(np.mean(np.all(arr(p,'foot_normal_force_n')>1e-6,axis=1))),joint_limit_excess_rad=float(arr(p,'joint_limit_excess_max_rad').max()),
   action_saturation=float(arr(c,'action_saturation').max()),torque_saturation=float(arr(p,'torque_saturation').max()),nonfoot_ground_frames=int(np.sum(arr(p,'undesired_ground_contacts')>0)),
   foot_speed_peak_mps=float(arr(p,'contact_tangent_speed_max_mps').max()),q_step_max_rad=float(abs(np.diff(q,axis=0)).max()),
   force_peak_n=float(np.linalg.norm(f,axis=-1).max()),arm_increment_rms_rad=float(np.sqrt(np.mean(dq[:,22:]**2))),nonarm_increment_rms_rad=float(np.sqrt(np.mean(dq[:,:22]**2))))
  if 'student'in mode:
   z=arr(c,'executed_z').reshape(-1,256);g=arr(c,'gate');check.update(norm_error_max=float(abs(np.linalg.norm(z,axis=-1)-16).max()),angle_max_deg=float(np.rad2deg(np.arccos(np.clip((z*standing).sum(-1)/256,-1,1))).max()),zero_gate_standing_exact=bool(np.array_equal(z[g==0],np.broadcast_to(standing,z[g==0].shape))))
   assert check['zero_gate_standing_exact'] and check['norm_error_max']<1e-4
  checks[key][mode]=check
  extra_w=w-arr(bp,'wrist');extra_root=root-arr(bp,'root');extra_rel=extra_w-extra_root;inc=np.diff(extra_rel,axis=0,prepend=np.zeros((1,3)))
  be=arr(bp,'anchor')-arr(bp,'wrist');bn=np.linalg.norm(be,axis=-1);unit=np.divide(be,bn[:,None],out=np.zeros_like(be),where=bn[:,None]>1e-6)
  windows=[('early',onset,onset+.2),('middle',onset+.2,onset+.5),('late',onset+.5,onset+1),('whole',onset,onset+1)] if short else [('pre',0,onset),('ramp',onset,onset+2),('hold',onset+2,4.5),('release',4.5,6.),('active',onset,4.5)]
  metrics={}
  for name,lo,hi in windows:
   pt=arr(p,'force_time');ct=arr(c,'time');tol=1e-9 if short else 0
   mask=(pt>=lo-tol)&(pt<hi-tol);cmask=(ct>=lo-tol)&(ct<hi-tol);ix=np.flatnonzero(mask);start=ix[0];end=ix[-1]
   before=initial if start==0 else p[start-1];err=arr(p,'error')[mask];force=f[mask];wr=w[end]-before['wrist'];rd=root[end]-before['root']
   metric=dict(error_rms_cm=float(np.sqrt(np.mean(err**2))*100),force_rms_n=float(np.sqrt(np.mean(np.sum(force*force,axis=-1)))),force_peak_n=float(np.linalg.norm(force,axis=-1).max()),force_mean_xyz_n=force.mean(0).tolist(),
    wrist_delta_xyz_cm=(wr*100).tolist(),root_delta_xyz_cm=(rd*100).tolist(),relative_delta_xyz_cm=((wr-rd)*100).tolist(),
    extra_relative_end_xyz_cm=(extra_rel[end]*100).tolist(),extra_relative_toward_common_baseline_error_cm=float((inc[mask]*unit[mask]).sum()*100),
    action_step_max_joint_deg=float(np.rad2deg(abs(np.diff(q,axis=0,prepend=q[:1]))[cmask].max())))
   if short:
    cd=arr(c,'rightarm_cartesian_target_delta')[cmask];ce=arr(c,'pre_error')[cmask];cn=np.linalg.norm(ce,axis=-1);cu=np.divide(ce,cn[:,None],out=np.zeros_like(ce),where=cn[:,None]>1e-6)
    metric['Jdq_mean_xyz_cm']=(cd.mean(0)*100).tolist();metric['Jdq_toward_own_error_mean_cm']=float((cd*cu).sum(-1).mean()*100)
   metrics[name]=metric
  comparisons[key][mode]=metrics
 # Report improvements and regressions, no outcome-based filtering.
 for phase in comparisons[key]['student']:
  base=comparisons[key]['baseline'][phase]['error_rms_cm'];new=comparisons[key]['student'][phase]['error_rms_cm'];teach=comparisons[key]['teacher'][phase]['error_rms_cm']
  comparisons[key]['student'][phase]['improvement_over_B_percent']=100*(1-new/base) if base>0 else None
  comparisons[key]['student'][phase]['teacher_gain_recovered']=(base-new)/(base-teach) if abs(base-teach)>1e-9 else None
  if 'old-student'in comparisons[key]:comparisons[key]['student'][phase]['change_vs_old_percent']=100*(new/comparisons[key]['old-student'][phase]['error_rms_cm']-1)
manifest=read(R/'manifest.json');sup=read(R/'supervisor-result.json')
analysis=dict(runtime_commit=manifest['worktree_head'],script_sha256=manifest['launcher_sha256'],checkpoint_sha256=hashlib.sha256(Path(result['checkpoint_path']).read_bytes()).hexdigest(),
 comparisons=comparisons,checks=checks,episodes=episodes,fit_before=read(O/'fit-before.json'),fit_after=read(O/'fit-after.json'),resources=sup)
(O/'repair-analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
for key,v in comparisons.items():
 for mode,m in v.items():
  phases=['whole'] if key.startswith(('dev3','new27')) else ['ramp','hold','active']
  print(key,mode,json.dumps({p:m[p] for p in phases}))
print('CHECKS',json.dumps(checks))
