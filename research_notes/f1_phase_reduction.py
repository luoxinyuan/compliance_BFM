"""Read saved actual/branch trajectories only; no dynamics and no candidate selection."""
import argparse,json,math,time
from pathlib import Path
START=time.monotonic()
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--l2');a=p.parse_args();S=a.stage
P=S.parents[1];L0=P/'experiments/f1-l0-array-audit-20261111'
manifest=json.loads((L0/'runs/l0-complete-02/state-manifest.json').read_text());specs={}
for r in manifest:
 if r.get('role')!='source_positive':specs[r['state_id']]=torch.load(L0/r['state_snapshot'],map_location='cpu',weights_only=False)['spec']

def phase(r,s):
 t=r['force_time'];stop=s.get('stop_time',s['onset']+s['ramp']);hold=stop+s.get('deceleration',0.)
 if t<s['onset']:return 'precontact'
 if not r['active']:return 'released'
 return 'moving' if t<stop else 'decelerating' if t<hold else 'holding'

def summary(ph,co):
 active=[r for r in ph if r['active']];e=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);f=np.array([r['force'] for r in active]);q=np.array([r['qpos'][7:] for r in ph]);feet=np.array([r['feet_xyz'] for r in ph]);root=np.array([r['root'] for r in ph]);yaw=lambda r,k:math.degrees(math.atan2(r[k][1][0],r[k][0][0]))
 return dict(physics_ticks=len(ph),seconds=.005*len(ph),active_ticks=len(active),error_rms_cm=float(100*np.sqrt(np.mean(np.sum(e*e,axis=1)))) if active else None,error_xyz_rms_cm=(100*np.sqrt((e*e).mean(0))).tolist() if active else None,signed_error_xyz_cm=(100*e.mean(0)).tolist() if active else None,force_rms_N=float(np.sqrt(np.mean(np.sum(f*f,axis=1)))) if active else 0.,force_xyz_rms_N=np.sqrt((f*f).mean(0)).tolist() if active else [0.,0.,0.],cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=9.999)) if active else None,loaded_slip_peak_mps=max(r['stage_loaded_slip_mps'] for r in ph),loaded_slip_p90_mps=float(np.quantile([r['stage_loaded_slip_mps'] for r in ph],.9)),pelvis_yaw_max_abs_deg=max(abs(yaw(z,'pelvis_matrix')) for z in ph),torso_yaw_max_abs_deg=max(abs(yaw(z,'torso_matrix')) for z in ph),pelvis_yaw_start_deg=yaw(ph[0],'pelvis_matrix'),pelvis_yaw_final_deg=yaw(ph[-1],'pelvis_matrix'),torso_yaw_final_deg=yaw(ph[-1],'torso_matrix'),root_delta_cm=(100*(root[-1]-root[0])).tolist(),root_xy_path_cm=float(100*np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1).sum()),feet_path_cm=(100*np.linalg.norm(np.diff(feet,axis=0),axis=2).sum(0)).tolist(),actual_joint_step_max_deg=float(np.degrees(np.abs(np.diff(q,axis=0)).max())) if len(q)>1 else None,q_PD_step_max_deg=max((math.degrees(c['q_PD_step_max_rad']) for c in co),default=None),actual_torque_abs_max=float(np.abs([r['torque'] for r in ph]).max()),torque_saturation_max=max(r['torque_saturation'] for r in ph),max_joint_limit_excess_rad=max(r['joint_limit_excess_max_rad'] for r in ph),support_fraction=np.array([r['support'] for r in ph]).mean(0).tolist(),external_work_J=float(sum(r['external_power']*.005 for r in ph)))

result=[]
try:
 for run in ['l1-merged']+([a.l2] if a.l2 else []):
  rows=json.loads((S/'runs'/run/'candidate-scores.json').read_text());queries=json.loads((S/'runs'/run/'query-ledger.json').read_text());selected={q['state_id']:q['selected_candidate_id'] for q in queries if q['information_tier']=='O-F'}
  for r in rows:
   if r['layer']=='L2-prediction':continue
   if r['layer']=='L1' and r['candidate'] not in (0,selected[r['state_id']]):continue
   x=torch.load(S/'runs'/run/r['artifact'],map_location='cpu',weights_only=False);spec=specs[r['state_id'].removesuffix('-continue')].copy()
   if r['state_id'].endswith('-continue'):
    # Original continuation removes stopping keys; task code is authoritative.
    import sys
    sys.path.insert(0,str(S/'runtime-l2/research_notes'))
    from frl_v2_task import continuation
    spec=continuation(spec)
   phases={}
   for k in ('precontact','moving','decelerating','holding','released'):
    ph=[z for z in x['physics'] if phase(z,spec)==k]
    if ph:
     co=[c for c in x['controls'] if ph[0]['force_time']-1e-8<=c['time']<=ph[-1]['force_time']+1e-8]
     phases[k]=summary(ph,co)
   result.append(dict(run=run,state_id=r['state_id'],parent_id=r['parent_id'],layer=r['layer'],task_family=spec.get('demand_family','mid'),candidate=r.get('candidate'),artifact=r['artifact'],sha256=r['artifact_sha256'],phases=phases,window=summary(x['physics'],x['controls'])))
 (S/'reports/f1_phase_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));status='completed'
finally:
 (S/'jobs/phase-reduction.json').write_text(json.dumps(dict(layer='L2' if a.l2 else 'L1',status=locals().get('status','failed_preserved'),elapsed_s=time.monotonic()-START,physics_steps=0,rows=len(result)),indent=2))
