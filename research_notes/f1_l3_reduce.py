"""CPU-only final reduction of matched L3 actual and forecast records."""
import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
START=time.monotonic()
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--run',default='l3-02');a=p.parse_args();S=a.stage;D=S/'runs'/a.run
sys.path.insert(0,str(S/'runtime-l3/research_notes'))
from f1_mjdata_contract import exact
from frl_v2_task import continuation
from f1_sequence_search import semantics
from frl_v3_risk import score
def phase(r,s):
 t=r['force_time'];stop=s.get('stop_time',s['onset']+s['ramp']);hold=stop+s.get('deceleration',0.)
 if t<s['onset']:return 'precontact'
 if not r['active']:return 'released'
 return 'moving' if t<stop else 'decelerating' if t<hold else 'holding'

def base_summary(ph,co):
 active=[r for r in ph if r['active']];e=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);f=np.array([r['force'] for r in active]);q=np.array([r['qpos'][7:] for r in ph]);feet=np.array([r['feet_xyz'] for r in ph]);root=np.array([r['root'] for r in ph]);yaw=lambda r,k:math.degrees(math.atan2(r[k][1][0],r[k][0][0]))
 return dict(physics_ticks=len(ph),seconds=.005*len(ph),active_ticks=len(active),error_rms_cm=float(100*np.sqrt(np.mean(np.sum(e*e,axis=1)))) if active else None,error_xyz_rms_cm=(100*np.sqrt((e*e).mean(0))).tolist() if active else None,signed_error_xyz_cm=(100*e.mean(0)).tolist() if active else None,force_rms_N=float(np.sqrt(np.mean(np.sum(f*f,axis=1)))) if active else 0.,force_xyz_rms_N=np.sqrt((f*f).mean(0)).tolist() if active else [0.,0.,0.],cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=9.999)) if active else None,loaded_slip_peak_mps=max(r['stage_loaded_slip_mps'] for r in ph),loaded_slip_p90_mps=float(np.quantile([r['stage_loaded_slip_mps'] for r in ph],.9)),pelvis_yaw_max_abs_deg=max(abs(yaw(z,'pelvis_matrix')) for z in ph),torso_yaw_max_abs_deg=max(abs(yaw(z,'torso_matrix')) for z in ph),pelvis_yaw_start_deg=yaw(ph[0],'pelvis_matrix'),pelvis_yaw_final_deg=yaw(ph[-1],'pelvis_matrix'),torso_yaw_final_deg=yaw(ph[-1],'torso_matrix'),root_delta_cm=(100*(root[-1]-root[0])).tolist(),root_xy_path_cm=float(100*np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1).sum()),feet_path_cm=(100*np.linalg.norm(np.diff(feet,axis=0),axis=2).sum(0)).tolist(),actual_joint_step_max_deg=float(np.degrees(np.abs(np.diff(q,axis=0)).max())) if len(q)>1 else None,q_PD_step_max_deg=max((math.degrees(c['q_PD_step_max_rad']) for c in co),default=None),actual_torque_abs_max=float(np.abs([r['torque'] for r in ph]).max()),torque_saturation_max=max(r['torque_saturation'] for r in ph),max_joint_limit_excess_rad=max(r['joint_limit_excess_max_rad'] for r in ph),support_fraction=np.array([r['support'] for r in ph]).mean(0).tolist(),external_work_J=float(sum(r['external_power']*.005 for r in ph)))

def summary(ph,co):
 r=base_summary(ph,co)
 r.update(wrist_delta_xyz_cm=(100*(np.array(ph[-1]['wrist'])-ph[0]['wrist'])).tolist(),anchor_delta_xyz_cm=(100*(np.array(ph[-1]['anchor'])-ph[0]['anchor'])).tolist(),qdot_abs_max=float(np.abs([x['qvel'][6:] for x in ph]).max()))
 return r

result=[];alignment=[]
try:
 rows=json.loads((D/'candidate-scores.json').read_text());queries=json.loads((D/'query-ledger.json').read_text());old=json.loads((S/'runs/l2-01/candidate-scores.json').read_text());plan=json.loads((S/'prep/state-plan.json').read_text())
 for row in [r for r in rows if r['layer']=='L3-O-F']:
  sid=row['state_id'];spec=copy.deepcopy(next(r['spec'] for r in plan['states'] if r['state_id']==sid.removesuffix('-continue')))
  if sid.endswith('-continue'):spec=continuation(spec)
  paired={};traces={}
  for arm,folder,r in [('C-P',S/'runs/l2-01',next(r for r in old if r['layer']=='L2-C-P' and r['state_id']==sid)),('O-F',D,row)]:
   path=folder/r['artifact'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['artifact_sha256']
   x=torch.load(path,map_location='cpu',weights_only=False);traces[arm]=x;phases={}
   for k in ('precontact','moving','decelerating','holding','released'):
    ph=[z for z in x['physics'] if phase(z,spec)==k]
    if ph:
     co=[c for c in x['controls'] if ph[0]['force_time']-1e-8<=c['time']<=ph[-1]['force_time']+1e-8];phases[k]=summary(ph,co)
   paired[arm]=dict(artifact=r['artifact'],sha256=r['artifact_sha256'],metrics=r['metrics'],score=r['score'],boundary=r['boundary'],phases=phases,window=summary(x['physics'],x['controls']),slip=semantics(x['physics']))
   if arm=='O-F':
    for q in [q for q in queries if q['information_tier']=='O-F' and q['state_id']==sid]:
     assert q['number_of_simulated_candidates']==8 and q['selection_horizon']==2 and q['future_trajectory_consumed']
     rr=next(z for z in rows if z['layer']=='L3-O-F-prediction' and z['state_id']==sid and z['replan']==q['replan'] and z['candidate']==q['selected_candidate_id'])
     pred=torch.load(D/rr['artifact'],map_location='cpu',weights_only=False);cmd=x['decisions'][q['replan']]
     assert exact(cmd['raw'],pred['raws'][0]);t=q['time'];control=next(c for c in x['controls'] if abs(c['time']-t)<1e-7)
     fields={k:exact(control[k],pred['controls'][0][k]) for k in ['bfm_obs','executed_z','action','q_PD']};assert all(fields.values())
     ph=[z for z in x['physics'] if t-1e-8<=z['force_time']<t+.1-1e-8];pr=pred['physics'][:len(ph)]
     physical_keys=['qpos','qvel','force','anchor','support','feet_xyz']
     pe={k:all(exact(l[k],r[k]) for l,r in zip(ph,pr)) for k in physical_keys}
     assert all(pe.values()),('oracle first100ms physical mismatch',sid,q['replan'],pe)
     alignment.append(dict(state_id=sid,replan=q['replan'],raw_exact=True,first_BFM_substep=fields,first100ms_physics=pe))
  result.append(dict(state_id=sid,parent_id=row['parent_id'],spec=spec,arms=paired,actual_pair_physics_exact=exact(traces['C-P']['physics'],traces['O-F']['physics']),actual_pair_commands_exact=exact([d['raw'] for d in traces['C-P']['decisions']],[d['raw'] for d in traces['O-F']['decisions']])))
 reuse=json.loads((S/'reports/f1_l3_CP_reuse.json').read_text());assert reuse['status']=='passed';reused_queries=reuse['queries'];assert len(reused_queries)==90
 pair='new_stop-sustained-5f6380cdec75';oracle_pair=[torch.load(D/(sid+'-L3-OF.pt'),map_location='cpu',weights_only=False) for sid in (pair,pair+'-continue')]
 first_oracle_raw_difference=next((a['time'] for a,b in zip(oracle_pair[0]['decisions'],oracle_pair[1]['decisions']) if not exact(a['raw'],b['raw'])),None)
 report=dict(status='completed',oracle_sameprefix_first_command_difference_time=first_oracle_raw_difference,oracle_prefix_note='future-visible branches may diverge before observable split; not causal controller',C_P_reused_query_validation=reused_queries,paired=result,alignment=alignment,queries=len(alignment),parents=len(set(r['parent_id'] for r in result)),forecast_only_matched=True,source_runtime=json.loads((D/'launch.json').read_text())['runtime'])
 target=S/'reports/f1_l3_results.json';target.write_text(json.dumps(report,indent=2,allow_nan=False));status='completed'
finally:
 (S/'jobs/l3-reduction.json').write_text(json.dumps(dict(layer='L3',status=locals().get('status','failed_preserved'),elapsed_s=time.monotonic()-START,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_steps=0),indent=2))
