"""Read-only information/command alignment audit of completed F1 L2 records."""
import argparse,hashlib,json,time,sys
from pathlib import Path
START=time.monotonic()
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--run',required=True);a=p.parse_args();S=a.stage;D=S/'runs'/a.run
sys.path.insert(0,str(S/'runtime-l2/research_notes'))
from f1_mjdata_contract import exact
rows=json.loads((D/'candidate-scores.json').read_text());queries=json.loads((D/'query-ledger.json').read_text());old=json.loads((S/'runs/l1-merged/candidate-scores.json').read_text());out=[];repeats=[]
required=['information_tier','actor_observation_schema','candidate_generator_inputs','partner_predictor_version','partner_predictor_inputs','score_features','simulator_state_origin','future_anchor_accessible','future_anchor_consumed','anchor_error_used_in_selection','selection_horizon','physical_eval_horizon','query_wall_seconds','state_parent_id','state_snapshot_sha','actual_runtime_sha']
try:
 for q in queries:assert not set(required)-set(q),('missing information fields',q['state_id'])
 for row in rows:
  if row['layer']=='L2-C-S' and not row['state_id'].endswith('-continue'):
   src=next(x for x in old if x['state_id']==row['state_id'] and x['candidate']==0)
   x=torch.load(S/'runs/l1-merged'/src['artifact'],map_location='cpu',weights_only=False);y=torch.load(D/row['artifact'],map_location='cpu',weights_only=False)
   match={k:exact(x[k],y[k]) for k in ['physics','controls','inputs','raws','target_z','previous_z','score']}
   repeats.append(dict(state_id=row['state_id'],same_actual_BC500_L1_vs_L2=match,passed=all(match.values())))
 pair='new_stop-sustained-5f6380cdec75'
 plan=json.loads((S/'prep/state-plan.json').read_text());cut=next(r['spec']['stop_time'] for r in plan['states'] if r['state_id']==pair)
 branches=[torch.load(D/next(r['artifact'] for r in rows if r['layer']=='L2-C-S' and r['state_id']==sid),map_location='cpu',weights_only=False) for sid in [pair,pair+'-continue']]
 pre_ph=[[r for r in x['physics'] if r['force_time']<cut-1e-7] for x in branches]
 pre_co=[[r for r in x['controls'] if r['time']<cut-1e-7] for x in branches]
 n=len(pre_co[0])//5
 assert n>0 and len(pre_ph[0])>0
 pre_match={k:exact(branches[0][k][:n],branches[1][k][:n]) for k in ['inputs','raws','target_z','previous_z']}
 pre_match.update(physics=exact(*pre_ph),controls=exact(*pre_co));assert all(pre_match.values()),('C-S stop/continue prefix',pre_match)
 cs_prefix=dict(physics_ticks=len(pre_ph[0]),commands=n,passed=True,fields=pre_match)
 def first_difference(left,right,key,time_key):
  for x,y in zip(left,right):
   if not exact(x[key],y[key]):return x[time_key]
  return None
 cs_prefix['first_force_difference_time']=first_difference(branches[0]['physics'],branches[1]['physics'],'force','force_time')
 cs_prefix['first_anchor_difference_time']=first_difference(branches[0]['physics'],branches[1]['physics'],'anchor','force_time')
 cs_prefix['first_restricted_input_difference_time']=next((branches[0]['controls'][5*i]['time'] for i,(x,y) in enumerate(zip(branches[0]['inputs'],branches[1]['inputs'])) if not exact(x[:,:1639],y[:,:1639])),None)
 cs_prefix['first_raw_difference_time']=next((branches[0]['controls'][5*i]['time'] for i,(x,y) in enumerate(zip(branches[0]['raws'],branches[1]['raws'])) if not exact(x,y)),None)

 actual={r['state_id']:torch.load(D/r['artifact'],map_location='cpu',weights_only=False) for r in rows if r['layer']=='L2-C-P'}
 for q in queries:
  if q['information_tier']!='C-P':continue
  r=next(x for x in rows if x['layer']=='L2-prediction' and x['state_id']==q['state_id'] and x['replan']==q['replan'] and x['candidate']==q['selected_candidate_id']);pred=torch.load(D/r['artifact'],map_location='cpu',weights_only=False);act=actual[q['state_id']];cmd=act['decisions'][q['replan']]
  assert exact(cmd['raw'],pred['raws'][0]),'executed command differs from chosen proposal'
  t=q['time'];control=next(c for c in act['controls'] if abs(c['time']-t)<1e-7)
  align={k:exact(control[k],pred['controls'][0][k]) for k in ['bfm_obs','executed_z','action','q_PD']}
  assert all(align.values()),('first BFM substep differs',q['state_id'],q['replan'],align)
  ph=[z for z in act['physics'] if t-1e-8<=z['force_time']<t+.1-1e-8];pr=pred['physics'][:len(ph)];assert len(ph)==len(pr)
  f=lambda rr:np.asarray([z['force'] for z in rr]);e=lambda rr:np.asarray([np.asarray(z['anchor'])-z['wrist'] for z in rr])
  out.append(dict(state_id=q['state_id'],parent=q['state_parent_id'],replan=q['replan'],selected=q['selected_candidate_id'],actual_raw_exact=True,first_substep_exact=align,query_state_sha=q['state_snapshot_sha'],force_prediction_error_rms_N=float(np.sqrt(np.mean(np.sum((f(ph)-f(pr))**2,axis=1)))),geometry_prediction_error_rms_cm=float(100*np.sqrt(np.mean(np.sum((e(ph)-e(pr))**2,axis=1)))),root_endpoint_prediction_error_cm=float(100*np.linalg.norm(np.asarray(ph[-1]['root'])-pr[-1]['root'])),interpretation='first 100ms same command; differences include constant-velocity forecast vs actual partner, not an oracle replay'))
 report=dict(status='passed',restricted_stop_continue_prefix=cs_prefix,required_fields=required,query_count=len(queries),causal_command_alignment=out,independent_BC500_repetition=repeats,all_BC500_repetitions_exact=all(x['passed'] for x in repeats),future_program_note='raw Task backup retains original future; candidate/score interfaces sanitize it, no claim of OS-level access isolation')
 (S/'reports/f1_query_alignment.json').write_text(json.dumps(report,indent=2,allow_nan=False));status='completed'
finally:
 (S/'jobs/query-reduction.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),layer='L2',status=locals().get('status','failed_preserved'),elapsed_s=time.monotonic()-START,physics_steps=0,queries=len(out)),indent=2))
