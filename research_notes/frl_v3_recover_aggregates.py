"""Recover only missing JSON summaries from already complete frozen branch archives."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();S=a.output.parents[1]
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else p.open() as f:return json.load(f)
def put(p,v):
 assert not p.exists(),p;p.write_text(json.dumps(v,indent=2,allow_nan=False))
old_failure=read(a.output/'failure.json');assert old_failure['error_type']=='TypeError' and 'bool_' in old_failure['error']
rows=read(a.output/'outcomes.json');prefix=read(a.output/'prefix-checks.json');plan=read(S/'prep/locked-protocol.json');ids=sorted(set(r['state_id'] for r in rows));assert ids==list(range(plan['P1']['states'])) and len(prefix)==len(ids)
assert sum(r['candidates'] for r in prefix)==len(rows)
credit=[];state_results=[]
for sid in ids:
 rr=[r for r in rows if r['state_id']==sid];base=next(r for r in rr if r['candidate']=='original_mean');sample=next((r for r in rr if r['candidate']=='archived_sample'),None)
 for r in rr:
  tag=f"state{sid:02d}-"+r['candidate'];assert (a.output/(tag+'.json.gz')).is_file() and (a.output/(tag+'-command.pt')).is_file()
 state_results.append(dict(state_id=sid,parent_id=base['parent_id'],phase=base['phase'],owner=base['state_owner'],candidates=len(rr),clean_success_candidates=sum(bool(r['metrics']['low_slip_task_success']) for r in rr)))
 if sample:
  ep=read(a.output/(f"state{sid:02d}-"+sample['candidate']+'.json.gz'));meta=ep['meta'];delta=float(sample['delta_return_vs_original_mean']);credit.append(dict(state_id=sid,parent_id=meta['parent_id'],historical_GAE=meta['historical_GAE'],PPO_adv=meta['historical_PPO_adv'],logprob_delta=meta['post_update_logprob_delta'],counterfactual_old_return_delta=delta,counterfactual_v3_return_delta=float(sample['delta_v3_return_vs_original_mean']),sample_clean_success=bool(sample['metrics']['low_slip_task_success']),adv_sign_agrees_with_counterfactual=bool((meta['historical_PPO_adv']>0)==(delta>0)),probability_sign_agrees_with_counterfactual=bool((meta['post_update_logprob_delta']>0)==(delta>0)),scope='finite2.1s CRN feedback return, not whole-episode ground truth advantage'))
put(a.output/'historical-credit.json',credit);put(a.output/'state-results.json',state_results)
success=[r for r in rows if r['metrics']['low_slip_task_success']];useful=[r for r in success if r['phase']!='entry' and r['metrics']['partner_dx_m']>.02]
gate=dict(P2_allowed=bool(useful),existence_answer='exists within measured subset' if useful else 'not established within bounded candidates',clean_success_states=len(set(r['state_id'] for r in success)),nontrivial_success_states=len(set(r['state_id'] for r in useful)),nontrivial_parents=len(set(r['parent_id'] for r in useful)),states=len(ids),parents=plan['P1']['parents'],candidates=len(rows),new_training_updates=0,normal_threshold_unchanged_N=1e-6,recovered_from_complete_actual_data=True,original_collection_process_status='failed only during final NumPy bool JSON serialization; no physical rerun or overwrite',job_s=time.monotonic()-start)
put(a.output/'gate.json',gate);put(a.output/'complete-recovery.json',dict(status='completed',complete_collection_verified=True,physical_candidates=len(rows),states=len(ids),raw_outcomes_sha256=hashlib.sha256((a.output/'outcomes.json').read_bytes()).hexdigest(),old_failure_preserved=True,new_physics_steps=0,new_model_updates=0,elapsed_s=time.monotonic()-start))
print(json.dumps(gate,indent=2))
