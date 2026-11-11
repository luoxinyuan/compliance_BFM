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

old=json.loads((S/'runs/l2-01/candidate-scores.json').read_text())
try:
 reused_queries=[]
 oldqueries=json.loads((S/'runs/l2-01/query-ledger.json').read_text())
 sids={'new_tail-hold-97ff10c4dcd9','new_stop-sustained-5f6380cdec75','new_stop-sustained-5f6380cdec75-continue'}
 for q in oldqueries:
  if q['information_tier']!='C-P' or q['state_id'] not in sids:continue
  assert not q['future_trajectory_consumed'] and q['number_of_simulated_candidates']==8 and q['selection_horizon']==2
  statepath=S/'runs/l2-01'/q['current_state_artifact']
  assert hashlib.sha256(statepath.read_bytes()).hexdigest()==q['state_snapshot_sha']
  matched=[r for r in old if r['layer']=='L2-prediction' and r['state_id']==q['state_id'] and r['replan']==q['replan']]
  assert len(matched)==8
  for r in matched:
   pred=torch.load(S/'runs/l2-01'/r['artifact'],map_location='cpu',weights_only=False)
   assert exact(score(pred['physics'],pred['controls'],q['time']),r['score']),('archived score recompute mismatch',q['state_id'],q['replan'],r['candidate'])
  safe=[r for r in matched if r['boundary'] is None]
  selected=max(safe or matched,key=lambda r:r['score']['v3_discounted_return'])['candidate']
  assert selected==q['selected_candidate_id']
  reused_queries.append(dict(state_id=q['state_id'],replan=q['replan'],snapshot_sha=q['state_snapshot_sha'],all8_scores_exact=True,selection_exact=True))
 assert len(reused_queries)==90
 (S/'reports/f1_l3_CP_reuse.json').write_text(json.dumps(dict(status='passed',queries=reused_queries),indent=2));status='completed'
finally:
 (S/'jobs/l3-cp-audit.json').write_text(json.dumps(dict(layer='L3',status=locals().get('status','failed_preserved'),elapsed_s=time.monotonic()-START,physics_steps=0,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
