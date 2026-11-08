"""Only saved real/prediction summaries; no model/physics/query/training calls."""
from pathlib import Path
import json,gzip,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent;out=S/'runs/final-analysis-01';out.mkdir(exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
D=S/'runs/diagnose-01';queries=read(D/'searches.json');shadow=[q for q in queries if 'shadow' in q];gate=read(D/'gate.json');rec=read(D/'recoveries.json');main=read(D/'summaries.json')
def summary(x):return dict(n=len(x),mean=float(np.mean(x)),p50=float(np.median(x)),p90=float(np.quantile(x,.9)),minimum=float(min(x)),maximum=float(max(x)))
basequery=[q for q in queries if 'shadow' not in q];seq=[c for q in queries for c in q['candidates']];nominal=True
# Count actual actor calls rather than historical field named BFM_calls, which counts only controls.
with gzip.open(D/'S4000_normal72.json.gz','rt') as f:sample=json.load(f)
nominal=all(c.get('nominal_diagnostic_recorded',False) for c in sample['control'])
fb=[]
for q in shadow:
 ss=q['shadow'];r=q['candidates'];f=ss['F']['all_F_scores'][:7];p=[c['score'] for c in r[:7]]
 def rank(v):
  v=np.asarray(v);return np.array([(np.sum(v<x)+.5*(np.sum(v==x)-1)) for x in v])
 rf,rp=rank(f),rank(p);rho=float(np.corrcoef(rf,rp)[0,1]) if rf.std()>0 and rp.std()>0 else None
 top4=ss['F_rank7'][:4];simple4=[0,1,4,5]
 fb.append(dict(label=q['label'],time=q['time'],F_top4_best_physical_score=max(p[i] for i in top4),same_budget_fixed4_best=max(p[i] for i in simple4),original7_best=max(p),replacement7_best=max(r[i]['score'] for i in ss['replacement_pool']),F_physics_rank_correlation=rho,F_top4_retained_best=ss['F_top4_retains_physical_top7'],nominee_wins=ss['nominee_wins'],replacement_gain=ss['replacement_gain'],duplicate_exclusions=ss['F']['duplicate_exclusions'],min_candidate_F_cost=ss['F']['seconds']))
checks=[]
for r in rec:
 eps=[]
 for arm in r['arms']:
  with gzip.open(D/(arm+'_'+r['case']+'_'+str(r['time'])+'.json.gz'),'rt') as f:eps.append(json.load(f))
 pp=[[x for x in e['physics'] if x['force_time']<r['time']-1e-7] for e in eps];cc=[[x for x in e['control'] if x['time']<r['time']-1e-7] for e in eps]
 assert all(x==pp[0] for x in pp) and all(x==cc[0] for x in cc)
 anchors=[]
 for ep in eps:anchors.append(next(x['anchor'] for x in ep['physics'] if x['active']))
 checks.append(dict(case=r['case'],time=r['time'],actual_physics_prefix_exact=True,BFM_control_prefix_exact=True,physics_prefix_ticks=len(pp[0]),control_prefix_steps=len(cc[0]),onset_anchor_differences_cm=[(100*(np.array(a)-anchors[0])).tolist() for a in anchors],note='precontact handoff preserves onset_actual rule, so subsequent one-time anchor alignment differs slightly; postonset handoffs share exact anchors',full_actual_branches=[next(x for x in main if x['label']==ep['label']) for ep in eps]))
E=S/'runs/bc-eval-01';selection=read(E/'selection.json') if (E/'selection.json').exists() else None
learning=read(S/'runs/bc-fit-01/training.json') if (S/'runs/bc-fit-01/training.json').exists() else None
model_shas={}
for folder in [S/'runs/bc-fit-01',S/'runs/student-ppo5301-01',S/'runs/student-ppo5302-01']:
 if folder.exists():
  for p in folder.glob('*.pt'):model_shas[str(p.relative_to(S))]=sha(p)
ledger=read(S/'ledger.json');assert all(j['status']!='running' for j in ledger['jobs'])
result=dict(status='completed_saved_artifact_analysis',scope='single pilot; not general contact Student; no new Expert PPO',counts=read(D/'counts.json'),Teacher_gate=gate,recovery=rec,prefix_checks=checks,full_actual_summaries=main,query_cost=summary([q['wall_s'] for q in basequery]),shadow_query_cost=summary([q['wall_s'] for q in shadow]),candidate_cost=summary([c['wall_s'] for c in seq]),state_save_cost=summary([q['snapshot_s'] for q in queries]),state_restore_cost=summary([v for q in queries for v in q['restore_s']]),candidate_nomination_cost=summary([q['nomination_s'] for q in queries]),F_cost=summary([q['shadow']['F']['seconds'] for q in shadow]),candidate_BFM_controls=sum(c['BFM_calls'] for c in seq),candidate_actor_calls=sum(c['BFM_calls'] for c in seq)*(2 if nominal else 1),standing_diagnostic_extra_call_enabled=nominal,historical_BFM_calls_field_means_control_steps=True,FB=dict(comparisons=fb,wins=sum(x['nominee_wins'] for x in fb),n=len(fb),top4_best_retention=sum(x['F_top4_retained_best'] for x in fb),stage='shadow only; formal Teacher unchanged; no-go if0wins',same4_fixed_reference=[0,1,4,5]),BC_training=learning,BC_selection=selection,BC_physics=read(E/'summaries.json') if (E/'summaries.json').exists() else None,Student_PPO=[read(f/'result.json') for f in [S/'runs/student-ppo5301-01',S/'runs/student-ppo5302-01'] if (f/'result.json').exists()],model_shas=model_shas,resources=dict(job_seconds=sum(j.get('elapsed_s',0) for j in ledger['jobs'])+ledger.get('auxiliary_CPU_analysis_reserve_s',0),jobs=ledger['jobs'],output_bytes=sum(p.stat().st_size for p in S.rglob('*') if p.is_file())),reward_audit=read(S/'runs/reward-readonly-02/result.json'))
(out/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(queries=len(queries),FBwins=result['FB']['wins'],Teacher_gate=gate,job_seconds=result['resources']['job_seconds']),indent=2))
