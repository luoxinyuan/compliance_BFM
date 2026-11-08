"""Read-only Issue53 cost/ranking reanalysis; zero new physical execution."""
import json,gzip,time,collections,math
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/interactive-contact-pilot-20261108');S=Path(__file__).resolve().parents[2]
def rd(p):
 with gzip.open(p,'rt')if str(p).endswith('.gz')else p.open()as f:return json.load(f)
def stats(x):return dict(n=len(x),sum=float(sum(x)),mean=float(np.mean(x))if x else None,p50=float(np.median(x))if x else None,p90=float(np.quantile(x,.9))if x else None)
def rank(x):x=np.asarray(x);return np.array([np.sum(x<v)+.5*(np.sum(x==v)-1)for v in x])
def agg(qs):return dict(queries=len(qs),wall_s=stats([q['wall_s']for q in qs]),candidate_high_level_decisions=sum(c['steps']for q in qs for c in q['candidates']),candidate_physics=sum(c['BFM_calls']*4 for q in qs for c in q['candidates']),candidate_count=sum(len(q['candidates'])for q in qs))
def main():
 start=time.monotonic();mainq=rd(P/'runs/diagnose-01/searches.json');repair=rd(P/'runs/repair-short-01/searches.json');moving=rd(P/'runs/moving-cost-01/searches.json');allq=mainq+repair+moving
 parts={'throughput_qualification':[q for q in mainq if q['label'].startswith('cost_')],'FB_shadow':[q for q in mainq if 'shadow'in q],'original_iterative_takeover':[q for q in mainq if q['label'].startswith('Teacher_replan')],'corrected_postcontact_takeover':repair,'actual_moving_cost_check':moving};assert sum(map(len,parts.values()))==96
 eps={k:rd(P/f'runs/diagnose-01/S4000_{k}.json.gz')for k in ['short08','normal72','mid24','stop66']};fb=[]
 for q in parts['FB_shadow']:
  name=q['label'].removeprefix('shadow_').rsplit('_',1)[0];ep=eps[name];near=min(ep['physics'],key=lambda x:abs(x['time']-q['time']));active=bool(near['active']);force=float(np.linalg.norm(near['force']));fs=q['shadow']['F']['all_F_scores'][:7];ps=[c['score']for c in q['candidates'][:7]];a,b=rank(fs),rank(ps);rho=float(np.corrcoef(a,b)[0,1])if a.std()and b.std()else None;pairs=[]
  for i in range(7):
   for j in range(i):
    df,dp=fs[i]-fs[j],ps[i]-ps[j]
    if abs(df)>1e-12 and abs(dp)>1e-12:pairs.append(float(df*dp>0))
  topF=int(np.argmax(fs));topP=int(np.argmax(ps));fb.append(dict(label=q['label'],time=q['time'],source_physics_time=near['time'],currently_active=active,actual_current_force_N=force,group='actual_contact'if active else'precontact_no_actual_force',spearman=rho,pairwise_order_agreement=float(np.mean(pairs))if pairs else None,F_top1_matches=topF==topP,F_top4_retains=q['shadow']['F_top4_retains_physical_top7'],physical_regret_of_Ftop1=max(ps)-ps[topF],FB_gain=q['shadow']['replacement_gain'],FB_wins=q['shadow']['nominee_wins']))
 groups={}
 for k in ['actual_contact','precontact_no_actual_force']:
  v=[r for r in fb if r['group']==k];groups[k]=dict(n=len(v),spearman_mean=float(np.mean([r['spearman']for r in v if r['spearman']is not None])),spearman_median=float(np.median([r['spearman']for r in v if r['spearman']is not None])),pairwise_order_agreement_mean=float(np.mean([r['pairwise_order_agreement']for r in v])),top1_matches=sum(r['F_top1_matches']for r in v),top4_retains=sum(r['F_top4_retains']for r in v),FB_wins=sum(r['FB_wins']for r in v),physical_regret_of_Ftop1=stats([r['physical_regret_of_Ftop1']for r in v]))
 fam={}
 for k in ['short_local','forward_motion_stop','mid_coordination']:
  qq=[q for q in allq if (('short08'in q['label']or'correct_1p4'in q['label'])if k=='short_local'else('normal72'in q['label']or'stop66'in q['label']or q in moving)if k=='forward_motion_stop'else'mid24'in q['label'])];fam[k]=agg(qq)
 # De-overlapped valid-label origins: one early first query belongs to shadow, remaining19 takeover; corrected14.
 valid_sources=dict(normal72=dict(valid_unique_labels=20,full_queries_for_window=20,first_query_reused_from_shadow=True,incremental_full_queries_after_shadow=19,qualified_seconds=2.),short08=dict(valid_unique_labels=14,full_queries_for_window=14,incremental_full_queries_after_shadow=14,qualified_seconds=1.4))
 for k,qs in [('normal72',[q for q in mainq if q['label']=='Teacher_replan_2s_normal72_4.7']+[q for q in mainq if q['label']=='shadow_normal72_4.7']),('short08',repair)]:valid_sources[k]['search_wall_s']=sum(q['wall_s']for q in qs);valid_sources[k]['wall_s_per_valid_label']=sum(q['wall_s']for q in qs)/len(qs)
 training=rd(P/'runs/bc-fit-01/training.json');output=dict(partition_nonoverlapping={k:agg(v)for k,v in parts.items()},total=agg(allq),role_overlap='3 shadow first-action queries reused for takeover initial labels; do not count them again.71 additional iterative takeover queries includes38 affected original post-onset;40 affected labels also include2 reused initial shadow states.34 valid labels=1shadow+19early+14corrected. All96 physically executed counted.',valid_label_origin=valid_sources,FB_by_current_contact=groups,FB_rows=fb,family_costs=fam,training=dict(original_old_only_and_mixed=training,notes='Joint BC updates shared across task families; actual sample label fractions/source draws needed for marginal allocation. Do not assign entire1000 updates to each family. Label lineage and cost estimates are descriptive; shared setup/eval not additive.'),wall_s=time.monotonic()-start,new_physics=0,new_queries=0)
 out=S/'runs/history-cost-01';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False));print(json.dumps({k:output[k]for k in ['partition_nonoverlapping','FB_by_current_contact','family_costs','valid_label_origin']},indent=2))
if __name__=='__main__':main()
