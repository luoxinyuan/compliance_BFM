"""Immutable QA2 P2 aggregation. All uncertain pairs remain visible."""
import itertools,json,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'research_notes')]
from research.interaction_fb.qa2_value import metrics
S=R.parent
def main():
    start=time.monotonic();run=S/'runs/P2-prospective-08';data=json.loads((run/'p2-result.json').read_text());assert data['status']=='P2_COMPLETE'
    states=data['states'];assert len(states)==4 and len({s['parent'] for s in states})==4
    base=json.loads((S/'runs/V1-MC/result.json').read_text())['baselines'];records=[];pairs=[]
    for state in states:
        rr=[]
        for pred in state['predictions']:
            bs=[b for b in state['branches'] if b['candidate']==pred['candidate']];assert len(bs)==2 and all(b['complete_50_steps'] for b in bs)
            bs=sorted(bs,key=lambda b:b['crn']);values=[b['return_1s'] for b in bs];g=float(np.mean(values))
            rec=dict(parent=state['parent'],family=state['family'],phase=state['phase'],candidate=pred['candidate'],crn_returns=values,G_mu_mean=g,CRN_return_difference=abs(values[0]-values[1]),magnitude_uncertain=abs(values[0]-values[1])>.25*max(1,abs(g)),predictions=pred['scores'],global_baseline=base['global']['global_mean'],phase_baseline=base['phase']['phase_means'].get(state['phase'],base['phase']['global_mean']),force_rms_mean=float(np.mean([b['force_rms'] for b in bs])),error_rms_mean=float(np.mean([b['error_rms'] or 0 for b in bs])))
            records.append(rec);rr.append(rec)
        for a,b in itertools.combinations(rr,2):
            d=[a['crn_returns'][i]-b['crn_returns'][i] for i in (0,1)];delta=a['G_mu_mean']-b['G_mu_mean'];uncertain=d[0]*d[1]<0
            pairs.append(dict(parent=state['parent'],family=state['family'],a=a['candidate'],b=b['candidate'],delta_G_mean=delta,CRN_delta_G=d,eligible=abs(delta)>=1e-4,ranking_uncertain=uncertain,correct={m:bool(delta*(a['predictions'][m]-b['predictions'][m])>0) for m in data['models']}))
    meta=[dict(parent=r['parent']) for r in records];labels=np.asarray([r['G_mu_mean'] for r in records]);ix=list(range(len(records)))
    baselines={name:metrics(np.asarray([r[name+'_baseline'] for r in records]),labels,meta,ix) for name in ('global','phase')}
    stats={}
    for name in data['models']:
        preds=np.asarray([r['predictions'][name] for r in records]);cal=metrics(preds,labels,meta,ix)
        baseline_errors=[v['high_macro_nmae'] for v in baselines.values() if v['high_macro_nmae'] is not None]
        baseline_best=min(baseline_errors) if baseline_errors else None
        improvement=(baseline_best-cal['high_macro_nmae'])/baseline_best if baseline_best is not None and baseline_best>0 and cal['high_macro_nmae'] is not None else None
        complete_calibration=(cal['high_parents']>=2 and cal['low_parents']>=2)
        uncertain_magnitude=[r for r in records if r['magnitude_uncertain']]
        if not complete_calibration or uncertain_magnitude:g2='UNDERPOWERED'
        else:g2='PASS' if cal['numerical_bounds'] and improvement is not None and improvement>=.20 else 'FAIL'
        eligible=[p for p in pairs if p['eligible']];certain=[p for p in eligible if not p['ranking_uncertain']];uncertain=[p for p in eligible if p['ranking_uncertain']]
        parents={p['parent'] for p in certain};correct_certain=sum(p['correct'][name] for p in certain)
        # Do NOT inflate the final gate by deleting uncertain comparisons.
        lo=correct_certain/len(eligible) if eligible else None
        hi=(correct_certain+len(uncertain))/len(eligible) if eligible else None
        observed=sum(p['correct'][name] for p in eligible)/len(eligible) if eligible else None
        if len(parents)<4:g1='UNDERPOWERED'
        elif lo is not None and lo>=.70:g1='PASS'
        elif hi is not None and hi<.70:g1='FAIL'
        else:g1='UNDERPOWERED'
        per_parent={p:dict(total=sum(q['eligible'] for q in pairs if q['parent']==p),uncertain=sum(q['eligible'] and q['ranking_uncertain'] for q in pairs if q['parent']==p),certain_correct=sum(q['eligible'] and not q['ranking_uncertain'] and q['correct'][name] for q in pairs if q['parent']==p)) for p in {q['parent'] for q in pairs}}
        stats[name]=dict(G1=g1,G2=g2,joint='PASS' if g1==g2=='PASS' else 'NOT_QUALIFIED',comparable_pairs=len(eligible),uncertain_pairs=len(uncertain),certain_pairs=len(certain),certain_correct=correct_certain,effective_parent_count=len(parents),observed_mean_label_accuracy=observed,conservative_accuracy_interval=[lo,hi],parent_counts=per_parent,calibration=cal,baseline_best_high_macro_nmae=baseline_best,baseline_relative_improvement=improvement,uncertain_magnitude_count=len(uncertain_magnitude))
    ans=dict(status='QA2_COMPLETE_NO_ACTOR_AUTHORIZATION',models=stats,independent_parents=4,physical_branches=40,records=records,pairs=pairs,baselines=baselines,A0='NOT_AUTHORIZED: stochastic-mu qualification is not deterministic pi_mean value qualification; no independently qualified pi_mean finite-horizon head/data.',actor_updates=0,U_updates=0,I_updates=0,training_new_online_rows=0,heldout_evaluation_rows=2000,uncertainty_policy='All40 unordered pairs retained. Eligible mean difference1e-4. Sign flips remain UNCERTAIN; final gate uses conservative accuracy interval and4 effective independent parents, not certain-only inflated accuracy.',wall_s=time.monotonic()-start)
    (S/'prep/p2-adjudication.json').write_text(json.dumps(ans,indent=2));print(json.dumps({k:v for k,v in ans.items() if k not in ('records','pairs')},indent=2))
if __name__=='__main__':main()
