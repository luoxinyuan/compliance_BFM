"""Read-only stage reduction; writes only small reports, no physics or selection rerun."""
import json,math,time
from pathlib import Path
S=None;R=None
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def fmt(x):return 'NA' if x is None else f'{x:.3f}'
def compact(r):
 m=r['metrics'];return dict(state=r['state_id'],parent=r['parent_id'],candidate=r.get('candidate'),error_cm=m['error_cm'],force_N=m['force_rms'],slip_mps=m['loaded_slip_peak'],yaw_deg=m['yaw_abs_deg'],progress=m['progress_pass'],partner_dx=m['partner_dx_m'],wrist_dx=m['wrist_dx_m'],good=m['low_slip_task_success'],seconds=r['actual_seconds'],boundary=r['boundary'],old_return=r['score']['old_discounted_return'],v3_return=r['score']['v3_discounted_return'],artifact=r['artifact'])
def run(l1,l2=None):
 a=read(S/'runs'/l1/'candidate-scores.json');q=read(S/'runs'/l1/'query-ledger.json');states=list(dict.fromkeys(r['state_id'] for r in a));by={s:[r for r in a if r['state_id']==s] for s in states}
 chosen=[];oracle_good=[];pricing=[];table=[]
 for s,rr in by.items():
  qi=next(x for x in q if x['state_id']==s);best=rr[qi['selected_candidate_id']];good=[r for r in rr if r['metrics']['low_slip_task_success']]
  clean=max(good,key=lambda r:r['score']['v3_discounted_return']) if good else None
  chosen.append(compact(best));oracle_good.extend(compact(r) for r in good)
  # Pair count within state only; correlated branches are not independent parents.
  bad=[r for r in rr if not r['metrics']['low_slip_task_success'] and r['actual_seconds']>=2.999 and r['boundary'] is None]
  comparisons=[(x,y) for x in good for y in bad]
  pricing.append(dict(state=s,parent=rr[0]['parent_id'],good_candidates=len(good),bad_candidates=len(bad),comparisons=len(comparisons),old_prefers_bad=sum(y['score']['old_discounted_return']>x['score']['old_discounted_return'] for x,y in comparisons),v3_prefers_bad=sum(y['score']['v3_discounted_return']>x['score']['v3_discounted_return'] for x,y in comparisons),v3_selected_good=bool(best['metrics']['low_slip_task_success'])))
  z=compact(best);table.append(f"|{s}|{len(good)}/{len(rr)}|{z['candidate']}|{fmt(z['error_cm'])}/{fmt(z['force_N'])}|{fmt(z['slip_mps'])}/{fmt(z['yaw_deg'])}|{fmt(z['seconds'])}|{fmt(clean['metrics']['error_cm'])+'/'+fmt(clean['metrics']['force_rms']) if clean else '未找到'}|")
 result=dict(L1_rows=len(a),L1_states=len(states),L1_parents=len(set(r['parent_id'] for r in a)),states_with_good=len(set(r['state'] for r in oracle_good)),parents_with_good=len(set(r['parent'] for r in oracle_good)),selected_good=sum(r['good'] for r in chosen),selected=chosen,good=oracle_good,pricing=pricing)
 write(R/'f1_l1_summary.json',result)
 (R/'F1_L1_RESULTS.md').write_text('# F1 L1：连续控制有限能力诊断\n\n全为已暴露条件、O-F真实未来评分，不是可部署/封存结果，不训练或反选历史模型。\n\n'+f"{len(a)}分支、{len(states)}状态、{result['L1_parents']}父组；至少一个3s合格候选的状态{result['states_with_good']}，父组{result['parents_with_good']}。v3评分实际选择合格{result['selected_good']}/{len(states)}。\n\n"+'|状态|合格候选|v3选择|误差cm/力N|slip m/s / abs yaw°|时长s|合格集最高v3者误差/力|\n|---|---|---|---|---|---|---|\n'+'\n'.join(table)+'\n\n判据未变：误差/力、.45m/s加载事件、20°绝对yaw、实际推进和完整3s同时满足；提前时间截断不是足额3s失败证据。原生motion/插值/冻结反馈策略只是有限来源，未搜索完整256D空间。分项及同状态旧/v3排序见 `f1_l1_summary.json`；相关候选不作独立样本。\n')
 if l2:
  b=read(S/'runs'/l2/'candidate-scores.json');qq=read(S/'runs'/l2/'query-ledger.json');actual=[r for r in b if r['layer'] in ('L2-C-S','L2-C-P')];pairs=[]
  for state in dict.fromkeys(r['state_id'] for r in actual):
   sr=[r for r in actual if r['state_id']==state];cs=next(r for r in sr if r['layer']=='L2-C-S');cp=next(r for r in sr if r['layer']=='L2-C-P');pairs.append(dict(state=state,parent=cs['parent_id'],CS=compact(cs),CP=compact(cp)))
  result['L2']=dict(pairs=pairs,checks=read(S/'runs'/l2/'checks.json'),counts=read(S/'runs'/l2/'counts.json'),C_P_queries=sum(x['information_tier']=='C-P' for x in qq),C_P_query_wall_s=[x['query_wall_seconds'] for x in qq if x['information_tier']=='C-P'])
  write(R/'f1_results_summary.json',result)
 return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--l1',required=True);p.add_argument('--l2');a=p.parse_args();S=a.stage;R=S/'reports';R.mkdir(exist_ok=True);run(a.l1,a.l2)
