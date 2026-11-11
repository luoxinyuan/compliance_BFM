"""Assemble small public F1 evidence from immutable private outputs (no simulations)."""
import argparse,collections,hashlib,json,math,statistics,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--l2',required=True);p.add_argument('--publication',type=Path,required=True);a=p.parse_args();S=a.stage;R=a.publication/'research_notes'
def read(p):return json.loads(p.read_text())
def write(n,x): (R/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def f(x,n=3):return '—' if x is None else f'{x:.{n}f}'
L1=read(S/'runs/l1-merged/candidate-scores.json');L2=read(S/'runs'/a.l2/'candidate-scores.json');Q1=read(S/'runs/l1-merged/query-ledger.json');Q2=read(S/'runs'/a.l2/'query-ledger.json');summary=read(S/'reports/f1_results_summary.json');phase=read(S/'reports/f1_phase_metrics.json');align=read(S/'reports/f1_query_alignment.json')
assert read(S/'runs'/a.l2/'complete.json')['status']=='completed'
for name in ['F1_L1_RESULTS.md','f1_l1_summary.json','f1_results_summary.json','f1_phase_metrics.json','f1_query_alignment.json']:(R/name).write_bytes((S/'reports'/name).read_bytes())
public=[]
for run,rows in [('l1-merged',L1),(a.l2,L2)]:
 for r in rows:
  s=r['score'];public.append({**{k:r[k] for k in ['layer','state_id','parent_id','actual_seconds','boundary','metrics','artifact','artifact_sha256','artifact_bytes']},'candidate':r.get('candidate'),'replan':r.get('replan'),'run':r.get('source_run',run),'runtime':r.get('actual_runtime_sha',read(S/'runs'/a.l2/'launch.json')['runtime']),'proposal':{k:v for k,v in r.get('proposal',{}).items() if k not in ['z','zs']},'score':{k:v for k,v in s.items() if k!='risk_events'},'slip_semantics':r['slip_semantics'],'raw_assets':'private; artifact SHA binds full z/control/physics, not included in public Git'})
for row in public:
 row['information_tier']='O-F' if row['layer']=='L1' else 'C-S' if row['layer']=='L2-C-S' else 'C-P'
 row['query_id']=row['state_id']+('-L1' if row['layer']=='L1' else '-CS' if row['layer']=='L2-C-S' else '-CP-q'+str(row['replan']) if row['layer']=='L2-prediction' else '-CP-executed-window')
 row['failure_reason']='hard boundary' if row['boundary'] is not None else None if row['metrics']['low_slip_task_success'] else 'one or more task/progress/slip/yaw window references not met; see metrics'
# Public per-window aggregate shards; complete raw trajectories stay in stage.
shards=[]
for i in range(0,len(public),300):
    name=f'f1_candidate_scores_part_{i//300:02d}.json'
    write(name,public[i:i+300]);shards.append({'file':name,'rows':len(public[i:i+300])})
write('f1_candidate_scores.json',{'format':'aggregate_shards_v1','rows':len(public),'shards':shards})
queries=[]
for q in Q1+Q2:
  z={k:v for k,v in q.items() if k!='initial_previous_z'};z['state_transform']='continuation(spec)' if q['state_id'].endswith('-continue') else 'identity';z['current_controller_identity']='fixed BC500' if q['information_tier']=='C-S' else 'O-F finite proposal choice' if q['information_tier']=='O-F' else 'causal replanning runtime '+q['actual_runtime_sha'];z['raw_Task_backup_contains_future_spec']=q['information_tier']!='C-S';z['future_access_field_scope']='candidate rollout/scoring interface; raw restoration snapshot is privileged';z['cumulative_job_seconds_scope']='legacy field measures Search lifetime, excludes cold setup; complete job wall is in f1_cost_ledger';z['query_wall_seconds_scope']='candidate query+restore+IO' if q['information_tier']!='C-S' else 'zero means no search, not zero policy execution cost';z['query_id']=q['state_id']+('-L1' if q['information_tier']=='O-F' else '-CS' if q['information_tier']=='C-S' else '-CP-q'+str(q['replan']));z['result_quality']='see linked actual/candidate physical summary, not inferred from reward';z['failure_reason']='all candidates unsafe' if q.get('all_candidates_unsafe') else None;queries.append(z)
for filename in ['f1_query_ledger.jsonl','f1_info_access_audit.jsonl']:
 (R/filename).write_text(''.join(json.dumps(q,ensure_ascii=False,allow_nan=False)+'\n' for q in queries))
jobs=[]
for path in sorted((S/'jobs').glob('*.json')):
 j=read(path)
 if 'elapsed_s' in j:jobs.append(dict(name=path.stem,layer=j.get('layer','reserve'),status=j.get('status'),elapsed_s=j['elapsed_s'],runtime=j.get('runtime'),physics_steps=j.get('physics_steps'),rows=j.get('rows')))
by=collections.defaultdict(float)
for j in jobs:by[j['layer']]+=j['elapsed_s']
old=3346.9409750220075;reserve_estimate=1200.;total=old+reserve_estimate+sum(j['elapsed_s'] for j in jobs)
assert by['L1']<=10800 and by['L2']<=14400 and total<=32400
reserve_used=old-1800+reserve_estimate+by['reserve'];assert reserve_used<=5400
counts=read(S/'runs'/a.l2/'counts.json')
cost=dict(prior_L0_and_authorized_reserve_s=old,prior_L0_cap_s=1800,prior_reserve_s=old-1800,new_jobs=jobs,new_measured_by_layer_s=dict(by),new_preparation_reporting_reserve_estimate_s=reserve_estimate,reserve_used_including_estimate_s=reserve_used,total_charged_s=total,total_charged_h=total/3600,non_L3_limit_s=32400,L3_charged_s=0,L3_locked_s=10800,L1_remaining_s=10800-by['L1'],L2_remaining_s=14400-by['L2'],reserve_remaining_s=5400-reserve_used,L1_candidate_branches=len(L1),L1_candidate_physics_ticks=sum(round(r['actual_seconds']*200) for r in L1),L2_counts=counts,new_RL_decisions=0,optimizer_updates=0,scope='Nested subprocess/evaluation costs not double counted. Preparation is a declared estimate, not GPU kernel timing. Prefix reconstruction/backend qualification and cache probes are in job wall; extra physics counts listed separately where archived.')
write('f1_cost_ledger.json',cost)
native=read(S/'prep/native-source-identities.json')
for item in native.values():
 if isinstance(item,dict) and isinstance(item.get('path'),str):item['path']=item['path'].split('/projects/compliance_BFM/',1)[-1]
native['old_high_actor_loaded_scope']='native proposal generation only; F1 separately loads frozen BC500/B64/B128/v3 comparators'
identities=read(S/'runs/l1-03/model-identities.json');write('f1_model_identities.json',dict(models=identities,BFM_SHA256='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361',BC500_original_SHA256='a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e',native_sources=native))
table=[];pairs=summary['L2']['pairs']
for p in pairs:
 cs,cp=p['CS'],p['CP'];table.append('|'+p['state']+'|'+f(cs['error_cm'])+'/'+f(cs['force_N'])+'|'+f(cp['error_cm'])+'/'+f(cp['force_N'])+'|'+f(cs['slip_mps'])+'→'+f(cp['slip_mps'])+'|'+f(cs['yaw_deg'],2)+'→'+f(cp['yaw_deg'],2)+'|'+str(cs['good'])+'→'+str(cp['good'])+'|')
reason_rows=[]
for p in pairs:
 for label in ['CS','CP']:
  x=p[label];r=next(q for q in L2 if q['layer']=='L2-C-'+('S' if label=='CS' else 'P') and q['state_id']==p['state']);m=r['metrics'];family=next(q['task_family'] for q in phase if q['state_id']==p['state'] and q['layer']==r['layer']);err=4 if family=='near' else 8;force=4 if family=='near' else 7
  failures=[]
  if m['error_cm'] is None or m['error_cm']>err:failures.append('error')
  if m['force_rms']>force:failures.append('force')
  if m['loaded_slip_peak']>.45:failures.append('slip')
  if m['yaw_abs_deg']>20:failures.append('yaw')
  if not m['progress_pass']:failures.append('progress')
  if family=='stop' and m['holding_error_cm'] is not None and (m['holding_error_cm']>8 or m['holding_force_N']>6.5):failures.append('holding')
  if r['boundary'] is not None:failures.append('physical_boundary')
  reason_rows.append(dict(state=p['state'],controller=label,task_family=family,failures=failures,progress_shortfall_mm=max(0.,1000*(m['partner_dx_m']-m['wrist_dx_m']-err/100)),window_good=m['low_slip_task_success']))
write('f1_window_failure_coordinates.json',reason_rows)
times=summary['L2']['C_P_query_wall_s'];t50=statistics.median(times);t90=sorted(times)[math.ceil(.9*len(times))-1]
checks=summary['L2']['checks'];prefix=[c for c in checks if c.get('test')=='real C-P stop/continue prefix']
goodcs=sum(p['CS']['good'] for p in pairs);goodcp=sum(p['CP']['good'] for p in pairs)
gate=f'''# F1 L2 正式报告点：已停，L3未授权

L1/L2执行完成；本报告不是训练成功、封存晋级或部署资格。BC500基线不变。L3没有启动或排队，3h完整保留。方法与权限详见[F1_L1_L2_METHODS](F1_L1_L2_METHODS.md)，历史与本轮修复见[F1_ERRATUM](F1_ERRATUM.md)。

## 覆盖与实际结果

L1：12父组24状态，384条连续3s候选；11状态/7父组至少存在一个同时满足任务、推进、滑移与yaw参考的候选，v3评分实际选中合格8/24。此为有限存在性证据，不是一个共享网络的成功率。

L2：六父组、七真实起始状态，加一个stop/continue相关分支，共八个3s窗口；C-P每100ms重规划，共{len(times)}次。固定BC500合格{goodcs}/8，C-P合格{goodcp}/8；两个分支或同父组多个状态不能当独立重复。全部是已暴露条件。

|状态|C-S误差cm/力N|C-P误差cm/力N|slip峰m/s C-S→C-P|末端abs yaw° C-S→C-P|完整窗口合格 C-S→C-P|
|---|---|---|---|---|---|
'''+ '\n'.join(table)+f'''

每项XYZ、cap、root/脚步、q_PD/真实q、支撑/力矩及分段数据见[f1_phase_metrics](f1_phase_metrics.json)；释放后anchor误差不计接触任务。完整窗口参考不等于完整回合恢复。逐坐标未过原因和进度差见[f1_window_failure_coordinates](f1_window_failure_coordinates.json)，不把边界小差异与严重退化混为一类。旧相对门的完整回合族统计不由这八个短窗替代；窗口相对风险坐标另列，不宣称旧门晋级。

## 因果与执行合同

每个窗口首次查询未来spec干预测试，共{sum(c.get('test')=='future-spec invariance' for c in checks)}项逐值通过。stop/continue共同前缀核验：`{prefix}`。C-S共同前缀核验：`{align['restricted_stop_continue_prefix']}`。同一实际首命令及首BFM子步对齐{len(align['causal_command_alignment'])}次，原BC500 L1/L2独立重复全量一致={align['all_BC500_repetitions_exact']}，见[query alignment](f1_query_alignment.json)。

C-P只用当前伙伴位置/速度/active/yaw的常速预测，不读取实际未来。C-S只有原1639D ideal-force BC500固定对照，**C-S planner缺格**。饱和力隐藏状态别名未构造，不做信息论不可辨识结论。原始恢复快照仍带实际Task spec，future-access字段只指候选/评分接口，不宣称OS级隔离。

L1的16候选/3s与L2的8候选/2s不同；因此当前证据**不能证明必须依赖未来oracle**。具体有限失败也不能证明所有合法latent不可达。

## 延迟与资源

重规划p50 {t50:.3f}s、p90 {t90:.3f}s，对应实际0.1s首命令；这是离线反馈测量，不是实时算法。L1实测{by['L1']/3600:.4f}h，L2实测{by['L2']/3600:.4f}h；含L0、失败、CPU、IO、准备估计累计{total/3600:.4f}h，未授权L3支出0。账本见[f1_cost_ledger](f1_cost_ledger.json)，准备估计与实测分别列出。

预测分支与真实C-P执行分开记账，所有模型冻结；RL决策/训练更新均0。runtime与checkpoint见[f1_model_identities](f1_model_identities.json)及逐query身份，发布HEAD不冒充运行SHA。

## 后续确认点

L3的单一候选假设、反证与预计成本在[F1_DECISION_TABLE](F1_DECISION_TABLE.md)列明。**本次到此停止；未获Yi明确确认不得运行L3或占用其3h。**
'''
(R/'F1_L2_GATE_REPORT.md').write_text(gate)
slips=[]
for name in ['l1-slip-kinematics-final','l2-slip-kinematics-final']:
 slips.extend(read(S/'runs'/(name+'.json')))
write('f1_slip_kinematics.json',slips)
print(json.dumps(dict(charged_h=total/3600,L1_h=by['L1']/3600,L2_h=by['L2']/3600,CP_good=goodcp,CS_good=goodcs,p50=t50,p90=t90,queries=len(times)),indent=2))
