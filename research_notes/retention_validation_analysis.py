"""Saved-result evaluation only; applies locked thresholds without model selection."""
from pathlib import Path
import json,gzip,time,hashlib
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/retention-newconditions-20261109';R=S/'repo';PLAN=json.loads((R/'research_notes/retention_validation_protocol.json').read_text());OUT=S/'runs/analysis';OUT.mkdir(exist_ok=False);beg=time.monotonic()
def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:return json.load(f)
rows={role:read(S/'runs'/role/'summaries.json') for role in PLAN['models']};init={role:read(S/'runs'/role/'initials.json') for role in PLAN['models']};assert all(len(v)==13 for v in rows.values());lookup={k:{r['case']:r for r in v} for k,v in rows.items()};cases=[];prefix=[]
def severity(c,b,case):
 bad=[];a=c['all'];v=b['all']
 if c['boundary'] or not c.get('complete',True):bad.append('physical_boundary_or_incomplete')
 if b['boundary'] or not b.get('complete',True):bad.append('baseline_physical_boundary_or_incomplete')
 if a['error_cm'] is not None and a['error_cm']>v['error_cm']+max(5,.5*v['error_cm']):bad.append('error_collapse')
 if a['force_rms']>v['force_rms']+2:bad.append('force_burden')
 if case['family'] in ['local','none'] and c['root_max_displacement']>b['root_max_displacement']+.10:bad.append('unrequested_root_excursion')
 peak=lambda r:max(q['slip_peak_mps'] for q in r['quality'].values())
 if peak(c)>max(peak(b)+.15,2*peak(b)):bad.append('relative_slip')
 def yaw(r):
  qq=list(r['quality'].values());initial=qq[0]['yaw_deg_initial'];return max(abs(q['yaw_deg_final']-initial) for q in qq)
 if yaw(c)>yaw(b)+15:bad.append('relative_yaw')
 if case['family']=='none' and a['force_rms']!=0:bad.append('unexpected_force')
 return bad
for item in PLAN['conditions']:
 key=item['name'];b=lookup['BC500'][key];c=lookup['s5403b7'][key]
 assert init['BC500'][key]==init['s5403b7'][key],('initial_pair_mismatch',key)
 bad=severity(c,b,item);a=c['all'];bb=b['all'];improvement=None;noninferior=None;costbad=[];cost=[]
 if item['family']=='long':
  improvement=a['error_cm']<=.95*bb['error_cm'] and a['force_rms']<=.95*bb['force_rms']
  q=c['quality'].get('sustained',{});lim=PLAN['promotion']['significant_cost_amplification_limits'];ref=PLAN['promotion']['dev_cost_reference']
  for k,rkey,lkey in [('slip_peak_mps','sustained_slip_peak_mps','sustained_slip_peak_mps'),('slip_p95_mps','sustained_slip_p95_mps','sustained_slip_p95_mps')]:
   x=q.get(k);assert x is not None
   if x>lim[lkey]:costbad.append(k)
   cost.append(dict(metric=k,value=x,dev_reference=ref[rkey],difference=x-ref[rkey],limit=lim[lkey],classification='amplified' if x>lim[lkey] else ('reduced' if x<ref[rkey] else 'equal')))
  for ph in ['holding','released']:
   x=abs(c['quality'][ph]['yaw_deg_final']);limit=lim['holding_or_released_abs_final_yaw_deg'];reference=ref['released_abs_yaw_deg']
   if x>limit:costbad.append(ph+'_yaw')
   cost.append(dict(metric=ph+'_abs_final_yaw',value=x,dev_reference=reference,difference=x-reference,limit=limit,classification='amplified' if x>limit else ('reduced' if x<reference else 'equal')))
 elif item['family'] in ['medium','local','stop']:
  noninferior=a['error_cm']<=bb['error_cm']+.5 and a['force_rms']<=bb['force_rms']+.25
 cases.append(dict(case=key,family=item['family'],initial_pair_exact=True,initial_sha=init['BC500'][key]['audit']['sha256'],BC=bb,candidate=a,severe_failures=bad,long_actual_benefit=improvement,side_noninferior=noninferior,cost_failures=costbad,cost_comparison=cost,passed=not bad and improvement is not False and noninferior is not False and not costbad))
# Whole actual same-prefix trajectory comparison; policy receives no future stop answer.
for item in PLAN['conditions']:
 if not item['common_prefix_of']:continue
 for role in PLAN['models']:
  a=read(S/'runs'/role/(role+'_'+item['common_prefix_of']+'.json.gz'));b=read(S/'runs'/role/(role+'_'+item['name']+'.json.gz'));end=item['args']['stop'];r={}
  for stream,clock,keys in [('physics','force_time',['qpos','qvel','force','wrist','root','support','foot_normal_force_n','feet_xyz','anchor']),('control','time',['q_PD','bfm_obs','executed_z','action'])]:
   aa=[x for x in a[stream] if x[clock]<end-1e-8];bb=[x for x in b[stream] if x[clock]<end-1e-8];assert len(aa)==len(bb)
   use=[k for k in keys if aa and k in aa[0]];assert ('qpos'in use) if stream=='physics' else ('q_PD'in use)
   equal=all(all(x[k]==y[k] for k in use) for x,y in zip(aa,bb));assert equal,(role,item['name'],stream)
   r[stream]=dict(rows=len(aa),keys=use,exact=equal)
  prefix.append(dict(role=role,continued=item['common_prefix_of'],stopped=item['name'],before=end,comparisons=r))
passed=all(x['passed'] for x in cases);result=dict(protocol=PLAN,runtime=json.loads((S/'runs/BC500/launch.json').read_text())['runtime'],rows=rows,conditions=cases,common_prefix=prefix,initial_states={role:{k:dict(audit_sha=v['audit']['sha256'],restricted_sha=v['restricted_sha'],source=v['audit']['source'],time=v['audit']['time'],root=v['root'],wrist=v['wrist'],source_physics_steps=v['source_physics_steps']) for k,v in values.items()} for role,values in init.items()},conclusion='new-condition validation passed' if passed else 'benefit limited to development conditions',promotion_passed=passed,no_model_reselection=True,no_training=True,Teacher_queries=0,analysis_seconds=time.monotonic()-beg,failed_conditions=[x['case'] for x in cases if not x['passed']])
(OUT/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False))
head=['# RETENTION：13条新条件冻结配对评价','', '条件和阈值在运行前锁定；两个模型用相同初态、随机流和真实源前缀。无训练、Teacher查询或原控制/物理代码修改。这里只检验未用于5403选模的有限新组合，不声称全任务generalization。','', '## 逐条件对照','', '|条件|源/接管s/onset s|BC误差cm/力N|5403b7误差cm/力N|严重保护/实际收益或容差/代价|','|---|---|---|---|---|']
for x,it in zip(cases,PLAN['conditions']):
 def metric(a):return ('—' if a['error_cm']is None else f"{a['error_cm']:.3f}")+f"/{a['force_rms']:.3f}"
 flags=x['severe_failures']+x['cost_failures'];
 if x['long_actual_benefit'] is False:flags+=['长距主要收益未成立']
 if x['side_noninferior'] is False:flags+=['容差外退化']
 a=it['args'];head.append(f"|{x['case']}|{a['source']} / {a['until']} / {a['onset']}|{metric(x['BC'])}|{metric(x['candidate'])}|{', '.join(flags) or '通过'}|")
head+=['','## 预声明判读','', '**'+('新条件验证通过' if passed else '收益限于开发条件')+'**。'+('下一决策为第二seed确认，本轮不自动执行。' if passed else '5403b7退回开发结论，BC500保持基线；保留具体条件的收益，不追加条件或训练找补。'),'', '失败条件：'+(', '.join(result['failed_conditions']) or '无')+'。','', '完整连续指标和判据见[聚合证据](retention_validation_artifacts/evidence.json)，长距分阶段见[物理表](RETENTION_VALIDATION_PHASES.md)。']
(OUT/'RETENTION_VALIDATION_RESULTS.md').write_text('\n'.join(head)+'\n')
phase=['# 长距新条件：持续推进、保持与恢复','', 'XYZ为世界坐标，误差只在实际接触时计算。末yaw是pelvis世界朝向；释放后不考核旧anchor跟随。滑移为有载荷足点运动代理。','', '|条件/模型|阶段|XYZ误差RMS cm|有符号XYZ误差cm|XYZ力RMS N|滑移峰/p95 m/s|末yaw°|q_PD20ms/真实q5ms峰°|','|---|---|---|---|---|---|---|---|']
def vec(v):return '—' if v is None else '/'.join(f'{z:.3f}'for z in v)
for it in PLAN['conditions']:
 if it['family']!='long':continue
 for role in PLAN['models']:
  for ph in ['startup','sustained','holding','released']:
   q=lookup[role][it['name']]['quality'][ph];phase.append(f"|{it['name']}/{role}|{ph}|{vec(q['error_xyz_rms_cm'])}|{vec(q['error_xyz_signed_cm'])}|{vec(q['force_xyz_rms'])}|{q['slip_peak_mps']:.4f}/{q['slip_p95_mps']:.4f}|{q['yaw_deg_final']:.2f}|{q['qpd_step_peak_deg']:.2f}/{q['actual_joint_5ms_peak_deg']:.2f}|")
phase+=['','## 相对开发参考的代价','', '开发参考：持续滑移峰.665922m/s、p95 .038076m/s、释放末绝对yaw15.149558°。显著放大界限分别.832403m/s、.048076m/s、20.149558°。在容差内的增加仍列真实差值，不称统计等价。','', '|条件|指标|新值|开发参考|差值|判读|','|---|---|---|---|---|---|']
for x in cases:
 for c in x['cost_comparison']:
  phrase='显著放大' if c['classification']=='amplified' else ('缩小' if c['classification']=='reduced' else '持平')
  if c['difference']>0 and c['classification']!='amplified':phrase='增加但在预定容差内'
  phase.append(f"|{x['case']}|{c['metric']}|{c['value']:.4f}|{c['dev_reference']:.4f}|{c['difference']:+.4f}|{phrase}|")
(OUT/'RETENTION_VALIDATION_PHASES.md').write_text('\n'.join(phase)+'\n')
print(json.dumps(dict(conclusion=result['conclusion'],failed=result['failed_conditions'],conditions=[dict(case=x['case'],BC=[x['BC']['error_cm'],x['BC']['force_rms']],candidate=[x['candidate']['error_cm'],x['candidate']['force_rms']],flags=x['severe_failures']+x['cost_failures']) for x in cases]),indent=2))
