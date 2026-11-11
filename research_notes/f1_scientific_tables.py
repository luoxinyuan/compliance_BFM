"""Produce descriptive F1 tables from already reduced records; no new selection."""
import argparse,json,collections,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--publication',type=Path,required=True);a=p.parse_args();R=a.publication/'research_notes'
read=lambda n:json.loads((R/n).read_text())
def f(v,n=3):return '—' if v is None else f'{v:.{n}f}'
s=read('f1_results_summary.json');slip=read('f1_slip_kinematics.json');rows=read('f1_candidate_scores.json');phases=read('f1_phase_metrics.json')
if isinstance(rows,dict): rows=[v for part in rows['shards'] for v in read(part['file'])]
selected={(x['state'],x['candidate']) for x in s['selected']};ss=[x for x in slip if x['layer']=='L1' and (x['state_id'],x['candidate']) in selected];l1=[x for x in slip if x['layer']=='L1'];l2=[x for x in slip if x['layer']!='L1']
def totals(rr):
 v=[x['semantics'] for x in rr]
 return dict(branches=len(v),old_rows=sum(x['old_loaded_overlimit_contact_rows'] for x in v),transient_rows=sum(x['loaded_landing_transient_rows'] for x in v),persist_rows=sum(x['persistent_pre_post_rows'] for x in v),old_ticks=sum(x['old_loaded_overlimit_ticks'] for x in v),persist_ticks=sum(x['persistent_pre_post_ticks'] for x in v),min_normal=min((x['event_normal_N_min'] for x in v if x['event_normal_N_min'] is not None),default=None),max_run=max((max(x['max_consecutive_persistent_s'].values(),default=0.) for x in v),default=0.))
aggregate={k:totals(v) for k,v in [('L1_all',l1),('L1_selected',ss),('L2_CS',[x for x in l2 if x['layer']=='L2-C-S']),('L2_CP',[x for x in l2 if x['layer']=='L2-C-P'])]}
txt='# F1：同候选定价、滑移三分与质量坐标\n\n只做冻结档案复算，不调整reward、成功定义或原风险阈值。候选共享状态、同父组窗口相关；以下不是独立统计检验。\n\n## 同状态奖励排序\n\n'
n=sum(x['comparisons'] for x in s['pricing']);o=sum(x['old_prefers_bad'] for x in s['pricing']);v=sum(x['v3_prefers_bad'] for x in s['pricing'])
txt+=f'L1在有合格候选的11状态内，共{n}对“合格/未合格”相关候选。旧reward偏好未合格{o}/{n}，v3为{v}/{n}。v3在3/11个有解状态仍选中未同时满足窗口参考的候选；其余13状态在本候选池未发现合格解。不能把这个候选对比例当PPO真实采样或优势错误率。\n\n'
txt+='P1-13：BC500连续3s为7.373cm/5.594N、slip .318m/s，v3 return −5.939460；所选motion包络为7.627cm/5.402N、slip .955m/s，旧return −5.434299、风险扣除 .400805 后仍为−5.835105。力和heading累计成本的改善抵消了风险成本；这是当前加权评分与硬质量参考的实际取舍，不证明奖励实现错误或PPO失败仅由reward造成。\n\n另两例是hold-7952的更低力与末端yaw>20°取舍，以及new-near的低误差/力但腕进度参考失败。不能把所有未合格候选称为全面物理劣解。原reward各项及同条件分项见f1_candidate_scores.json。\n\n本轮没有产生新的训练样本/GAE或更新，因此没有新增“没采到/没记住/没保留”的因果比例；现有连续控制证据仅缩小可执行性缺口。\n\n## 滑移三分口径\n\n'
txt+='|集合|分支|旧超限contact行/tick|积分后回落contact行|前后均超限contact行/tick|超限最小法向力N|最长连续前后超限s|\n|---|---:|---|---:|---|---:|---:|\n'
for k,x in aggregate.items():txt+=f"|{k}|{x['branches']}|{x['old_rows']}/{x['old_ticks']}|{x['transient_rows']}|{x['persist_rows']}/{x['persist_ticks']}|{f(x['min_normal'])}|{f(x['max_run'])}|\n"
txt+='\n每个tick可有多个加载接触点，contact行数与tick数不能互换。所有旧风险事件保留；“积分后回落”不是无载擦地，也不保证下一阶段没有滑动。最长连续超限描述仅限0.45阈值，不能推出阈值下真实支撑位移为零。下表同时给材料点路径，路径按每脚每tick最大加载点运动累计，不按接触点数重复加时。\n\n'
txt+='|状态/执行|旧超限tick|持续前后超限tick|瞬态only tick|法向力min/max N|左右材料点路径cm|q_PD最大变化°|真实关节5ms最大变化°|\n|---|---:|---:|---:|---|---|---:|---:|\n'
for x in ss+l2:
 z=x['semantics'];row=next(r for r in rows if r['state_id']==x['state_id'] and r['layer']==x['layer'] and r['candidate']==x['candidate']);m=row['metrics'];paths=x['loaded_material_point_xy_path_m'];tag=f"L1-{x['candidate']}" if x['layer']=='L1' else x['layer']
 txt+=f"|{x['state_id']} / {tag}|{z['old_loaded_overlimit_ticks']}|{z['persistent_pre_post_ticks']}|{z['transient_only_ticks']}|{f(z['event_normal_N_min'])}/{f(z['event_normal_N_max'])}|"+'/'.join(f(paths[k]*100) for k in sorted(paths))+f"|{f(m['qpd_max_step_deg'],2)}|{f(m['q_actual_max_step_deg'],2)}|\n"
txt+='\n任务/slip/yaw窗口参考通过，不等于q_PD连续性完全良好。例如P1-13的合格BC500分支仍有约52.49°目标跳变；这不是实际关节在20ms内转过52.49°。实际q、dq、力矩、关节余量与支撑见机器数据。\n\n## 原相对门与窗口坐标分开\n\n原FRL-v2正式门为族中位误差不恶化>max(2cm,20%初始)、力不增>.75N、loaded slip峰不增>max(.20m/s,50%初始)、release yaw不恶化>10°。本次不是完整开发族/完整释放回合，不计算冒充正式晋级的总pass。下表仅报告同状态同窗口相对BC500的坐标；末端yaw不是完整释放yaw。\n\n|L2窗口|误差变化cm|力变化N|slip变化m/s|对应旧slip容限m/s|末端abs yaw变化°|完整释放旧门|\n|---|---:|---:|---:|---:|---:|---|\n'
for p in s['L2']['pairs']:
 c,b=p['CP'],p['CS'];txt+=f"|{p['state']}|{f(c['error_cm']-b['error_cm'])}|{f(c['force_N']-b['force_N'])}|{f(c['slip_mps']-b['slip_mps'])}|{f(max(.2,.5*b['slip_mps']))}|{f(c['yaw_deg']-b['yaw_deg'],2)}|未以短窗替代|\n"
txt+='\n## 分段实际坐标\n\n阶段由真实伙伴时序与active划分，仅用于评价，不输入控制器。无active时不评价旧anchor跟随；支撑与滑移仍记录。分段短于20ms时q_PD可能无控制边界，标NA不补造。\n\n|状态/执行/阶段|秒|误差cm/力N|slip峰/p90 m/s|pelvis yaw起→末°|root ΔX cm|足部/XYZ等完整数据|\n|---|---:|---|---|---|---:|---|\n'
for row in phases:
 if row['layer']=='L1':continue
 for k,x in row['phases'].items():
  txt+=f"|{row['state_id']} / {row['layer']} / {k}|{f(x['seconds'],2)}|{f(x['error_rms_cm'])}/{f(x['force_rms_N'])}|{f(x['loaded_slip_peak_mps'])}/{f(x['loaded_slip_p90_mps'])}|{f(x['pelvis_yaw_start_deg'],2)}→{f(x['pelvis_yaw_final_deg'],2)}|{f(x['root_delta_cm'][0])}|f1_phase_metrics.json|\n"
(R/'F1_PRICING_AND_SLIP_RESULTS.md').write_text(txt)
(R/'f1_slip_aggregate.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n')
