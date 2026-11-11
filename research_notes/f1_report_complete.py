from pathlib import Path
import json,hashlib,statistics,shutil,subprocess
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/f1-l0-array-audit-20261111';R=S/'runtime/research_notes'
load=lambda p:json.loads(p.read_text());dump=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
C=S/'runs/l0-complete-02'; checks=load(C/'checks.json');assert checks['status']=='completed'
rows=load(C/'state-manifest.json');timing=load(C/'timings.json');queries=load(C/'timing-query-permissions.json');prefix=load(C/'common-prefix-checks.json');models=load(C/'model-identities.json')
jobrows=[dict(name=f.stem,**load(f)) for f in sorted((S/'jobs').glob('*.json'))];assert all(j['status']!='running' for j in jobrows)
old=1113.1868878559908;new=sum(j.get('elapsed_s',0) for j in jobrows);cost=old+new+600;assert cost<=7200
free=shutil.disk_usage(P).free;output=sum(f.stat().st_size for f in (S/'runs').rglob('*') if f.is_file())
by={h:[t['candidate_wall_s'] for t in timing if t['horizon']==h] for h in (1.,3.)}
stat=lambda v:dict(n=len(v),min=min(v),median=statistics.median(v),max=max(v))
restores=[v for r in rows for v in r['restore_seconds']];save=[r['save_seconds'] for r in rows];reads=[r['read_seconds'] for r in rows]
bench=dict(candidate_1s=stat(by[1.]),candidate_3s=stat(by[3.]),full_restore=stat(restores),snapshot_save=stat(save),snapshot_read=stat(reads),replan_100ms_two_1s_candidates=stat([q['query_wall_seconds'] for q in queries]),snapshot_sizes=stat([r['snapshot_bytes'] for r in rows]),branch_bytes=stat([r['bytes'] for r in timing]),backend_setup_s=checks['backend_setup_s'])
cold=[]
for j in jobrows:
 if not j['name'].startswith('l0-complete'):continue
 launch=S/'runs'/j['name']/'launch.json';files=list((S/'runs'/j['name']).glob('P1-1.pt'))
 # No mtime reconstruction of cold-start benchmark: previously recorded job includes other work.
cold_probe=load(S/'runs/cold-probe-01/checks.json');assert cold_probe['status']=='passed'
bench['cold_start']=cold_probe
# Matched observed O-F backend only. C-P predictor remains unimplemented for this stage.
per3=max(by[3.]);per1=max(by[1.]);restore=max(restores);iorate=max(t['save_wall_s'] for t in timing)
unit=per3+restore+iorate
plans=[]
for stage,n in [('L1',24*24),('L2',6*30*4),('L3_locked',4*30*4)]:
 lower=n*(min(by[3.])+min(restores));upper=1.5*(n*unit+max(200,cold_probe['process_entry_to_restored_P1_1_s']*1.2))
 plans.append(dict(stage=stage,planned_3s_branches=n,lower_job_s=lower,conservative_job_s=upper,output_bytes_estimate=n*max(t['bytes'] for t in timing),status='projection_only_not_started',caveat='O-F timing proxy; causal predictor overhead needs own check; L3 requires separate approval'))
ledger=dict(prior_L0_charged_s=old,new_jobs=jobrows,new_job_actual_s=new,preparation_and_reporting_reserve_s=600,cumulative_L0_plus_reserve_s=cost,authorized_combined_cap_s=7200,F1_total_cap_s=43200,L3_reserved_s=10800,L1_actual_s=0,L2_actual_s=0,L3_actual_s=0,remaining_nonL3_total_s=32400-cost,free_bytes_at_report=free,new_output_bytes=output,benchmarks=bench,projections=plans)
dump(R/'f1_cost_ledger.json',ledger)
manifest=dict(version='F1_L0_VALID_STATE_V2',source_plan_sha256=hashlib.sha256((S/'prep/state-plan.json').read_bytes()).hexdigest(),actual_runtime=load(C/'launch.json')['runtime'],states=rows,actual_parent_count=len(set(r['parent_id'] for r in rows)),actual_state_count=len(rows),models=models,comparison_contract='f1_mjdata_contract.py; NULL-pointer zero effective length; typed-byte equality',alias_group='1 extra parent/2 hidden interventions preregistered only; excluded from capability counts',sealed_read=False,permission_audit='f1_info_access_audit.json')
dump(R/'f1_state_manifest.json',manifest);mh=hashlib.sha256((R/'f1_state_manifest.json').read_bytes()).hexdigest()
dump(R/'f1_state_manifest_lock.json',dict(file='f1_state_manifest.json',sha256=mh,source_plan_sha256=manifest['source_plan_sha256'],runtime=manifest['actual_runtime'],sealed_read=False))
dump(R/'f1_l0_complete_checks.json',dict(checks=checks,prefixes=prefix,benchmark=bench,query_permissions=queries,byte_intervention_checks=load(S/'runs/array-audit-03/byte-digest-validation.json')))
table='\n'.join(f"| {q['stage']} | {q['planned_3s_branches']} | {q['lower_job_s']/3600:.3f}–{q['conservative_job_s']/3600:.3f}h | {q['output_bytes_estimate']/2**30:.2f}GiB |" for q in plans)
text=f'''# F1 L0完整续验报告

2026-11-11，按Yi 21:12里程碑式授权完成数组因果审计与L0续验。旧[0d5b92d失败记录](https://github.com/luoxinyuan/compliance_BFM/blob/0d5b92d/research_notes/F1_PREFLIGHT.md)保留；这不是修改旧判据使其通过。

**L0状态：QUALIFIED（当前配置的状态恢复与测速完成；不代表任何接触能力晋级）。**

## 结果与剩余权限

156个真正C指针干预×3秒全部逐值一致，且CPU独立重读156条完整实际输出，字节敏感摘要也全部一致。详见[F1_L0_ARRAY_AUDIT](F1_L0_ARRAY_AUDIT.md)，逐数组生成者、有效域、读写依赖与证据齐全。

恢复覆盖：**{len(rows)}状态/{manifest['actual_parent_count']}父组**，原12主父组24状态＋3原生来源；存盘读回、完整callback/history/RNG恢复和有效MuJoCo状态精确校验完成。没有读封存，没有新模型/训练。共{len(prefix)}个stop/continue父组的已存完整物理前缀核对一致；兄弟不当独立样本。

源码runtime `{manifest['actual_runtime']}`；逐模型完整SHA见[f1_state_manifest.json](f1_state_manifest.json)。manifest锁SHA `{mh}`。C-S固定BC500对照权限保留；**合格C-S planner仍缺格**，没有用真实anchor/未来程序包装受限搜索。此轮测速用真实未来Task程序，明确O-F；C-P因果预测器仅有旧实现来源，尚未在F1验证。饱和别名仍只预登记，未作控制相关性结论。

## 有效状态规则修订

当前3.3.3/Newton/dense/island禁用配置下，原25个非零形状属性实际C指针为NULL。Python读出的临时内存不是有效动力学状态。新增元数据记录NULL字段与实际计数；比较视图将无C存储字段有效长度置0，其他数组、积分/solver/history/RNG仍按dtype/shape/字节严格比。恢复只跳过NULL→NULL的假数组复制，源/目标有效性不同直接失败。具体表及额外iM临时字段说明见逐数组报告。原始失败bytes与快照均保留。

没有改浮点容差、物理质量门、reward、BFM、PD、10N/K80D8/频率；没有宣称别的MuJoCo配置也可忽略这些字段。

## 吞吐与成本

- 完整恢复 n={len(restores)}：中位{statistics.median(restores):.4f}s，范围{min(restores):.4f}–{max(restores):.4f}s。
- 1秒候选 n={len(by[1.])}：中位{statistics.median(by[1.]):.3f}s，范围{min(by[1.]):.3f}–{max(by[1.]):.3f}s。
- 3秒候选 n={len(by[3.])}：中位{statistics.median(by[3.]):.3f}s，范围{min(by[3.]):.3f}–{max(by[3.]):.3f}s。
- 100ms重规划实测：每次2个1s候选→物理评分选择→只执行首100ms，n={len(queries)}，中位{bench['replan_100ms_two_1s_candidates']['median']:.3f}s。远慢于100ms，不称实时部署。
- 快照保存中位{statistics.median(save):.4f}s、读回{statistics.median(reads):.4f}s；backend设置{checks['backend_setup_s']:.3f}s。独立新进程从入口到model/environment就绪{cold_probe['process_entry_to_model_environment_s']:.3f}s、task/actor就绪{cold_probe['process_entry_to_usable_task_and_actor_s']:.3f}s、P1-1重建就绪{cold_probe['process_entry_to_restored_P1_1_s']:.3f}s；这是进程冷启动，没有清空共享文件/驱动缓存。
- 新作业实测{new:.2f}s＋新准备/收尾reserve600s＋旧L0保守{old:.2f}s＝{cost:.2f}s（{cost/3600:.3f}h），使用L0与授权reserve合计7200s内；没有动L1/L2/L3额度。新输出{output/2**30:.3f}GiB，报告时共享盘{free/2**30:.2f}GiB>100GiB。

## 后续成本投影（不是发车）

| 层 | 暂拟3s候选分支数 | 基于实测的成本范围 | 按完整分支存档估计 |
|---|---:|---:|---:|
{table}

L1按24状态×24候选；L2按6状态×30个100ms反馈点×4候选；L3只作锁定额度投影4状态×30×4，不生成队列。上界用实测最大候选/恢复/IO＋max(200s,1.2×实测新进程前缀就绪时间)固定开销再乘1.5，非统计置信区间。C-P预测器若改变实际成本需重测；C-S planner缺格不能用这些数字冒称已实现。正式后续还须锁候选/权限并满足对应层额度；不在本次执行。

## 失败与范围

保留两次未完成的Python数组干预检查，以及CPU/GPU读回比较器设备不一致的工程失败。没有删去失败成本；更正后的比较按数值原始字节，不通过设备一致性或宽松epsilon掩盖。所有运行唯一目录，旧a39be58/Issue57/Issue59不改判。

本次只完成L0；没有L1/L2/L3科学搜索、Teacher标签生产、PPO/Student/FB更新。下一步是按合同决定后续有限搜索，不自动启动。
'''
(R/'F1_PREFLIGHT.md').write_text(text)
print('report',len(rows),'cost',cost,'runtime',manifest['actual_runtime'])
