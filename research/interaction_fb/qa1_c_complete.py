"""Complete matched QA1 report after successful C; never edits old result files."""
import hashlib,json,sys,time
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/r2-g17-qa1-20261115';O=P/'experiments/r2-v11-qentry-20261114';R=S/'runtime-cfix';RUN=S/'runs/C-repaired-04'
sys.path.insert(0,str(R/'research/interaction_fb'))
from supervision import tree_bytes
data=json.loads((RUN/'qa1-result.json').read_text());proof=json.loads((RUN/'post-first-update-load-isolation.json').read_text())
assert data['status']=='QA1_FIXED_REPLAY_COMPLETE' and proof['source_unchanged'] and proof['next_independent_learner_exact']
ledger=json.loads((S/'ledger.json').read_text());old=json.loads((O/'ledger.json').read_text())
assert ledger['jobs'][-1]['name']=='C-repaired-04' and ledger['jobs'][-1]['exit_code']==0
assert data['retrospective']['U']['new_Q_groups']==data['retrospective']['C']['new_Q_groups']==456
assert data['actor_updates']==data['I_updates']==0
if data['prospective_eligible']:
    print('STOP_REPORT_ONLY: prelocked prospective condition triggered; do not finalize before its authorized evaluation');sys.exit(2)
entry=dict(name='QA1-C-final-audit-report-publication',category='G17-QA1',wall_s=300,cost_type='conservative_upper_bound',
    includes='full state/optimizer hash review, matched sampler comparisons, strict storage scan, documents and public milestone updates',physics_ticks=0,optimizer_updates=0)
assert not any(x.get('name')==entry['name'] for x in ledger['jobs'])
ledger['jobs'].append(entry);old['job_entries'].append(entry);ledger['charged_upper_s']+=300;old['total_charged_upper_s']+=300
physics={k:sum(j.get('physics_counts',{}).get(k,0) for j in ledger['jobs']) for k in ('actual','reserved')}
output=tree_bytes([S]);alloutput=tree_bytes(old['output_roots'])
assert ledger['charged_upper_s']<=3600 and output<1024**3 and physics['reserved']<=18000
assert old['total_charged_upper_s']<=57600 and alloutput<12*1024**3
ledger.update(status='QA1_COMPLETE_ENDPOINT_FAIL',diagnostic_Q_groups=dict(U=456,C=456),physics_counts=physics,
    output_bytes=output,R2_output_bytes=alloutput,stop_reason='Both fixed512 retrospective endpoints/calibration not qualified. No prospective or Actor/I authorized by results.',
    C_repair_regression=proof)
old.update(status='G17_QA1_COMPLETE_ENDPOINT_FAIL',g17_qa1=dict(new_Q_groups_U=456,new_Q_groups_C=456,actor_updates=0,
    original_G17='FAIL_UNCHANGED',main_replay_rows_added=0,prospective='NOT_TRIGGERED'))
(S/'ledger.json').write_text(json.dumps(ledger,indent=2));(O/'ledger.json').write_text(json.dumps(old,indent=2))
rows=[]
for node in (56,128,256,512):
    row=dict(node=node)
    for arm in ('U','C'):
        rr=data['retrospective'][arm]['nodes'][str(node)];row[arm]=dict(correct=rr['correct'],accuracy=rr['accuracy'],unique=rr.get('unique',1838),
            parent_balanced=rr['parent_balanced_accuracy'],by_round=rr['by_round'],scores=rr['scores'])
    rows.append(row)
updates={a:json.loads(((S/'runs/fixed-replay-02') if a=='U' else RUN).joinpath(a+'-updates.json').read_text()) for a in ('U','C')}
assert len(updates['U'])==len(updates['C'])==456
for u,c in zip(updates['U'],updates['C']):
    assert u['total_group']==c['total_group']
    assert {s:v['draws'] for s,v in u['by_stratum'].items()}=={s:v['draws'] for s,v in c['by_stratum'].items()}
snapshots={p.name:json.loads(p.read_text()) for p in RUN.glob('C-Q*.manifest.json')}
for name,info in snapshots.items():assert hashlib.sha256((RUN/name.replace('.manifest.json','.ptz')).read_bytes()).hexdigest()==info['sha256']
ans=dict(status=ledger['status'],runtime_C='e1fe70b20fc48994638fade189e415e9da61dd0e',runtime_U='9d48d4d7328c24ef8518a8af67e097c8af4be38a',
    old_G17='FAIL_UNCHANGED',source_manifest_sha256=data['source_manifest_sha256'],nodes=rows,C_snapshots=snapshots,
    repair_proof=proof,U_reused_without_new_update=True,actual_stratum_draws_equal_all456_groups=True,
    new_Q_groups=dict(U=456,C=456),actor_updates=0,I_updates=0,new_online_rows=0,new_teacher_queries=0,
    prospective='NOT_TRIGGERED_ENDPOINT_FAIL',physics_counts=physics,QA1_job_s=ledger['charged_upper_s'],R2_job_s=old['total_charged_upper_s'],
    QA1_output_bytes=output,R2_output_bytes=alloutput,C_job={k:ledger['jobs'][-2][k] for k in ('name','runtime','limit_s','wall_s','exit_code','physics_counts','status')})
(R/'research_notes/r2_ei_qa1_complete_results.json').write_text(json.dumps(ans,indent=2))
lines=['# G17-QA1完整匹配诊断：C修复完成，固定终点未获资格','',
 '2026-11-15。Yi追加授权修复C并在原子额度内跑完。U原结果严格只读复用，C从原E Q56/3500行/targets/Adams出发完成456新增Q组；两臂总组数512硬终点。**本轮完成；不触发前瞻，不启动Actor/U/I。原G17 FAIL不改。**',
 '', '## 更新量与同层行覆盖', '',
 '|总Q组数|U正确/58|C正确/58|U已曝光行|C已曝光行|','|---|---:|---:|---:|---:|']
for x in rows:lines.append(f"|{x['node']}|{x['U']['correct']} ({x['U']['accuracy']:.2%})|{x['C']['correct']} ({x['C']['accuracy']:.2%})|{x['U']['unique']}|{x['C']['unique']}|")
u=rows[-1]['U'];c=rows[-1]['C']
lines += ['',f"固定512终点：C−U={(c['accuracy']-u['accuracy'])*100:+.2f}个百分点。原门仍为58非平凡比较、四有效父组、≥70%；不改用父组等权（U{u['parent_balanced']:.2%}、C{c['parent_balanced']:.2%}）或事后选择中间节点。旧40条均为已暴露retrospective fit，不是新G17或新物理收益。",'',
 '完整核对456批实际累计父组×物理phase抽样次数逐项相同。C只改变同一层内选哪些低曝光行，包含历史56组曝光；未改变task/phase配额。3500行来自六父组；资格旧40条为五锁定父组，其中四组有有效比较。没有把帧数当独立父组。',
 '', '|旧暴露状态|U512正确/有效|C512正确/有效|','|---|---:|---:|']
names={2:'near入口',4:'mid入口',5:'mid保持',7:'zero',9:'stop保持',11:'continue',12:'continue恢复',14:'新near恢复'}
for n,label in names.items():
    a=u['by_round'][str(n)];b=c['by_round'][str(n)];lines.append(f"|{label}|{a['correct']}/{a['n']}|{b['correct']}/{b['n']}|")
lines+=['', '## 价值与target语义', '',
 '新C运行重做原同状态只读前置审计；每20ms reward/discount/backup和1s n-step残差保留。原U的1968个target mean与实际共同续跑action/z逐值一致；当前C审计读取其独立审计文件。没有用“无限时域不同”遮掩高成本前缀的明显价值误差。隐藏未来造成相反最优动作仍UNMEASURED；不能把条件价值解释风险直接当作已证明的软件bug。',
 '', '|状态|真实1s G|U512 Q|C512 Q|U n-step残差|C n-step残差|','|---|---:|---:|---:|---:|---:|']
for n in (4,5,9,11):
    a=next(x for x in u['scores'] if x['round']==n and x['candidate']=='mean');b=next(x for x in c['scores'] if x['round']==n and x['candidate']=='mean')
    lines.append(f"|{names[n]}|{a['return_1s']:.6f}|{a['q']:.6f}|{b['q']:.6f}|{a['nstep_residual']:.6f}|{b['nstep_residual']:.6f}|")
target=json.loads((RUN/'preupdate-target-audit.json').read_text());assert target['exact_controls']==target['controls_compared']==1968 and target['max_action_difference']==target['max_z_difference']==0
lines += ['', 'C前置审计1968/1968控制action/z逐值一致，包括100ms边界。本轮目标策略错位检查通过；不意味着价值拟合或动作排序已可信。',
 '', '## 修复、回归与可恢复资产', '',
 '新增入口原失败源于CPU Adam step storage别名。每次load传入完整deep-copy；逐项检查Q Adam CPU张量不与原source共享storage。CPU小回归验证同数学更新、源不变和下一独立装载完全一致；C第57组真实更新后再创建独立learner，完整source/model/targets/Adams/RNG检查通过，且不额外增加Q组数。128/256/512节点均核原source缓存和非Q状态不变，四个C完整agent包逐值读回。',
 '', 'U的四个既有manifest和包SHA验证后只读复用，无U optimizer重跑。C源码`e1fe70b20fc48994638fade189e415e9da61dd0e`，U源码`9d48d4d7328c24ef8518a8af67e097c8af4be38a`；原Q56 manifest `2068e10d41e6cace8dfc5bef840e444562ad395af2767b06d996fb911bcc21de`。所有C包SHA、逻辑digest见[机器摘要](r2_ei_qa1_complete_results.json)。',
 '', '首版CPU测试因fork_rng默认枚举CUDA而产生初始化警告；已显式devices=[]且CUDA不可见，最终回归未初始化CUDA。一次启动在修复提交未完成时被clean-worktree守门拒绝，子物理/Q均0；等提交完成后使用新运行身份启动。两项保留，不改科研数据或判据，成本纳入480秒修复准备上界。原IPC/装载首失败与旧部分报告保留，见[原erratum](R2_EI_G17_QA1_ERRATUM.md)及[修复结果](R2_EI_G17_QA1_C_REPAIR_RESULTS.md)。',
 '', '## 成本与停止', '',
 f"QA1累计{ledger['charged_upper_s']:.6f}/3600job秒，新增输出{output/1024**2:.3f}MiB/1GiB，actual/reserved ticks={physics['actual']}/{physics['reserved']}（reserved已包含actual）；R2累计{old['total_charged_upper_s']:.6f}/57600秒、输出{alloutput/1024**3:.3f}/12GiB。C作业{ledger['jobs'][-2]['wall_s']:.6f}秒。新主轨/Teacher查询/Actor/U/I/Dev/Sealed均0。",'',
 '修复与worktree准备保守480秒、最终核查/IO/发布保守300秒均追加原账，不重置旧1041.132584秒。两臂各新增456 Q组（每组两个Q Adam各一步）；这是固定数据诊断，非新的on-policy行为训练。新tick仅冷启动/真实原生前缀，不是新接触资格分支。',
 '', '## 证据分级与下一问题', '',
 '**已证实：**CPU起点隔离修复及实际更新后独立再装载通过；同批配额匹配成立；更多Q更新/行曝光对旧排序有影响，但固定512终点尚未取得可指导Actor的资格。',
 '**相容：**一步TD自举的时间传播/价值条件性问题仍可能重要；行覆盖即使充分，也不自动保证细小动作收益排序。',
 '**未证：**I交互表示是否有额外行为价值；改善旧排序是否可迁移到新父组；这些Q是否能真正带来更好的动作。',
 '', '下一最有判别力的问题是用同一实际数据、固定后续策略验证有限多步return/价值传播目标，而不是在当前Q上自动增加Actor更新或再补节点；这只是下一方法建议，**未启动、需另立科学合同**。本轮至此停止。']
(R/'research_notes/R2_EI_G17_QA1_COMPLETE_RESULTS.md').write_text('\n'.join(lines)+'\n')
repair=f'''# QA1 C装载修复结果

Yi已在当前会话明确授权C修复并在原子额度内跑完。原erratum的历史失败不重判；本次装载问题已通过实际回归关闭。

唯一修复：`pristine_load`将完整source深拷贝后交给原`ApprovedLearner.load`。不修改原q_update、loss、lr、EMA、sampling schedule或科学门。

CPU Adam fixture：PASS，源缓存不变、相同数学更新、后续独立装载逐值一致，最终CUDA_initialized=false。C第一实际更新组57（属于456组内）后，原source完整digest仍一致、另一个learner载入完整Q56逐值一致，RNG不变。C128/256/512非Q部分与source缓存hash均保持，C56/128/256/512完整agent包读回一致。原U包SHA只读复核，未重跑。

实际运行e1fe70b20fc48994638fade189e415e9da61dd0e；456新增C Q组、Actor/U/I更新0。最终科学结果见[完整QA1报告](R2_EI_G17_QA1_COMPLETE_RESULTS.md)，不能以工程修复PASS替代Q资格。

原失败/日志/93b5f48部分报告保留。首次CPU fixture的默认fork_rng CUDA初始化警告已改为devices=[]，最终回归不初始化CUDA；一次clean-worktree启动拒绝没有子训练/物理。成本均在480秒修复准备上界中，不漏计也不虚构实际训练。
'''
(R/'research_notes/R2_EI_G17_QA1_C_REPAIR_RESULTS.md').write_text(repair)
err=R/'research_notes/R2_EI_G17_QA1_ERRATUM.md';text=err.read_text();err.write_text(text+'\n## C修复闭环（2026-11-15追加）\n\nYi新授权后已完成修复及C456组，工程别名问题关闭；历史未关闭状态描述保留为原时点。见[C修复结果](R2_EI_G17_QA1_C_REPAIR_RESULTS.md)与[完整科学结果](R2_EI_G17_QA1_COMPLETE_RESULTS.md)。未放宽任何断言或门。\n')
headline=f'''# 最新：G17-QA1 U/C均完成，固定512终点未获资格

2026-11-15。Yi追加授权C修复续跑，独立快照装载＋真实首更新后source/Adam/target/RNG回归通过。U原数据只读复用，C原锁456组已完成。固定512：U {u['correct']}/58={u['accuracy']:.2%}，C {c['correct']}/58={c['accuracy']:.2%}；未取得原门及校准资格，不触发前瞻、不启动Actor/U/I/Dev/Sealed。旧G17 FAIL和全部错误保留。完成即停，无自动追加节点/调参。

[完整结果](R2_EI_G17_QA1_COMPLETE_RESULTS.md) · [C修复回归](R2_EI_G17_QA1_C_REPAIR_RESULTS.md) · [机器摘要](r2_ei_qa1_complete_results.json)。QA1累计{ledger['charged_upper_s']:.6f}/3600job秒、{output/1024**2:.3f}MiB/1GiB；R2累计{old['total_charged_upper_s']:.6f}/57600秒。原U runtime9d48d4d、C runtimee1fe70b；原Q56/BFM/PD不改。以下为历史时点。

---

'''
path=R/'research_notes/CURRENT_STATE.md';path.write_text(headline+path.read_text())
path=R/'research_notes/RESEARCH_OVERVIEW.md';text=path.read_text();note=f'''## 最新补充：G17-QA1完整匹配诊断

C装载别名问题已在Yi追加授权下修复并通过实际更新后独立再装载回归。U只读复用、C完成原456组，512终点U {u['accuracy']:.2%}、C {c['accuracy']:.2%}，尚未获原Q资格及价值校准，Actor/U/I和前瞻不启动。旧40条仅retrospective诊断；工程修复、更多更新和旧排序变化不等于学会新接触动作。见[完整结果](R2_EI_G17_QA1_COMPLETE_RESULTS.md)；旧部分报告/失败/erratum保留。下一问题应聚焦Q的多步信用与排序目标，未自动开新实验。

''';path.write_text(text.replace('## 科学目标',note+'## 科学目标',1))
print(json.dumps(dict(status=ledger['status'],U=u['correct'],C=c['correct'],QA1_s=ledger['charged_upper_s'],R2_s=old['total_charged_upper_s'],bytes=output,ticks=physics),indent=2))
