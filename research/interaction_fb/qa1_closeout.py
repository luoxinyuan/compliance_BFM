"""Publish partial QA1 and first-failure evidence, without new updates/physics."""
import hashlib,json,sys,time
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'experiments/r2-execution-20261114/runtime';S=P/'experiments/r2-g17-qa1-20261115';O=P/'experiments/r2-v11-qentry-20261114'
sys.path.insert(0,str(R/'research/interaction_fb'))
from supervision import tree_bytes
d=json.loads((S/'prep/failure-readonly-01/diagnostic.json').read_text());ledger=json.loads((S/'ledger.json').read_text());old=json.loads((O/'ledger.json').read_text())
assert ledger['status']=='STOP_FIRST_FAILURE' and not any((S/'runs/fixed-replay-02').glob('C-Q*.ptz'))
entries=[dict(name='QA1-single-readonly-failure-diagnosis',category='G17-QA1',wall_s=d['wall_s'],physics_ticks=0,optimizer_updates=0),
 dict(name='QA1-ipc-fixture-erratum-report-and-publication',category='G17-QA1',wall_s=240,cost_type='conservative_upper_bound',physics_ticks=0,optimizer_updates=0)]
for rec in entries:ledger['jobs'].append(rec);old['job_entries'].append(rec)
charge=sum(x['wall_s'] for x in entries);ledger['charged_upper_s']+=charge;old['total_charged_upper_s']+=charge
ledger.update(status='STOP_AWAIT_DECISION',diagnostic_Q_groups=dict(U=456,C=0),actor_updates=0,I_updates=0,
    physics_counts=dict(actual=1401,reserved=1401),new_main_online_rows=0,stop_reason='C initial full-agent digest mismatch before optimizer; CPU Adam step alias confirmed. One readonly diagnosis used, no automatic repair/retrain.')
old.update(status='G17_QA1_STOP_AWAIT_DECISION',g17_qa1=dict(new_Q_groups_U=456,new_Q_groups_C=0,actor_updates=0,original_G17='FAIL_UNCHANGED',main_replay_rows_added=0))
outbytes=tree_bytes([S]);allbytes=tree_bytes(old['output_roots']);ledger['output_bytes']=outbytes;ledger['R2_output_bytes']=allbytes
assert ledger['charged_upper_s']<=3600 and outbytes<=1024**3 and old['total_charged_upper_s']<=57600 and allbytes<=12*1024**3
(S/'ledger.json').write_text(json.dumps(ledger,indent=2));(O/'ledger.json').write_text(json.dumps(old,indent=2))
public=dict(status=ledger['status'],old_G17='FAIL_UNCHANGED',runtime='9d48d4d7328c24ef8518a8af67e097c8af4be38a',
    source_manifest_sha256='2068e10d41e6cace8dfc5bef840e444562ad395af2767b06d996fb911bcc21de',
    original_source_unchanged=d['source_manifest_unchanged'],U56_equals_original=d['U56_equals_original'],U512_nonQ_exact=d['U512_nonQ_exact'],
    U_nodes=d['summary'],C_groups=0,new_Q_groups=dict(U=456,C=0),new_online_rows=0,actor_updates=0,I_updates=0,
    ticks=ledger['physics_counts'],qa1_job_s=ledger['charged_upper_s'],R2_job_s=old['total_charged_upper_s'],
    qa1_output_bytes=outbytes,R2_output_bytes=allbytes,prospective='NOT_RUN: U512 fails70%; C not started due immutable-initial-state guard',
    snapshots={p.name:json.loads(p.read_text()) for p in (S/'runs/fixed-replay-02').glob('U-Q*.manifest.json')},
    alias_evidence=[{k:v for k,v in x.items() if k!='torch_policy_source'} for x in d['alias_evidence']])
(R/'research_notes/r2_ei_qa1_results.json').write_text(json.dumps(public,indent=2))
lines=['# G17-QA1：U固定终点完成，C起点断言停止（部分结果）','',
 '2026-11-15。Yi已批准修订并归档；原G17 FAIL不改。**当前STOP_AWAIT_DECISION，不是U/C实验完整结项，不具备恢复Actor/U或I的资格。**',
 '', '## 已执行与结果', '',
 '|总Q组数|U正确/原58对|准确率|已曝光行/3500|', '|---|---:|---:|---:|']
for x in d['summary']:lines.append(f"|{x['node']}|{x['correct']}/58|{x['accuracy']:.2%}|{x['unique']}|")
lines+=['', '各节点仅旧40条已暴露物理分支上的critic前向，**retrospective fit，非新G17物理通过**。512为固定终点，未按128/256挑模型。父组等权在512为71.45%，但合同门是原58比较的62.07%，不得替换分母宣称通过。',
 '', '512分状态：near入口9/9，mid入口3/10，mid保持0/9，zero10/10，stop7/10，continue5/7，continue恢复2/3，新near恢复0/0（缺少非平凡对）。四有效父组、所有旧失败保留。',
 '', '## 目标与价值语义审计', '',
 '1968个实际控制子步，冻结target mean与物理共同续跑的action/z逐值相同（包括100ms边界）；故未发现该续跑策略错位。原40条首动作、previous/target z和奖励重积分依据原D11复核，本轮另保存每20ms backup及n-step残差。',
 '', '|状态|真实1s G|Q56|U512 Q|U512 1s n-step残差|','|---|---:|---:|---:|---:|']
for rnd in (4,5,9,11):
 x=next(x for x in d['summary'][0]['mean_values'] if x['round']==rnd);y=next(x for x in d['summary'][-1]['mean_values'] if x['round']==rnd)
 lines.append(f"|round{rnd}|{y['return_1s']:.6f}|{x['q']:.6f}|{y['q']:.6f}|{y['nstep_residual']:.6f}|")
lines+=['', '中距保持及停止仍10N、高误差，Q数值仍明显乐观；增加更新并未消除长程传播/条件价值风险。负的确定性前缀成本只在相同确定性未来条件下给路径上界；Q未必观测完整未来spec，不能仅以这些差值宣布实现bug，也不能以“时域不同”忽略它。相同完整过去、不同未来导致相反最优动作：**UNMEASURED**，现存40分支不能证明。',
 '', '## 首错误与有效范围', '',
 '运行源码 `9d48d4d7328c24ef8518a8af67e097c8af4be38a`。U从原Q56起点完整hash一致，456新增Q组及各节点读回通过，U512非Q状态逐值不变。切C时完整起点断言失败，C优化0。一次只读诊断确认：`Optimizer.load_state_dict`在capturable/fused false时保留CPU Adam step storage；源内存缓存与U优化器step别名，U更新将该缓存step从56改为512。磁盘原source manifest及完整agent digest均未变。',
 '', '这是新增两分支装载入口错误；**不能拿C不一致起点继续训练，也不能把单U结果写成U/C比较**。U自己的Adam正常更新、存盘前后完整检查成立，故保留其描述性结果；未启动C/前瞻/Actor/I/Dev/Sealed。详见[独立erratum](R2_EI_G17_QA1_ERRATUM.md)。',
 '', '## 成本与身份', '',
 f"QA1累计{ledger['charged_upper_s']:.6f}/3600job秒；新增输出{outbytes/1024**2:.3f}MiB/1GiB；R2累计{old['total_charged_upper_s']:.6f}/57600秒、输出{allbytes/1024**3:.3f}/12GiB。1401 actual /1401 reserved 5ms ticks，仅冷启动/源前缀；reserved包含actual，不重复相加。新增主轨0，U Q组456（每组两Adam各一步），C0，Actor/U/I0，Teacher query0。",'',
 '首作业IPC启动错误11.242017s，physics/updates0，保留原log。修复复用既有抽象socket，forkserver回归通过；第二作业163.076023s。只读错误诊断26.814544s、CPU未初始化CUDA；准备上界600s及修复/报告/IO上界240s分别计账。',
 '', '原E Q56 manifest `2068e10d41e6cace8dfc5bef840e444562ad395af2767b06d996fb911bcc21de`；完整SHA与节点模型/Adam/target逻辑digest见[机器摘要](r2_ei_qa1_results.json)。BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`不改。',
 '', '## 判读与下一单一干预', '',
 '**已证实：**旧目标续跑合同匹配；U补更新/曝光令旧排序44.83%→62.07%，但未过门；全部3500行已曝光仍不能解决mid排序。**相容：**目标传播和同层内部曝光结构仍可能相关。**未证：**C能否带来额外收益、Q是否能真正改善Actor、I的额外价值。',
 '', '最小工程修正是每次分支load使用独立deep-copy完整初态，并用新实例验证model/target/每个Adam字段与起点全hash一致；一次隔离更新应证明源缓存未变，再验证另一分支起点。修复不涉及sampler/lr/EMA/reward，但本修订§6明确错误诊断后STOP_AWAIT_DECISION，故**本轮仅提供修复建议，未自动改装载/重跑C**。下一科学问题仍是同预算U/C，不追加节点、旧样本或调参。']
(R/'research_notes/R2_EI_G17_QA1_RESULTS.md').write_text('\n'.join(lines)+'\n')
err='''# G17-QA1 独立工程勘误（不覆盖旧G17）

2026-11-15；状态：已定位，未关闭。首科学正确性阻断：切换C分支前的完整起点hash不同，C更新0。保留原运行与traceback；不回写旧模型、old G17结论或成功门。

## 证据

- `research/interaction_fb/qa1_trial.py:121–122` 对同一个内存source再次`agent.load(s['agent'])`，随即全hash断言失败。
- `approved_learning.py::load` 直接将传入各Adam state交给`optimizer.load_state_dict`；PyTorch `_process_value_according_to_param_policy` 对非capturable/fused的`step`返回原tensor。
- 同真实Q56第一参数状态、CPU read-only load fixture：Q1/Q2 step storage pointer均相同；未backward、未optimizer.step、未初始化CUDA。Q56磁盘step56，U512磁盘step512。
- 原source manifest以及完整agent SHA/digest与签前一致；U56完整agent等原source；U512所有非Q可变部分digest相同。

直接证据见私有stage `prep/failure-readonly-01/diagnostic.json`；公开摘要不携带原始轨迹/路径。影响仅本新入口的多分支共享内存初态。不能据此声称历史其它恢复路径出错；历史单进程/新进程恢复已有独立验证。

## 最小修正建议（未施作）

`agent.load(copy.deepcopy(pristine_initial_agent))`，每个分支独立对象与完整校验。回归必须包括一次真实隔离Q更新前后源缓存不变、下一分支model/target/Adams/RNG初态一致，以及loss/sampler合同不变。没有通过这些检查前不得重跑C。当前QA1§6要求一次只读诊断后STOP_AWAIT_DECISION，因此没有自动修复或继续。

## 另一个启动错误（已工程关闭，未改变学习）

首作业49dc30a在模型加载前因CPFS Unix socket Errno95退出，物理/优化0。新入口顶层补入已有`m11_ipc`，使forkserver子进程也复用Linux abstract socket；独立Manager/Queue读写fixture通过。修复源码9d48d4d，科学参数不变；原log及11.242017s保留。它不涉及Q、物理状态恢复、概率或采样定义。
'''
(R/'research_notes/R2_EI_G17_QA1_ERRATUM.md').write_text(err)
head=f'''# 最新：G17-QA1部分完成，C起点一致性失败后按合同停止

2026-11-15。Yi批准修订已归档。E-only U从Q56固定3500行续至512：旧40条排序26/58→36/58=62.07%，未过70%；mid保持仍0/9。U的3500行全部曝光、非Q状态不变。切C前full-agent hash断言捕获CPU Adam step与起点缓存别名，C更新0；一次只读定位完成，磁盘原source未变。**STOP_AWAIT_DECISION**，按QA1§6不自动修复重跑；非完整U/C比较，非新G17通过。Actor/U/I/Dev/Sealed/前瞻均0，旧FAIL不改。

[实际结果](R2_EI_G17_QA1_RESULTS.md) · [独立erratum](R2_EI_G17_QA1_ERRATUM.md) · [批准协议](R2_EI_G17_QA1_PROTOCOL.md)。1968控制子步target action/z与实际共同mean续跑逐值一致；中距/停止Q仍乐观。QA1累计{ledger['charged_upper_s']:.6f}/3600job秒、{outbytes/1024**2:.3f}MiB/1GiB；R2累计{old['total_charged_upper_s']:.6f}/57600秒。GPU0已释放，无后续队列。以下为历史时点。

---

'''
p=R/'research_notes/CURRENT_STATE.md';p.write_text(head+p.read_text())
p=R/'research_notes/RESEARCH_OVERVIEW.md';text=p.read_text();marker='## 科学目标';note='''## 最新修订：G17-QA1部分诊断

Yi已批准固定3500行的E-only U/C匹配诊断。U增加至512组后旧排序44.83%→62.07%，仍未过原70%；mid保持0/9，不能用父组等权71.45%替代门。C在首更新前被完整起点hash守门拦下：CPU Adam step与重复load源缓存别名；原磁盘未变。一次只读定位后按QA1§6 STOP_AWAIT_DECISION，没有自动重跑、Actor或I。它是部分工程阻断＋单U描述性收益，不是行为训练失败或U/C方法胜负。见[QA1结果](R2_EI_G17_QA1_RESULTS.md)和[erratum](R2_EI_G17_QA1_ERRATUM.md)；历史结论保持。

'''
p.write_text(text.replace(marker,note+marker,1))
print(json.dumps(dict(QA1=ledger['charged_upper_s'],R2=old['total_charged_upper_s'],new_bytes=outbytes,all_bytes=allbytes,status=ledger['status']),indent=2))
