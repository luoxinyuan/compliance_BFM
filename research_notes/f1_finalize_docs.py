from pathlib import Path
import json,hashlib
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/f1-l0-array-audit-20261111';R=S/'runtime/research_notes'
load=lambda p:json.loads(p.read_text());dump=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
C=S/'runs/l0-complete-02';cold=load(S/'runs/cold-probe-01/checks.json');ledger=load(R/'f1_cost_ledger.json');man=load(R/'f1_state_manifest.json')
man['original_BC500_sha256']='a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e';man['frozen_BFM_sha256']=cold['bfm_sha'];man['parent_count_semantics']='12 main exposed parent groups +3 native-source reference IDs; native references not independent randomized replicates and not in contact capability denominator'
dump(R/'f1_state_manifest.json',man);oldlock=load(R/'f1_state_manifest_lock.json');oldhash=oldlock['sha256'];oldlock['sha256']=hashlib.sha256((R/'f1_state_manifest.json').read_bytes()).hexdigest();dump(R/'f1_state_manifest_lock.json',oldlock)
p=R/'F1_PREFLIGHT.md';txt=p.read_text().replace(oldhash,oldlock['sha256']).replace('**27状态/15父组**','**27状态/15个登记父组或来源ID**').replace('原12主父组24状态＋3原生来源；','原12主父组24状态＋3原生来源参考（后者不作为3个独立随机重复，不纳入接触胜率）；');p.write_text(txt)
ledger['budget_allocation']=dict(L0_charged_s=min(1800,ledger['cumulative_L0_plus_reserve_s']),additional_reserve_charged_s=max(0,ledger['cumulative_L0_plus_reserve_s']-1800),L1_borrowed_s=0,L2_borrowed_s=0,L3_borrowed_s=0)
dump(R/'f1_cost_ledger.json',ledger)
p=R/'f1_info_access_audit.json';x=load(p);x['version']=2;x['runtime']=man['actual_runtime'];x['scopes']['L0_timing_repeat']='O-F exact scenario future benchmark; three two-candidate/first100ms timing queries completed; not scientific L1/L2 search';x.pop('missing_query_records',None);x['queries']=load(C/'timing-query-permissions.json');x['all_L1_L2_L3_counts']=0;dump(p,x)
p=R/'F1_STATE_AND_SEARCH_COVERAGE.md';p.write_text('''# F1状态与信息覆盖（L0续验完成）

锁定清单见[f1_state_manifest.json](f1_state_manifest.json)，文件锁见[f1_state_manifest_lock.json](f1_state_manifest_lock.json)。原source plan hash未变：12主父组24状态；3原生source参考各1状态，共27状态/15个登记ID。3原生参考不作为独立随机重复，也不计接触能力成功分母。全部已实际重建、保存、读回及三次有效状态恢复校验。旧sealed不读。

角色选择沿用首L0预登记：P1两个严格持续正例及同父组后续相位；新dev三个指定mid/stop负例；旧reg已曝光退化的2个stop和2个mid；另1near、2stop父组。未按本轮结果换父组或挑好状态。实际快照时间单列，区分日志中的执行前obs与执行后decision时间。

三原生参考由原始z与原生初态真实执行到6s；qpos/qvel/force/support前缀逐值核对旧存档。只确认这些恢复资产，不以此宣布前进motion的滑移/yaw局限已消失。5个stop/continue父组存档前缀分别500/400/740/320/1020个5ms tick一致，兄弟分支保持同一父组身份。

数组干预是3个状态上156条相关分支，绝不是156个独立父组。2个真实伙伴受力状态与1个无伙伴接触状态各52条3秒干预；逐字段与联合扰动均有未扰动参考。无伙伴力状态仍含真实地面支撑约束。

额外别名父组1、相关分支2，只预登记，不构造、不入能力统计。L1/L2/L3为0。L0另测三次2候选1秒→执行首100ms的端到端开销，及三个状态1s/3s各两次严格重复；均标O-F计时控制，不当能力搜索结果。

权限见[f1_info_access_audit.json](f1_info_access_audit.json)。C-S固定BC500可作受限对照；**C-S planner缺格**；旧C-P因果预测模块未在本L0完成新F1实现/资格。不能用精确未来Task程序补出C-S/C-P成功。
''')
p=R/'F1_ERRATUM.md';p.write_text(p.read_text()+'''

## 2026-11-11 21:12新授权后的更正与证据

此前“25数组可能为工作区”的假设已补足实际C指针和156条物理干预证据：当前配置这些指针NULL，Python绑定生成未初始化临时数组。额外发现同类iM非零形状字段，其他NULL字段多数shape0。新规则见F1_L0_ARRAY_AUDIT；旧首轮严格失败原判和产物保留，不能声称它当时已经通过。

- `array-audit-01`（bbd207b）：第一个NumPy字段置零后非目标portable hash不同，立即停；只完成3s未干预参考，没有合格干预。
- `array-audit-02`（147a93d）：保存before/after和别名检查；观察到NULL属性反复读取的临时内存变化，包括iM。不是实际有效状态被连带改写。未作合格干预。
- `array-audit-03`（5e5b3a4）：用已安装header的offsetof访问真正C指针，逐一/联合挂入owned零值或存档随机buffer；其他C结构和有效数组/RNG不动。156×3秒全部通过；追加CPU重读字节摘要156/156一致。
- `l0-complete-01`（0b36af7）：存盘map_location=CPU与现场GPU张量被旧torch.equal直接比较，发生设备RuntimeError。不是数值不一致；没有放宽容差。`l0-complete-02`（5f4c82c）用dtype/shape/原始bytes比较，27状态、81次恢复、重复物理和5个共同前缀通过。
- `cold-probe-01`（8bb0de3）：独立进程冷/热测速及3s重复通过。没有清空共享cache。

所有失败计入账本。计时现逐分支写出再断言，避免首轮timings未落盘。只修复状态保存/比较/恢复工具，不改概率、物理、reward、质量门或封存选模。没有新FB update，因此只回填Issue57的portable有效域残余，不改变旧数值更新失败的历史判读。
''')
p=R/'CURRENT_STATE.md';old=p.read_text();tail=old[old.index('## Issue59已结项状态'):];head=f'''# CURRENT_STATE — Issue60 L0完成，未进入L1

最后核对2026-11-11。[逐数组审计](F1_L0_ARRAY_AUDIT.md) · [完整L0报告](F1_PREFLIGHT.md) · [manifest锁](f1_state_manifest_lock.json) · [全部失败](F1_ERRATUM.md)。

按Yi 21:12授权，25数组的C指针/源码/干预证据已完成：当前MuJoCo3.3.3 Newton dense、island禁用配置下均为NULL；Python临时数组不是有效状态。3真实状态（两受力、一个无伙伴力）156条3秒C指针干预，physics/control/评分逐值一致，CPU重读字节摘要同样一致。无容差放宽。

12主父组24状态＋3原生参考全部真实重建；27状态保存/读回/81次完整有效状态恢复通过，5个stop/continue父组前缀一致。3原生参考不计独立接触成功组。规则变化单列报告：NULL有效长度0，其余有效数组/积分/solver/历史/RNG全量严格核验，当前配置以外不能沿用。旧失败不追溯改判。

1s候选中位0.885s、3s候选2.753s；2候选1s→执行100ms重规划中位1.877s；新进程到task/actor就绪131.667s。L0与授权reserve累计{ledger['cumulative_L0_plus_reserve_s']/3600:.3f}job小时<2h，仍计入F1 12h/原240h；未占用L1/L2/L3。GPU0已释放，无训练/自动队列。

实际runtime：数组干预5e5b3a464d9c1bf301d375f197bc9acc4cc3a653；27状态续验5f4c82c7e1ae1c8c44606e388261d6f4c93107e5；独立冷启动8bb0de396f3999abcc3c63f7a294afb1927f6372。模型未更新，BC500/BFM身份及状态SHA见manifest。

**当前只完成L0**。C-S固定BC500对照保留，但C-S planner缺格；C-P旧预测器未在F1新实现资格化。测速标O-F，不能冒称因果搜索。饱和别名只预登记。L1/L2未启动，L3须另获明确批准。能力/学习科学结论仍以Issue59及更早正式报告为准。

''';p.write_text(head+tail)
p=R/'RESEARCH_OVERVIEW.md';s=p.read_text();s=s.replace('L0目前因完整portable末态数组严格门停止，实际1秒物理重复一致不代表全部恢复资格；无新的能力或学习结论。详见F1_PREFLIGHT，L1/L2未启动，L3必须另行用户确认。','L0经逐数组C指针和156条3秒干预证据，修复了NULL指针对应Python临时数组的比较错误；27状态有效恢复资格与测速完成，未改变科学质量门。该工程通过不代表接触能力或学习成功。C-S planner仍缺格；详见F1_PREFLIGHT与F1_L0_ARRAY_AUDIT，L1/L2未启动，L3必须另行用户确认。');p.write_text(s)
(R/'P0_A_PORTABLE_ARRAY_FOLLOWUP.md').write_text('''# Issue57 P0-A portable数组残余跟踪

2026-11-11回填，证据来自Issue60 L0，见[F1_L0_ARRAY_AUDIT](F1_L0_ARRAY_AUDIT.md)、[完整续验](F1_PREFLIGHT.md)及[F1_ERRATUM](F1_ERRATUM.md)。

当前MuJoCo3.3.3、Newton/dense/island禁用、35nv配置下，原25个不稳定数组的实际C指针均NULL；另发现iM及shape0的同类NULL字段。Python以非零逻辑形状包装NULL而产生未初始化临时内存，不能算有效动力学状态。

已补齐原缺失的有效域证明：本地header/对应tag引擎分支＋真实C指针读取；两伙伴受力/一无伙伴力状态，逐数组置零/存档范围随机和联合干预，156条3秒真实续跑逐值一致。未改其他有效C状态，且按字节重读复验；27完整状态恢复与后继重复通过。

修复限F1个人独立入口：NULL→NULL不复制临时数组；保存有效域元数据；其他积分/有效数组/solver/history/RNG继续精确恢复。其他求解器、island启用、不同物理模型必须重新核实，不能全局忽略island或AR字段。

**本次关闭的是当前配置下portable数组有效范围的证据缺口，不是重新执行FB更新恢复试验。** 没有FB训练/agent.update，没有重写D2旧失败或宣称新的compiled恢复资格；P0-A梯度累加顺序与后续数值验证继续引用各自原报告。
''')
# Preserve the executable report generators as small reproducibility artifacts.
for name in ('report_arrays.py','report_complete.py','finalize_docs.py'):
 src=S/'prep'/name
 if src.exists():(R/('f1_'+name)).write_bytes(src.read_bytes())
print('finalized',oldlock['sha256'])
