import json,subprocess,hashlib,time,statistics
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/source-coverage-m1-20261109';N=S/'repo/research_notes';O=N/'source_coverage_m1_artifacts'
e=json.loads((O/'m1-evidence.json').read_text());seg=json.loads((O/'segments.json').read_text());ledger=json.loads((O/'resource-ledger.json').read_text());cov=json.loads((O/'coverage5403.json').read_text())
def f(x):return '—' if x is None else f'{x:.3f}'
lines=['# M1 联合源状态覆盖与 Teacher 资格：结项','', '2026-11-09。仅采集和验证；BC500 保持冻结，5403b7 不晋级；训练、RL、F 拟合均为 0。','',f"## 预声明结论：{e['decision']}",'',f"12 个配对评价单元中 {e['usable']} 个通过全部门槛（standing / arm / moving：{e['by_source']}），未满足 ≥8/12 且 arm、moving 各至少 1 个，按协议停止，不扩采、不进入 M2/M3。",'', '**范围限定**：这指当前原七池、冷启动提议和有限 2 秒接管方案未提供本批跨源任务的合格反馈。不是证明冻结 BFM 永远不可行，也不撤销此前完整滚动 Teacher 的有限域正结果。实际后果中包含交回 BC500 后的段落，不能把整个失败全部归给 Teacher。','', '长距用户修订门槛在查询前锁定：误差≤10cm、实际力 RMS≤6.5N；中距≤8cm/7N，局部≤4cm/4N，饱和≤15%、有载荷足滑移峰值≤0.45m/s、释放末绝对 yaw≤20°，停止保持≤8cm/6.5N；对 BC500 误差/力各不得恶化超过5%。7.42/5.47历史结果仅说明这两项有余量，误差阈值比测量高34.8%，力高18.8%，不等于全部资格通过。','', '## 实际联合覆盖与独立性','', '5403 归档逐条重建并与全部 5376 个实际 family 标签一致：50完整回合+1部分回合；standing 49完整+1部分，arm仅1回合92决策，moving0。standing接管固定4.0s；arm固定5.6s且仅16–30cm域。旧课程概率不能代替经历。源×请求×阶段的真实回合/决策和可见失败见 [账本](source_coverage_m1_artifacts/coverage5403-phase-episodes.json)。训练归档没有全量足滑移/yaw，不能补造完整质量成功率。','', '本轮12单元来自3条原生共同前缀族，不能记作12独立组，更不能把91查询/637候选当独立组。实际train3/dev0/sealed0；目标36/8/20未达，前置资格失败后不扩采。3条精确旧课程补充控制不加入12单元分母或用于找补。','', '源由真实 BFM 执行、身体/支撑筛选：standing4.0s；arm5.0s（相对肩前伸约3.19cm）；moving5.0s（vx约0.263、vy约0.184m/s，一脚卸载）。moving带侧向速度，不能按名称视作合格直行。goal源来自离线IK目标再由BFM真实执行；motion来源有已存档M10手臂/腰参考修正，无 I/C0 actor 在线输出。','', '## 全接触窗口实际结果','', '单次选定24命令执行后交回BC，或滚动20个首命令后交回BC；表格是完整接触窗口，滑移/末yaw另见JSON。每格误差cm/力N。','', '| 源/请求 | BC500 | 原生继续 | 选定24命令→BC | 2s滚动→BC | 可用 |','|---|---:|---:|---:|---:|---|']
for c in e['cells']:
 def pair(a):return '—' if not a else f"{a['error_cm']:.2f}/{a['force_rms']:.2f}"
 arms=c.get('arms',{});a=arms.get('actual_selected_plan24_then_BC_aftermath',{});b=arms.get('actual_Teacher20_firstcommands_then_BC_aftermath',{})
 lines.append(f"| {c['cell']['id']} | {pair(c.get('BC'))} | {pair(c.get('native'))} | {pair(a.get('metrics'))} | {pair(b.get('metrics'))} | {'是' if c['usable'] else '否'} |")
lines+=['', '逐项失败原因、饱和、滑移、yaw和保持窗口见 [完整证据](source_coverage_m1_artifacts/m1-evidence.json)。未施力原生继续结果也在该文件，不能用外力为零替代稳定性检查。','', '## 2秒实际 Teacher 窗口与后继必须分开','', '| 单元 | BC同窗 cm/N | Teacher同窗 cm/N | 交回BC后接触 cm/N |','|---|---:|---:|---:|']
for w in seg['rolling_windows']:
 def pair(m):return f"{f(m['error_cm'])}/{f(m['force_rms'])}"
 lines.append(f"| {w['cell']} | {pair(w['BC_same_window_metrics'])} | {pair(w['Teacher_window_metrics'])} | {pair(w['BC_after_Teacher_metrics'])} |")
lines+=['', '这些是分别真实执行的同初态分支，不能由单个短窗改善推出整条轨迹可恢复。Teacher实际首命令80条保存为待资格标签；本轮严格门槛下没有自动升级为合格正向模仿标签。候选预测分支全部为排序/能力负例资产，不是 PPO 数据。','', '## 预测—实际与数据资格','', '每次查询原七池不变，当前伙伴位置/速度作恒速短期预测，未读取未来停止或轨迹。候选低分/高分都保留。早期伙伴尚在加速，恒速预测和后续真实需求不同；相同24命令真实执行与预测的误差、力、anchor偏差以及首个q_PD一致性见 [分段审查](source_coverage_m1_artifacts/segments.json)。不能把预测2.4s低误差当完整任务成功。','', '每查询保存完整物理/solver/历史快照，current obs/previous-z，7×24×256计划，目标z与首子步插值z、低层action、完整原F及评分embedding；所有预测物理轨迹/边界也保存。每文件SHA在证据清单。3类源各取query001新进程按真实前缀重放，integration、q/qdot、solver、BFM/动作历史、privileged/restricted和last-z逐值相同，digest一致；这是3个代表快照的验证，不泛化为每个文件均完整物理重跑。','', 'F 使用 first interpolated action 对机器人状态reward embedding评分；物理排名是24命令、变化z、外部接触的有限预测结果。这是不同合同，配对存档可供后续研究，不能把F称为完整接触return。无F拟合，无额外FB候选，无封存调参。','', '## 精确旧课程正控制','', '| 条件 | BC cm/N | 原生 cm/N | 选定计划→BC cm/N |','|---|---:|---:|---:|']
rs=e['exact_original_controls']
for cell in sorted(set(r['cell'] for r in rs)-{'native_no_contact'}):
 vals=[]
 for role in ['actual_Student_frozen_BC500','actual_native_continuation_contact_control','actual_selected_plan24_then_BC_aftermath']:
  r=next(r for r in rs if r['cell']==cell and r['identity']==role);a=r['all_quality'];vals.append(f"{a['error_cm']:.3f}/{a['force_rms']:.3f}")
 lines.append('| '+cell+' | '+' | '.join(vals)+' |')
lines+=['', '8cm/72cm standing复现既有BC指标；arm24cm保留原5.6s源/6.2s onset的域内控制仍不等于成功。24cm未被用户绝对门槛明确分箱，只报告物理结果，不新设通过标准。','', '## 身份、资源与安全','', f"BC500 SHA `a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`；BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。主采集runtime `{e['assets'][0]['runtime']}`；辅助runtime `a3f06b8c74c8e6147eab638d099101526b347fb0`。原权重/PD/main未修改。",'',f"完整查询 {ledger['completed_queries']}，候选预测 {ledger['original7_candidate_rows']}；3条补充正控制包含在内。job {ledger['supervised_job_seconds']:.2f}s（失败准备与复放均计），runs输出 {ledger['output_bytes']/2**30:.3f}GiB；分析/文档耗时另加最终账本。查询≤96，未扩至400。训练/RL/F拟合/额外FB均0。",'', '详细分类物理步数、F调用、reset/源前缀及实际控制步数见证据counts；失败/成功job侧账见 [资源账本](source_coverage_m1_artifacts/resource-ledger.json)。成本/合格正标签不能以0作分母虚报；本轮合格标签0，成本/原始待资格首命令仅为采集成本，不是有效标签成本。','', '新实现准备期支撑布尔量误用为牛顿阈值，0查询/0标签，独立 [erratum](SOURCE_COVERAGE_M1_ERRATUM.md) 保留；随后仅加入只读有载荷滑移记录，未改物理。所有旧结果不覆盖。','', '## 接力与发布','', '完成即停止。BC500继续冻结；不启动M2/M3/PPO/Student/F拟合。本轮结果只支持在当前方案中识别生成/预测/恢复边界，下一轮任何改变需新的研究决策。','', '公开GitHub创建Issue被自动审批拒绝；本轮报告、代码及私有资产已保留，没有声称已更新公开Issue/Master或推送个人分支。拒绝理由为具体内部实验内容的公开目的地授权未获认可；未改用其他通道绕过。']
(N/'SOURCE_COVERAGE_M1_RESULTS.md').write_text('\n'.join(lines)+'\n')
p=N/'CURRENT_STATE.md';old=p.read_text();marker='# CURRENT_STATE — RETENTION新条件验证已结项';old=old[old.index(marker):] if marker in old else old
head=f'''# CURRENT_STATE — M1联合源覆盖资格已结项

2026-11-09。**仅采集/验证完成，未训练，未进入M2/M3。BC500仍是冻结基线，5403b7不晋级。**

[完整M1报告](SOURCE_COVERAGE_M1_RESULTS.md) · [配对资产与质量门槛](source_coverage_m1_artifacts/m1-evidence.json) · [真实分段/预测差距](source_coverage_m1_artifacts/segments.json) · [覆盖账本](source_coverage_m1_artifacts/coverage5403-phase-episodes.json)。

预声明结论：**{e['decision']}**。长距门槛在查询前改为10cm/6.5N。12单元通过{e['usable']}，未达到8/12及arm/moving各1；不扩采。3个真实原生共同前缀族，不把查询/相邻状态冒充64独立组；train3/dev0/sealed0。本轮有限2秒Teacher后交回BC，其短窗和后继分开评价，不能把整个失败全部归给Teacher，更不否定历史完整滚动正结果。

完成{ledger['completed_queries']}原七池查询/{ledger['original7_candidate_rows']}候选分支、4个实际2秒窗口，实际Teacher首命令80条但无自动批准的正向模仿标签。3类源代表快照完整新进程重放逐值一致。原F/首动作与24命令物理评分合同差异可从完整存档检验。

5403真实5376决策：arm仅1回合92决策、moving0；standing接管固定4.0s。旧课程未覆盖当前联合源需求。历史日志不足以重建所有训练回合滑移/yaw，完整质量成功率未知。

模型BC500 `a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`，冻结BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。实际runtime、路径、失败准备和成本见报告/账本；private stage `source-coverage-m1-20261109`。公开Issue创建被自动审批拒绝，尚未发布或push，不以私有提交冒称GitHub同步。

后续等待新的科研决策；不自动启动训练。以下历史结论原样保留。

---

'''
p.write_text(head+old)
print('report and current state written')
