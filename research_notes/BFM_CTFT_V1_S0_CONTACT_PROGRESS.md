# BFM-CTFT v1 S0：恢复修复后的实际推进

## 奖励修订：批准、落地与待执行事项（2026-11-17补正）

Yi已经批准奖励单位科学修订，见[Issue67批准记录](https://github.com/GaoE05/robot-research-bridge/issues/67#issuecomment-6940383957)及[A1签署附件](BFM_CTFT_V1_APPROVED_AMENDMENT_A1.md)。本报告此前没有显式区分批准状态与执行进度，容易将“尚未校准”误读为“尚待批准”；本节补正表达，不改变历史结果或科学合同。

|事项|批准状态|实际执行状态|
|---|---|---|
|A1接触奖励单位修订|已批准|已实现；实际冻结接触probe已按该口径计算和存档|
|两臂共同、固定的总aux尺度校准|已授权执行，无需再批准|尚未完成train-only数值校准和锁定；不是等待Yi决定|
|使用上述完整目标的正式U/T训练|已授权，须先通过原S0资格|尚未启动；不能把代码或probe通过写成正式训练完成|

实际单位接口在[reward.py](../research/bfm_ctft_v1/reward.py)：送aux路径的接触项为 `-C_contact`，物理时间积分另存 `-dt*C_contact`。20ms下前者是后者的50倍，已明确作为科学修订归档，不声称与原字面相加等价。原aux内部权重、接触内部比例及其它科学门不变。

本次对runtime `5b940c69880d27f2d32bf7572c8fae04a2d96fdd` 的 `s0-contact-interface-01` 两份既有实际执行存档只读复核：有力/无力各525条，合计1050条控制记录，全部标记A1单位合同；逐条 `native_reward*0.02 == integrated_reward`，最大绝对差0。实际非零示例为 `-0.0013118149586922691` 与 `-0.000026236299173845383`。未重跑物理或更新模型。此处证明实际日志的单位执行，**不证明已经完成含接触总aux目标的learner训练**。

工程learner恢复使用g0来源及 `aux_scale=1` 占位值，只用于工程逐值恢复；该值不是共同尺度校准结果，不得直接写入正式两臂合同。共同尺度将按已批准的train-only口径计算、冻结并归档，仍属原授权内后续工作。

2026-11-17。此为S0中间记录，不是正式U/T或S0完整通过。

## 存档源注释

`b399451`，`s0-source-annotation-01`只读检查12个原生祖先，34.084098s核心时间，新物理/更新均0。按预先速度/支撑定义，在8个祖先发现原生推进或减速候选事件，其余4个没有满足本次规则的事件，不删除或伪造。材料点local→world几何重建最大误差≤5.551116e-16m。实际接触材料点瞬时速度p95与固定材料点累计滑移分开：后者仍UNMEASURED，不能据几何恒等式宣布滑移门通过。源分组和训练入选均未在此审计中进行。

## 冻结接触链

`892b0e9cacea59dffb96618a5c7802d7d1dab0de`，`s0-contact-interface-01`，同一真实3s原生standing前缀后：有力与无力两支均完整525个20ms控制，分别2100个5ms后缀tick；两支合计只是1个物理父组。全job实际6001ticks，含初始化/reset/prefix，不能只计4200后缀ticks。无新agent.update，不继承工程probe权重。

原K80/D8/10N和单腕作用点/COM力矩不变；伙伴平滑加减速、巡航.12m/s、总位移.60m、保持1s、release后4s。位置/速度边界fixture先行通过；协议在物理评价前写入独立输出。force最大10.000000000000002N（原公式浮点舍入）；no_force全程实际力0。两支均未触原机械、姿态或55ms无支撑边界。

真实Actor46与当前critic9分开存档；因果force history来自实际前缀/实际控制，Actor字段无anchor、伙伴速度、raw force、active或时钟。此冻结probe实际仍由原native mean Actor执行；先前零新增列等价资格不能与本次probe合并冒充已经训练的新Actor。每支保存4个完整物理/历史快照，尚未做本批contact新进程恢复。原aux逐项和A1新contact reward分别保留；未将旧ComplianceEnv聚合reward冒充本轮训练目标。终止与截断分开，release不作为终止。

## 尚未完成

learner严格新进程下一更新恢复已通过，见[工程报告](BFM_CTFT_V1_RESTORE_ENGINEERING_RESULTS.md)。本次没有完成随机采样执行资格、接触Task与learner联合恢复、材料点累计滑移、共同aux尺度、独立source分组或全链吞吐/N选择。U/T正式训练仍0。

预算包含全部失败与工程，完整最新数值见`bfm_ctft_v1_restore_continuation_cost.json`；sidecar核心与连续工程成本不可再相加。旧A2与后续编译失败未覆盖。
