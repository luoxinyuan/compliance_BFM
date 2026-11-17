# BFM-CTFT v1 S0：恢复修复后的实际推进

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
