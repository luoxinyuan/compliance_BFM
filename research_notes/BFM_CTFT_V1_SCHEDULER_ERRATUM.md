# BFM-CTFT v1 — S0轮次scheduler计数独立勘误

2026-11-18，Issue #67。此勘误仅覆盖一次性S0存盘资格，不改旧科学结论。

## 已证实缺陷

`s0-production-reference-01/store/round-10252`实际保存了10252行、完整Task/模型/Adam/replay/RNG，但最后12条续采没有同步累加Python scheduler的最后回合计数：回合controls之和10240；Task的实际live_control61，而最后回合记录49。恢复包的实际物理/学习内容正确，新进程12控制和下一更新原本逐值一致；**完整调度计数资格不通过**。

## 最小复现与修复

`s0-scheduler-metadata-audit-01/result.json`：对旧真实包运行新writer守恒检查，确定性拒绝；只修最后回合controls/complete后通过。没有新物理、模型更新或容差修改。

新writer在commit前核对：updates=learner step；各回合controls之和=collected；回合数=source_index；当前回合controls/spec/complete分别等于Task live_control/live_spec/done。旧包与依赖tensor全部保持原址、原bytes，不覆盖或物理移动。

`s0-scheduler-reference-01/store/round-10252`作为独立修正包，仅修上述计数；非scheduler内容完整digest相同，state SHA `9d8968ee96dfe6c1549c12c97ca1031853b36f24a68b7d9dcaa724a36d1d882d`。共享原不可变tensor块使新增payload=0，这不是零成本完整checkpoint。

## 决定性回归

runtime44e7b63：参考与独立恢复各实际执行620控制，完成当前回合、reset至真实下一源并继续12控制，再执行1组完整原生联合更新。完整rows/5ms physics/Task/scheduler/抽样/RNG/模型/五Adam/指标逐值相同，`comparison.json`为PASS/different=[]。runtime1a45626带资源计量再独立重复一次亦逐值通过。

旧10252包逻辑标为“learner/Task可复现、scheduler metadata未资格”，不得作为正式全状态调度恢复证据。旧失败保留；已验证新包取代其后续资格用途。正式U/T尚未开始，未污染任何正式训练或Dev。
