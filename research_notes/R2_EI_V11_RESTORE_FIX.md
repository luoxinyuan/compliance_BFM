# v1.1恢复比较修复：实际证据与授权

Yi在2026-11-14明确批准一次复现并授权此类工程问题自主修复、继续推进，不放宽科学门。

诊断runtime 94ccf44a38f1e2bc772a2fc1f1f9235d9fbebdac，作业v11-restore-diagnostic-04。完整before/after及两端metadata已保存；24个raw差异全部来自当前底层NULL数组，已有L0 canonical比较valid_diff为空，完整RNG逐值一致。包括iM；排除规则采用既有metadata/inactive验证后的全部NULL集合，不是根据本次差异名单临时删字段。模型版本、Newton/dense、无island计数及底层指针验证均通过。

修复只将资格入口原始容量hash断言改为既有L0有效长度逐值断言；原始数组继续全量保存，真实有效数组/身体/BFM/history/Task/RNG无容差。任一有效字段差异仍停止。原sample-03与diagnostic-04失败不重判；新运行独立目录、原父组/seed/噪声/物理门不变。

证据资产：case-0/restore-before-after.ptz、restore-audit.json、same-state-release-start.ptz及原完整同态参考；原始差异24项、有效差异0、rng_equal=true。这验证该同态分支恢复，不代替未来每轮checkpoint新进程恢复资格。

CPU归档回归：原真实before/after有效比较PASS；integration和qvel各注入1e-12改动均被拒绝；nisland配置改为1触发fail-closed。此为故障注入逻辑检查，不冒充新物理结果。诊断04耗198.516510秒/4321物理tick，全部计费，未更新任何学习参数。
