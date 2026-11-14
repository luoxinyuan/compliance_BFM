# R2 v1.1 释放分支恢复阻塞（sample-03）

状态：STOP_AWAIT_DECISION；按v1.1 D13只读诊断完成，无训练、无新物理重跑、未读取Dev/Sealed。旧失败和门不重判。

## 实测

runtime `861828a4ba5c7cc8c1215ebb71a6178b5cfe7c69`；job `v11-remaining-sample-03`，198.00702858秒，4321 actual/reserved physics ticks，exit 1。日志定位 qualification_v11.py:55 `Full release state restore mismatch`。已执行的首个near回合释放分支为9.805–13.800秒、800个5ms tick。冻结零U参考：root path 0.012357753m，yaw excursion 1.819340958°，loaded slip peak 0.002842213m/s、p95 0.001129999m/s，unsupported 0，joint excess 0；参考分支failures为空。不把参考自身通过当成原采样完整资格通过。

## 已证实与未证

- 已证实：同初始E/I共同网络逐值一致、优化器fresh；尚无学习更新。initial-alignment的checked_control_steps仍是初始化写盘值0，异常前未最终提交，不把它当完整采样一致性数量。
- 已证实：TrainingBranch.restore内层fingerprint断言未触发；随后更广的portable完整哈希断言触发。内层fingerprint只覆盖其列出的控制/状态，不能证明全部portable状态已还原。
- 已证实：diagnostic_frozen.portable包含所有numpy MjData数组；qualification_v11.same_state_reference对其直接tree_digest。另一方面f1_mjdata_contract.install_restore_guard明确跳过NULL指针伪数组；其canonical/same比较只在既定模型、计数器和NULL验证成立时按零有效长度比较。这两套比较口径不一致。
- 相容假设：差异来自旧L0已验证的零有效长度伪数组。尚未证实该次差异仅限于此。
- 无法从存档确定：首个差异字段及恢复后逐值RNG结果。只保存了release起点及分支后继，没有恢复后portable；RNG断言位于失败断言之后，尚未执行。不能凭字段名或内层digest通过判恢复合格。

证据：本job sidecar/log、case-0/same-state-release-start.ptz、same-state-zeroU-reference.json.gz；源码qualification_v11.py:33–59、diagnostic_frozen.portable、dual_track_runtime.TrainingBranch、rolling_snapshot.BranchState、f1_mjdata_contract.canonical/same/install_restore_guard。result.json遗留running是异常前初始化标记；实际终态以不可变job sidecar exit1为准。

## 下一项最小工作（待确认，不执行）

只复现已暴露首个near释放分支一次，保存before/after原始portable、逐字段diff、两端metadata和RNG；先不改变断言。若差异严格限于已批准L0有效范围，显式采用既有canonical/same规则并保留原始全量数据；任何有效字段差异必须修复恢复路径并重新资格。不能删除任意字段或放宽容差。该诊断无需新父组、无学习，建议最多600 job秒，实际物理须计诊断池；是否允许在D13停界后执行由Yi决定。

## 成本

该job后R2累计6906.087755秒。本次只读复核/归档保守计300秒，累计7206.087755秒；资格已用1717.248190秒（3000秒分账），无学习。共享GPU无本阶段运行进程，历史输出全部保留。
