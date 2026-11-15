# E/I v2 P0 — Source correction authorized by Yi

2026-11-15。承接P0源资格实测失败，Yi追加明确授权：

> 这种你自己的实现问题可以自己修正后继续

该追加授权允许自行修正本轮源实现与祖先映射并重过资格，原v2其余合同/预算/判据不变，不再因这一已知实现遗漏等待签署。不抹去[原始P0阻断](CONTACT_GOAL_EI_V2_P0_SOURCE_RESULTS.md)及[erratum](CONTACT_GOAL_EI_V2_SOURCE_ERRATUM_01.md)。后者是修复前已发生事实，不是最终阶段结项。

## 精确最小修复

新stage私有`ei_v2_sources`仅替换新v2的goal_arm5源分支，旧共享Task/资产不动。保留原smoothstep((t−3)/1.5)与原arm5端点，使用**各原父组已经锁定的**pulse_fraction/pulse_duration构造实际准备历史：

`alpha(t)=arm_fraction * smoothstep(clamp((t−3)/1.5,0,1)) * (1−pulse(t))`

`pulse(t)=pulse_fraction*sin(pi*(t−3.05)/pulse_duration)^2`，只在0<(t−3.05)/pulse_duration<1内非零。3.05为旧standing准备分支已有常数，未扫描新幅度/时长；不输入伙伴/未来/任务标签，不把它作为部署force-to-skill控制。standing执行分支保持原样。该公式是经本次用户授权的源实现修正，**不是宣称与旧错误arm轨迹数学等价的§9路径修复**。

原parent_id/source_parent_id保留为来源追踪；另存`execution_source_parent_id`，只由实际参与执行的source/version、幅度/持续时间、handoff构成。无效phase字段不得用于证明独立性。先锁12个现有arm源（train6/dev6）完整source-only资格，检查真实控制/物理历史，而非换hash继续。dev源资格不执行Goal/伙伴任务、不读取结果选源；sealed案例仍未读。任何源失败保留，不跳过、不挑表现好的替代源。

修复资格通过后按原P0→P1等门连续推进，无需逐段审批。正式RL仍未启动。此次修复准备预算保守600job秒记P0；资格另逐实测计入P0，原41,000未来总帽、P0 4,000秒及新6,000额度不扩大。
