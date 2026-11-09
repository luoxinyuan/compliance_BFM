# CURRENT_STATE — RETENTION新条件验证已结项

最后核对：2026-11-09。**26回合全部结束，无训练、Teacher查询或后续队列。**

[13条件对照与结论](RETENTION_VALIDATION_RESULTS.md) · [分阶段XYZ/滑移/yaw表](RETENTION_VALIDATION_PHASES.md) · [锁定协议](retention_validation_protocol.json)。

**结论：收益限于开发条件，5403b7不晋级，BC500维持基线。** 四条新长距均未达到误差与力共同实质收益；61cm触发相对BC滑移保护，移动源57cm末yaw约−24.22°，较开发约−15.15°显著放大。34cm中距有明确收益（13.959→7.154cm、8.347→5.785N），停止/局部相对指标在容差内，但arm-ready起点两模型共有严重跟随失败，不能因相对不劣称成功。

13组配对完整初态一致；6组停止/继续前缀实际物理/动作逐值一致。精确模型SHA沿用下方身份，评价runtime `3a59503b2f4ef9255830a441a8fdddcac030c051`。私有stage `retention-newconditions-20261109`；累计保守job 1271.15s，产物约0.347GiB，均低于1800s/2GiB。未新增视频或重复实验。

**下一决策：保留开发正结果和共同失败域，等待用户决定；本轮不自动启动第二seed或任何新训练。** 历史RETENTION/INTERACTIVE_INCREMENT结论完整保留如下。

---

# CURRENT_STATE — A/B/C有界续轮已结项

最后核对：2026-11-09。**全部训练与评价结束，无队列，不自动开启下一轮。** [完整报告](RETENTION_RESULTS.md) · [只读审计](RETENTION_CODE_AUDIT.md) · [分段物理表](RETENTION_PHASE_TABLES.md) · [Issue53](https://github.com/GaoE05/robot-research-bridge/issues/53) · [Project Master](https://github.com/GaoE05/robot-research-bridge/issues/18)。

## 当前结论

- 有限域Teacher→ideal-force Student最小闭环仍成立。干净基线仍是 **BC500**，不是原S4000或历史S0。34条修复后交互标签的原形态仍未显示整体增量，原结项保留，见[INTERACTIVE_INCREMENT_RESULTS](INTERACTIVE_INCREMENT_RESULTS.md)。Issue52独立Expert PPO未重启。
- **A0**：固定相同初态/随机流重评BC500、5401b4/b7、5402b4，旧指标复现，5402无接触检查补齐。batch4长距收益真实，主要来自保持段减轻侧向牵拉；移动滑移增加。b4/b7第一分叉支持微小命令差异经状态/支撑反馈放大，约6.585s首次支撑不同，非只在最后停止时失配；没有证明唯一critic或梯度冲突根因。
- **A1**：新seed5403从相同BC500 actor/normalizer、新critic/Adam，原PPO数值/物理合同，7×768=5376决策。每批独立固定评价，8次训练RNG/状态隔离通过。事前两层规则选**batch7**：72cm **8.047cm/5.346N**（BC11.316/7.485）；8cm2.765/2.326（BC2.795/2.347）；停止9.332/5.833（BC9.384/5.851）。
- 最佳检查点保留收益成立；**持续稳定改善未证明**：batch5曾回退10.698/7.052，batch1因滑移保护排除。末轮仍有移动滑移和约−15°偏航代价。仅一个新seed开发验证，不是新封存泛化或全任务替换BC。旧5401/5402与失败保持。
- **B**：712候选不是712独立标签。20shadow状态/180 F-物理分数配对，12接触状态仅来自4独立回合，低于约10组门槛；逐候选首z/action需重建而非显式保存。本轮不补算F、不拟合校准器、不新增查询，校准主张暂停，非整个FB方法否定。
- **C**：物理callback跨reset、SHA/normalizer、混合索引核实。确认快照不全面恢复NumPy任务RNG，旧开发评价复用Task后重置训练回合；明确使用边界，未修复旧实现。未发现需撤回已复现BC/b4物理结果的新缺陷。完整原入口清单、最小复现与影响范围见审计。

## 精确身份与位置

- BC500：`a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`；原stage `interactive-contact-pilot-20261108/runs/bc-fit-01/old_data_only-500.pt`。
- 新5403b7：`c1b9f7c04b4c0154b8644929bbeebbd64f55a227b199e6f5c7ed0dc116257f76`；新stage `ppo-retention-audit-20261109/runs/A1-train-5403/candidate-7.pt`。只是本轮开发选定候选，BC不覆盖。
- BFM：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。A0 runtime `48e756fadfeffcb5cf447e21d114344044291cf2`；同状态诊断/5403 runtime **`8ef38c44ce71b910d11af10e3abe44b2bff9283e`**；报告提交不冒充运行源码。
- 旧模型完整SHA、所有新检查点/阶段成本、逐批学习与评价见[retention_artifacts/evidence.json](retention_artifacts/evidence.json)。服务器根 `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/`；各stage ledger/jobs与失败保留。
- 个人审查分支 `luoxinyuan/compliance_BFM:gaoe05/research`，本轮最终文档SHA见Issue53最新结项链接；main未改。
- 本轮不重复生成视频，新增两张由真实轨迹/评价生成的科研图已核查；历史关键1×视频仍在Mac `~/Projects/compliance-bfm-diagnostics/videos/interactive-increment/`（2条MP4＋INDEX）。

## 成本、边界与接力

A0 768.16s、A1 3543.73s、B14.08s、C6.05s实测job；另250s明确CPU分析预留，总4582.02s。A1双计父等待与评价子job，保守。输出约.759GiB，5376Student RL，0新Teacher搜索/0BC/0Expert RL。所有分项低于用户上限。只用获准NVIDIA卡；启动实时核验CPU/RAM/I/O；不据NVML名称臆断型号。

此前建议的新条件评价已完成，结论见本文件顶部；不再重复发车。不把回合内高度相关候选当独立校准样本。若修复旧RNG/评价生命周期需独立回归小轮，带归档物理复算。

新会话先读RESEARCH_OVERVIEW、本文件、RETENTION_RESULTS及Issue53；需要启动时重新核验实际资源和身份，不沿用旧PID/空闲快照。保持BFM/PD/main/SoftSONIC/他人任务不动、不真机。ideal-force只是传感假设；既有Library 2026-10-28文档是历史快照，不覆盖当前GitHub事实。
