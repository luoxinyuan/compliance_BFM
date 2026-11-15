# Current — Issue66 E/I v2 P0：Yi已批准修复源实现后继续，正在资格

2026-11-15。整体v2及独立6,000job秒批准保持；首轮P0发现arm-ready源不同ID实为同一轨迹，完整证据/失败见[P0原始报告](CONTACT_GOAL_EI_V2_P0_SOURCE_RESULTS.md)和[独立erratum](CONTACT_GOAL_EI_V2_SOURCE_ERRATUM_01.md)。未开始正式训练。

Yi随后明确：“这种你自己的实现问题可以自己修正后继续”。[追加授权与精确源修复](CONTACT_GOAL_EI_V2_SOURCE_FIX_AUTHORIZATION_01.md)已归档：仅新stage arm5准备轨迹实际使用各父组已锁pulse参数，旧共享Task/原BFM/PD不动；新增实际source祖先字段，不靠无效hash证明独立。先对现有12个arm源做source-only真实资格，train6/dev6固定，不执行Dev任务、不读sealed；失败不跳过。通过后按原证据门连续推进。

原reward、Actor/FiLM/联合PPO合同、硬界、分组判读、32k/臂及总41,000未来job秒/P0 4,000秒帽不变。source修正不是伪称旧轨迹等价的路径修复，而是这次新增明确授权内的实现纠正。所有旧H0正结果与负例保留。

当前尚无新E/I学习checkpoint，P1/P2/P3均未启动；概率/奖励/新策略恢复/新接口等尚未实测，不借旧PASS发车。资源/成本以独立stage账本和每job sidecar为准，单GPU0串行/共享free≥100GiB。
