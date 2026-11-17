# BFM-CTFT v1.0 — Yi批准记录

签署日期：2026-11-17。入口：[Issue #67](https://github.com/GaoE05/robot-research-bridge/issues/67)。

Yi在本次会话明确批准：Issue链接的科学合同正文及规范性执行锁；新独立stage累计40 job小时、120GiB新增输出、16,000,000全流程5ms physics ticks；最新两代完整checkpoint加预锁里程碑永久保留策略。批准连续S0→共同N预锁→U/T→固定终点评价，前提是合同资格通过；仅合同科学、物理、完整性或资源阻断停报。

权威原文逐字归档，不修改其历史DRAFT字样：

- 正文：[BFM_CONTACT_TRANSITION_FINETUNE_CONTRACT_V1.md](BFM_CONTACT_TRANSITION_FINETUNE_CONTRACT_V1.md)，Bridge源commit `7881ef4395f14af66f77e68f1b1dd8a77de3403b`，Git blob `a3ee3523e7872655ba30f42dc5eab4495b7f22fd`。
- 执行锁：[BFM_CTFT_V1_EXECUTION_LOCK.md](BFM_CTFT_V1_EXECUTION_LOCK.md)，Bridge源commit `f7bc3164b4501d96b35178dc7c555d78c29f8535`，Git blob `5620c77206fc5c55105327e46730b033d6f955df`。

签署状态：**APPROVED**。当前执行状态：**S0数学身份前置检查遇执行锁A4明确阻断，尚未正式训练**。批准仍有效；这里没有撤销批准，也没有擅自采用奖励修订。详见[原始辅助奖励单位核查](BFM_CTFT_V1_S0_REWARD_UNIT_BLOCK.md)。

独立stage名`bfm_ctft_v1`，从个人research的`17e107381d569a81761b681e09a8ddf6157fdcf2`创建隔离工作树。没有加载旧E/I、U、selector、FiLM、residual、预测头、旧Q或Adam。未启动模型前向、物理、优化；原始模型仅做SHA、config及safetensors头部核查。

本次先满足阻断证据和签署归档，不将未运行的源池、数学更新、恢复或吞吐记为通过；共同N尚未锁定。
