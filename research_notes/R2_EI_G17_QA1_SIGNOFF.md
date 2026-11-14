# G17-QA1 批准归档

研究批准人：Yi。批准日期：2026-11-15（当前会话）。

用户明确授权原文：

> 我批准G17-QA1 修订，归档后按修订执行继续推进吧，不用再停下来等我批准

采用 Issue #63 最新修订源文件 `contracts/R2_EI_G17_Q_DIAGNOSTIC_AMENDMENT_V1.md`，Bridge Git blob `4a50e971c6b14f2c1fa2ce95a03885dbd936c0ab`，正文 SHA256 `726d2baf5160b0e973a8ef15729230385c0727477804d69248470718cea53dc9`。正文按来源原字节归档，[原文](R2_EI_G17_Q_DIAGNOSTIC_AMENDMENT_V1.md)中的 REVIEW ONLY 是审批前历史状态，本附件记录本次新批准，不回写源文或旧 G17 FAIL。

批准范围：从同一 E Q56/Adam/target 和固定3500真实 replay 出发，U/C各到总512组（各新增456），128/256为诊断节点，不按中途结果挑模型。batch64、lr、EMA=.01、loss、真实action/next/discount不变；每组父组×真实物理phase槽位一一匹配。I/Actor/U/Goal/原BFM/normalizer/PD冻结。

执行前完成目标/后续策略/真实phase只读审计、完整schedule fixture及拒绝测试、源SHA与新前瞻四父组锁。旧40分支仅retrospective；512达到修订要求才允许一次最多20条新train-only物理分支。前瞻通过也不自动启动Actor/I/32k或其他训练。

专用子额度≤3600job秒/1GiB/18000全流程5ms ticks，包含在原R2 57600秒/12GiB与项目240h内。继承已结算15241.192158秒，不重置；本stage新输出独立统计且也并入R2根清单。实际/预留counter均存，预算tick=actual+尚未执行的预留=reserved，不重复加同一个已执行tick。所有冷启动、prepare、IO、失败、评价及发布都入账，单GPU0串行、shared free≥100GiB。

原70%/四独立父组、非平凡差异门、权限、物理/安全、旧结项与erratum不变。普通实现自主处理；修订规定的科学失败/数据资格不足/正确性/预算边界仍按原条款停，不以本批准宣称已经通过任何新资格。

当前状态：**APPROVED_BY_YI，准备与锁定中；尚未新增Q优化或新物理。**
