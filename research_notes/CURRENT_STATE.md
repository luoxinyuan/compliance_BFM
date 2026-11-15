# Current — Issue66 Contact-Goal E/I v2：初始完整物理和奖励资格通过，P1工程更新测试运行中

2026-11-15。Yi整体批准v2与独立新增6,000秒，并明确授权自行修复实现问题后继续；普通阶段不再逐项请批。原BFM/PD/normalizer/BC500与旧结果不动。

[已签合同](CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_APPROVED.md) · [签署](CONTACT_GOAL_EI_V2_SIGNOFF_20261115.md) · [P0源码/全部失败修复](CONTACT_GOAL_EI_V2_P0_PROGRESS.md) · [P1实际资格与汇总错误](CONTACT_GOAL_EI_V2_P1_REWARD_RESULTS.md) · [初始物理读数](contact_goal_ei_v2_initial_qualification.json) · [实际奖励读数](contact_goal_ei_v2_reward_qualification.json)。

源生成已修复并验证12条arm-ready真实前缀/66对历史，原源祖先错误及erratum保留，不把不同hash冒充独立性。冻结mean12/12、实际Gaussian12/12完整回合通过；含实际停止/保持/释放后4s、同态原生恢复，stop/continue两父组分叉前的控制/物理逐值一致。不是新学习收益。

G02：4个固定train父组、FOLLOW/HOLD各2，24真实物理分支，11个强Pareto有效配对来自3个有信息父组，父组等权奖励同向100%。第四HOLD-stop父组为位置/力取舍，不伪造第4有信息组或声称全域reward已正确。2400个native reward重算逐值正确。末尾numpy.bool_汇总错误已由不可变档案CPU恢复，原exit1/trace保留，无重跑物理。原文4测试父组与实现误写4有信息父组的计数纠正单列，不改80%/8pair门。

**正在运行**：P1 E臂独立scratch四个1024块（2低层/2联合）；之后I同合同、非空Adam新进程完整继续采样/更新及G07全流程成本。scratch不是正式训练，完成后丢弃其可训练状态；G05/G07未通过不得P2/P3。当前正式E/I训练0、Dev任务0、sealed未读。Source/冻结资格/候选分支不得进入on-policy PPO；每种角色有显式断言。

已记新增5,856.122700job秒（运行中E另计）；P0 3,110.639066/4,000，P1 2,135.483634/4,500，全流程136,246/950,000 ticks，修复池607.307783/800。原R2累计27,394.050957/57,600；独立新增6,000尚未用。单GPU0串行、共享盘≥100GiB、本轮7GiB。无Student、旧Expert PPO、原FB长跑或共享资产修改。旧H0/QA2负结果不重判。
