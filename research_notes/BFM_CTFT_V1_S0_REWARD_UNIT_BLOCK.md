# BFM-CTFT v1 S0 — 原辅助奖励单位阻断

2026-11-17。Yi已批准完整v1.0；独立stage已建立。**当前为合同规定的科学口径阻断，不是行为学习阴性结果。** U/T未启动，N未锁定，没有采样、物理、Actor/FB更新或旧学习权重继承。

## 1. 已完成的原始资产身份核查

- 原始BFM safetensors SHA256：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`，3,384,968,272 bytes，现场重算一致。
- 找到原始完整TrainConfig，而非用库默认值补齐。SHA256：`96889c351a919a907f1f6e3001c5213fe5469ffc4fc2d87c0ef52e422eba7d0f`。其`agent.model`与匹配checkpoint的model/config.json逐字段一致（差异为空）。
- 原参数：actor_std=.05，stddev_clip=.3，discount=.98，reg_coeff=.05，reg_coeff_aux=.02，scale_reg=True，train_goal_ratio=.2，expert_asm_ratio=.6，relabel_ratio=.8，ortho_coef=100，seq_length=8，batch1024。
- 头部键确认Actor44、F76、B6、CPR critic76、aux critic76、discriminator10个张量；F/B/critic/aux targets齐全，robot/aux normalizer12/3个张量。这里只确认存储拓扑完整，不宣称已strict加载或可更新。
- 原训练代码只读来源commit `6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`，工作树clean。原始配置/模型/共享资产没有修改。
- 本次CPU审计执行时工作树HEAD为`7cad51880f418db5757a2057bbe202acac00d0cb`，审计脚本当时为新增未提交文件；其完整源随后归档，禁止把后来的提交冒充运行前SHA。机器证据文件记录所有读取文件SHA。

## 2. 发现的两条真实奖励路径

主环境的`rew_buf`和FB辅助critic消费的`aux_rewards`不同，不能因前者乘dt就认定后者也乘dt。

|环节|精确代码依据|实际语义|
|---|---|---|
|原环境总reward|[legged_robot_base.py:150–158](https://github.com/luoxinyuan/compliance_BFM/blob/6023c6e257fab8fe58d13d9207fb5e85c32ee9c3/humanoidverse/envs/legged_base_task/legged_robot_base.py#L150-L158)、[569–584](https://github.com/luoxinyuan/compliance_BFM/blob/6023c6e257fab8fe58d13d9207fb5e85c32ee9c3/humanoidverse/envs/legged_base_task/legged_robot_base.py#L569-L584)|reward_scales乘dt，rew_buf使用缩放后的rew；但extras['aux_rewards']保存rew_raw，明确无scale/curriculum|
|环境wrapper|[humanoidverse_isaac.py:467–500](https://github.com/luoxinyuan/compliance_BFM/blob/6023c6e257fab8fe58d13d9207fb5e85c32ee9c3/humanoidverse/agents/envs/humanoidverse_isaac.py#L467-L500)|new_info浅拷贝后直接返回，没有把aux乘dt|
|replay写入|[train.py:250–251](https://github.com/luoxinyuan/compliance_BFM/blob/6023c6e257fab8fe58d13d9207fb5e85c32ee9c3/humanoidverse/train.py#L250-L251)|原样复制next_info['aux_rewards']，与单独reward字段分开|
|辅助critic优化目标|[fb_cpr_aux/agent.py:156–175](https://github.com/luoxinyuan/compliance_BFM/blob/6023c6e257fab8fe58d13d9207fb5e85c32ee9c3/humanoidverse/agents/fb_cpr_aux/agent.py#L156-L175)|sum(aux_rewards_scaling[k]×raw_k)，之后normalizer，无dt|
|原normalizer|[nn_models.py:707–720](https://github.com/luoxinyuan/compliance_BFM/blob/6023c6e257fab8fe58d13d9207fb5e85c32ee9c3/humanoidverse/agents/nn_models.py#L707-L720)|scale-only EMA，不是时间积分器；总和后缩放不能纠正组成项间相对比例|

因此原实际辅助训练标量为`R_old = Σ c_i raw_i`，不是`dt × Σ c_i raw_i`。v1正文§3.2却指定新项`r_contact = -dt × C_contact`，同时要求原aux尺度/聚合不变。直接相加是混合口径。

## 3. 最小执行证据：实际原函数的CPU算术

审计脚本：[asset_reward_audit.py](../research/bfm_ctft_v1/asset_reward_audit.py)。AST提取并执行原`_compute_reward`、`make_transition`、实际`update`中aux加权循环，绕开Isaac依赖，不改函数计算。输入为清楚标记的标量fixture：ankle_roll平方和=.01、原系数−4；不是新物理轨迹。

|dt(s)|环境rew_buf|replay raw|辅助critic的加权和（normalizer前）|若按dt积分该项|
|---|---:|---:|---:|---:|
|.01|−.000400|.010000|−.040000|−.000400|
|.02|−.000800|.010000|−.040000|−.000800|
|.04|−.001600|.010000|−.040000|−.001600|

三例均通过精确源代码路径断言；FP32原始值见[机器证据](bfm_ctft_v1_s0_asset_reward_audit.json)。20ms时原加权和与其dt积分相差50倍。这是表示单位差，不是测得的行为退化或奖励最优性证据。

## 4. 为什么停，以及一个可审阅的局部修订建议

执行锁A4原文：**“校准若发现原reward单位不是积分dt，则先核明单位再停止报告，不能猜系数。”** 本次已确认恰好命中该前提。不能把它降级为装载、路径或数值舍入问题，也不能用固定normalizer静默修补。

建议后续局部修订（**未执行，待Yi批准**）：保留原辅助reward逐步口径与全部原系数；送入辅助critic的新接触项使用`r_contact_native = r_contact_integral / dt = -C_contact`。物理日志继续明确保存`-dt*C_contact`用于真实时间积分、跨时长比较。两臂共同采用相同规则，随后对总native标量做同一train-only冻结尺度校准。

影响必须公开：相对当前v1字面相加，该建议把新接触项在辅助目标内提高50倍（dt=.02），不能称无效应等价工程修复；所有接触成本内部比例、物理、原aux权重、CPR/aux系数保持。另一种科学选择是把整个原辅助总和乘dt后与接触积分相加；它改变“原聚合不变”条款，也需要明确批准。不能在本轮擅自二选一。

没有决定就没有正式训练。此阻断不证明新方法无效，也不能说明已通过剩余源池、严格权重加载、完整联合更新/恢复、步态量尺与吞吐门。

## 5. 资源、执行状态和下一步

- GPU0现场实测RTX PRO5000 72GB Blackwell，73,415MiB；初查0MiB/0%，并非80GB。没有申请其它GPU。
- 容器共享CPU配额210核、RAM可用约1TB；不是独占CPU承诺。共享盘初查约663GiB，满足本stage100GiB护栏。
- 审计核心CPU计时2.2225828839s；Python冷导入、Git/读取/归档等额外成本用独立stage保守wall上界统一计费，不把此2.22s冒充完整阶段成本。
- 新增真实physics ticks=0、transition=0、agent updates=0；旧模型从未在线加载。精确成本及后续归档预留见[预算快照](bfm_ctft_v1_s0_budget_snapshot.json)。
- 门状态：原始文件/SHA/配置拓扑身份已核；`A4_REWARD_UNITS=BLOCKED`；source pool、Actor/F数学、保存恢复、单卡吞吐、共同N、U/T训练及终点评价均**NOT_RUN**。
- 保留完整批准和执行锁，独立stage不关闭为失败、不挪旧预算、不改main或共享资产。等待局部奖励单位决定后，仍按原合同恢复S0其余资格，不能从旧E/I初始化。
