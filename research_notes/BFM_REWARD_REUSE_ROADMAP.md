# 当前状态补充（2026-11-07，Issue50）

Issue48/49与C0专项均已结项，不恢复旧PPO/AR扫描。Issue50已实际复用原26.7GB同权重replay，完整SHA核验后完成150000 successor、3配置×普通/weighted共6z，保留embedding与API零差对照。它们尚未成为可靠低力接触行走来源；ordinary候选在负载中推进不代表无力自主walking。当前原生goal/motion＋短程真实物理搜索已完成首条有效滚动回合，停止配对仍在运行；不能提前写成已训练的新神经Expert/Student。

实际结果和完整方法将在NATIVE_ROLLING_RESULTS/METHODS维护。下方Issue46/47的优先级与“尚未推断”措辞是历史计划，数学/接口合同仍有效。没有新建F/SF planner、重训FB、下载大型资产或自动启动Student。

# BFM reward prompting / composition / F-SF：后续最小研究记录

2026-11-05更新：Issue47已优先授权C0延迟全身参与诊断；本记录继续作为后备工具，不是其前置条件。原Issue46建议是历史阶段决策。此文是研究积压项，不启动新推断、FB训练或planner。当前主线为Issue47的C0机制接管与有条件反馈PPO；I初始化PPO是Issue46历史设置。保留0/1/2/4/8/16帧force-history独立消融，未运行。

## 当前实际接口与权重

冻结BFM SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。依据当前实际继承源码而非另一checkpoint的演示：

- `humanoidverse/agents/fb/model.py:134`：`reward_inference(next_obs,reward,weight=None)`。
- 同文件 `reward_wr_inference`、`goal_inference`、`tracking_inference`、`forward_map(obs,z,action)`。
- `humanoidverse/reward_inference.py`官方入口默认调用`reward_wr_inference`，不是普通线性推断；对真实replay后继状态重标reward。
- `humanoidverse/agents/fb/agent.py`训练中使用 `M=F(s,a,z) B(s')^T` 与 `Q`相关投影。F是以z所指定行为为条件的后继特征表示，不是任意目标下的精确物理预测器。
- 团队当前仓库保留这些接口；固定runtime继承源码与配置来源，不能混用其他权重的z。旧同权重buffer及其SHA/后继物理别名修正合同见`reward_skill_protocol.json`、`reward_stage_contract.py`、`reward_skill_infer.py`；不重新下载。

## A. 原生reward prompting

实际实现：`u_r = Σ_i w_i r(s'_i) B(s'_i)`，`z_r = project_z(u_r) = 16 u_r / ||u_r||`。默认w=1；`reward_wr_inference`取w=softmax(10r)，会强调样本高reward尾部。数值累计与batch、原normalizer及B输入必须一致；不是给一个字符串就保证技能出现。

可以重标已存在qpos/qvel/动作的状态任务：站立/高度、躯干姿态、腕相对身体位置、速度/朝向、已提供的动作成本。相同权重旧reward名称只定义目标，不等于已验证行为。

当前外部接触reward还依赖伙伴anchor/速度、Tether状态、作用点/接触历史，以及机器人动作改变后的闭环外力。这些不在原无伙伴replay/B状态合同中。可人为给buffer指定固定几何目标来算一个状态reward，但这只能产生特定几何行为候选，**不能声称零样本contact policy**，也不能预测未知伙伴意图。

下一轮最小实验（若另获授权）：固定旧可信buffer、种子、样本与推断预算，分别生成站立、小伸臂、适度躯干调整候选；同真实初态执行进入/维持/退出，再在同5/10N轻牵引正控制里对比。可证伪点：候选若不能稳定产生所需局部响应，或只有目标z相似而物理不改善，就不作为技能来源。不能无界扫描seed。

## B. 状态reward线性组合

对同一采样与同一B，普通推断在**投影前**满足 `u_(a r1+b r2)=a u_r1+b u_r2`。先归一化两个z再相加通常不等价（丢失u的模长）；softmax reward weighting也破坏上述线性关系。需保存未归一化reward embedding和组合规则，禁止混用不同数据/normalizer/权重。

arm+loco候选不是把两个独立控制器的关节动作相加；策略π(s,z)对z非线性，组合reward向量不保证技能可叠加或不抢占行走。最小正控制：预固定一个中间组合，比较原两端与组合在同真实起点的腕局部响应、速度、支撑、受力与退出；若组合牺牲walking或局部任务，就否定当前组合的实用性，不继续扫描大量权重。

## C. F / successor-feature后果预测

当前`forward_map`可得到多头F(s,a,z)。与B嵌入或reward向量内积可以形成模型内的行为/价值排序候选；数值尺度、训练discount、normalization和多头不确定性必须按实际checkpoint解释。它不是到达概率，不保证接触外力分布外也校准。

最小诊断：仅对已有成功/失败同状态候选，固定真实观测与第一BFM动作，比较F排序和随后真实短程MuJoCo的局部伸臂、root/支撑、任务回报排序。按条件/初态分组，保留反例。若排序不能超过简单原生/物理基线，不建立planner；即便排序通过，也先限定能力域，不重训F/B或宣称已实现通用预测。

## 当前决策边界

如果本轮有“局部解真实有效、reward认可、但I附近探索未覆盖它”的一致证据，下一阶段优先做A的小型来源实验，再决定B/C是否值得。需要有限物理证据支持，不因一次离线z距离或零命中就声称整个BFM无能力。本轮不改变Direct动作空间，也不自动启动这些项目。

实际checkpoint配置进一步核对：模型类FBcprAuxModel，z_dim256/norm_z=true/amp=false。B输入`state, privileged_state`；F输入`state, privileged_state, last_action, history_actor`，双forward heads；actor只用`state,last_action,history_actor`。其中privileged_state由原`max_local_self`身体状态构建，不是我们当前新增的伙伴误差/接触槽位。原FB训练discount=.98，不能与当前高层PPO .99直接当同一时间尺度。原FB model/reward入口与共享6023c6e基线无diff。本轮未执行F forward、未把F迁至GPU；后续若使用必须遵守原normalizer/设备合同并预算化。

外部原作者来源已于本阶段核对：[LeCAR-Lab/BFM-Zero README](https://github.com/LeCAR-Lab/BFM-Zero)、[FB model实现](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/humanoidverse/agents/fb/model.py)、[reward入口](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/humanoidverse/reward_inference.py)。公开main与当前冻结运行版本必须区分；本记录的输入维度/权重/执行结论以本地固定checkpoint及runtime为准，未升级到公开main。


方法讨论补充：原生候选可作为时间连贯探索的起点，候选来源检验与当前接触策略学习分工见[CONTACT_LEARNING_RATIONALE第8节](CONTACT_LEARNING_RATIONALE.md)。不因候选可推断就宣称覆盖当前全部接触/过渡状态；Issue46没有证明std=.008完全到不了有利单步方向。本记录不启动新的候选推断、planner或Student训练。
