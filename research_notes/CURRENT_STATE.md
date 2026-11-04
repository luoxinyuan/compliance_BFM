# CURRENT_STATE — 新会话先读

最后核对：2026-11-04。**M41启动因果诊断已完成并停止：41个冻结物理回合，新增RL/监督更新均0。没有下一轮训练排队。**

入口：[本轮接管与训练决策](M41_STARTUP_CAUSAL_RESULTS.md) · [原Transition A/B结项](TRANSITION_RESULTS.md) · [Issue41](https://github.com/GaoE05/robot-research-bridge/issues/41) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。

## 研究主线与可靠基线

总目标Contact-Conditioned Whole-Body Motion Transitions：共享策略根据真实接触及身体状态组织全身行为。当前是特权Direct latent Teacher获取；未来Student只能使用部署可得观测。旧有限域直行/停止Teacher→ideal-force Student、小幅双向转腰成果保留，不宣称已经实现general、迈步转身或真机精确力传感。完整原则见RESEARCH_OVERVIEW。

不可覆盖的模型：
- BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，runs/stop-sensing-20261103/B1-train-02/job/best.pt。
- S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`，runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt。
- D2 `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`，runs/m38c-history-20261103/C-train-05/job/D2.pt。
- I：M39-A best64 `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`，runs/m39a-demand-20261104/B-ppo-03/job/best.pt。
- A32/B32及A/B16/48/64身份完整见m41_causal_artifacts/identities.json；均保留，不按不同任务拼接。

runs前缀为`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/`。长期审查分支`gaoe05/research`；最新文档commit见Issue41精确链接。

## 原Transition课程结果（不改写）

A共同站立与B真实源状态接管各49152RL决策；开发best均iteration0，未获得优于初始化的新模型。选定best SHA`e1e33b521e829647e7aaac3a04b7e019f0ed99eb94883371f4437ae3d6da1fcb`与I权重相同，元数据不同。旧48封存回合含重复初始化身份，不能计为新学习成功。A32/A64、B32发生预定回滚，并非连续64批优化。

A32/B32启动误差48.03/50.03cm，I为5.34cm；B48已有行走状态5.37cm/3.85N→4.84cm/3.36N，但启动退化14.23cm。保留这项取舍，不说所有能力消失。

## 本轮真实因果证据

独立runtime`9e3771a8db036763152895748564bb1b302d597b`，原训练runtime`a8afaff087f535124d9e1b28be7ada57caa10bdf`。I/A32/B32新运行完整qpos/qvel/force与旧保存轨迹逐值差0。28接管/单命令分支接管前全记录状态/solver/历史/prefix哈希逐项一致；使用同一live环境继续，没有qpos-only重置。

- A32→I在4.5s接管恢复5.04cm/3.20N；在6s接管仍50.52cm/4.62N。
- B32→I在5.5s接管仍50.08cm/4.62N；**I→B32在同一5.5s时刻却7.75cm/3.70N**。
- I先执行到6.5s，再交A32/B32，均能行走，5.23/5.34cm；失败模型并非完全失去walking能力。
- 5.5s单个100ms命令替换不能救回失败，也不能在I轨迹复制严重失败；持续状态/动作历史演化更重要。
- 同状态完整5子步q_PD查询差通常约0.03–0.2°，但自身闭环轨迹差可以大很多；不能继续只用单帧MSE判断物理保持。
- 初始关节持续分叉约5s，cap饱和约6s。现有5/10/20N结果已复用，本轮不重跑cap矩阵、不增力。

主要结论：**早期反馈把身体及自身控制历史带入难启动区域。** 尚未分离机械状态与z/动作历史各自因果作用，也未唯一证明哪个优化因素造成它；不能从此否定BFM或把所有失败归为样本少。

## 覆盖、探索及奖励边界

stand_walk真实决策A7352/49152=14.96%、B8204/49152=16.69%；45/53完整回合。完整回合中其接触开始前2s仅900/1060决策；这是定义的时间窗，不等于检测到的真实启动。全局released-recovery占29.94/38.03%。部分回合没有family×phase联合日志，报告明确边界。

旧保持项/PPO梯度范数约0.27%；提示保持强度/覆盖不足，但不是唯一因果证明。奖励把失败评为明显更差：I−7.50，A32/B32−26.03/−26.61。critic对冻结失败回报明显乐观，但不冒充已恢复训练时old GAE；逐批原rollout未保存，不能伪造任务梯度分解。

std=.008三种子随机误差：I7.94±0.86cm，A32 24.63±21.29，B32 23.39±21.06。噪声偶尔救起步但不可靠，也仍弱于初始化；不能用单个好种子宣布成功。

## 下一决策（方案已写，未执行）

优先C+B：真实启动状态/历史分布与有效反馈保持。建议下一次匹配两臂均从I开始、保持原B课程/5N/std/reward不变，各32×768决策；一臂原保持，另一臂增加经恢复证据支持的启动域同状态函数保持，训练前校准并冻结保持/PPO梯度比例。完整方案、开发/新留出、禁止无效晚状态标签、均值/随机和最差任务评价见本轮报告。若不足，再单独测0.12vs0.24启动课程；不同时调课程、std、reward和力。

本方案不构成自动发车，不续训旧A/B，不新训Student。没有“修完所有历史反例”前置门槛，也不能只延长训练。

## 数据、视频与接力

原run`runs/transition-curriculum-20261104`保持完整。本轮`runs/m41-startup-causal-20261104`包含physical-01、causal-summary、same-state、coverage、历史逐值parity及resource-summary。41物理回合/6191评价命令；RL样本0；物理973.99job秒、渲染24.20秒、主要CPU聚合28.19秒，另有小型只读分析；新增约0.8GiB。上限5000秒/3GiB，最多2GPU；所有owned PID结束，GPU5/6释放。共享checkout仍clean6023c6e257fab8fe58d13d9207fb5e85c32ee9c3。

Mac`~/Projects/compliance-bfm-diagnostics/videos/m41-startup-causal/`仅2条15.08s的1×真实qposMP4及INDEX：01为B32→I与I→B32的5.5s顺序对照，02为A32失败与4.5s恢复。SHA、时长、完整解码和代表帧视觉检查通过；不是新训练模型/最佳噪声seed视频。

新会话先读本文件、纲领、本轮报告、Issue41及实时资源。Library2026-10-28是历史快照。保护共享BFM/PD/权重/main、SoftSONIC与其他用户；不运行Bridge/watch、不做真机。完成本阶段后停止。
