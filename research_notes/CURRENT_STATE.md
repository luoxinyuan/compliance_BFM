# CURRENT_STATE — Issue53小续跑已获授权，干净交互标签增量验证

2026-11-08：独立stage interactive-increment-20261108；协议 interactive_increment_protocol.json。34正确标签与冻结BC500核验完成；准备1000交互监督更新，原开发选模、原固定条件评价，之后按预登记起点规则做两seed各6144 Student真实RL决策。尚无新结果，旧#53结论全部保留。硬预算12000累计job秒/4GiB/2000监督/64新查询/12288RL；不恢复独立Expert PPO、不扩技能、不改BFM/PD。

---
以下#53结项快照供接力：

# CURRENT_STATE — Issue53完成并停止：有效恢复与旧数据Student增益，交互比较未完成

最后核对：2026-11-08。当前无训练、无队列、无Student PPO。[Issue53](https://github.com/GaoE05/robot-research-bridge/issues/53) · [完整报告](INTERACTIVE_PILOT_RESULTS.md) · [精确指标/身份/成本](interactive_pilot_evidence.json) · [长期目标](RESEARCH_OVERVIEW.md)。Issue52独立Expert PPO保持结项。

## 可信结果

- 冻结原生goal/motion＋7候选/2.4s因果物理搜索Teacher，在S4000早期4.7s状态接管2秒后交回Student，整段55.217cm/9.218N→7.706/5.539。20标签经fresh连续重放，输入/last-z/q/qdot/qPD/力/active-anchor全0差。
- 修复后post短请求5.4–6.8s窗口2.688cm/2.775N→2.319/2.098；交回后整段仍10.043/6.152，保持未解决。只确认2处有限窗口，晚期长距6s域尚未合规复验。
- **有效的新受限Student结果来自旧数据续BC500，而非交互/PPO**。从S4000起点freshAdam/lr1e−4额外500更新，固定开发选模后新6cm3.200/2.671、新59cm继续10.794/7.201、停止9.127/5.916，对照S4000为15.656/9.178、49.505/9.724、42.808/9.685。同一checkpoint；新前缀8.4s前880physics/220BFM记录完全相同，之后真实需求才分叉。单seed/有限身体状态，不称general。
- 质量仍有限：59cm继续保持11.958cm/8.573N、37.24%接触限幅，整段yaw约−18.4°，释放后root约+18cm；停止保持更好但yaw约−22.6°。无接触仍有约5.9cm左足调整。qPD60°目标峰不等于实际关节同幅瞬转。
- FB shadow20配对/40额外序列，0新增胜出，Ftop4保留5/20物理最优，暂停当前提名器。只测了当前有限Student域，不否定整个FB。moving成本点原reward-z曾被选但预测仍9.55N，未当有效Teacher标签。

## 必须保留的失败与修复

本轮跨reset使用旧BranchState遗漏接触函数的spec/base闭包绑定，post-onset2处有1–2cm非预期anchor偏移。**原3/3资格撤回，交互臂混入40个受影响状态，不推广，不以其比较宣布DAgger胜负。** 原始模型/gate/轨迹保留，另有erratum。局部CrossResetState修复在3状态对不间断Student的完整physics/control逐值通过，不改BFM/PD/接触公式。现有34个正确去重标签=20早期＋14post，相关后继而非34独立任务，**尚未训练**。

本阶段两臂各1000监督已全部计入2000上限；停止受影响评价，不扣除失败成本后重训。只完成无错误新标签的旧BC冻结对照/留出，**Student PPO=0**，不能判RL有效、无效或必要。Issue50/51/52及历史S0迁移成功不抹去。

## 当前唯一推荐（未授权新训练）

以有效旧BC500为有限S-init，下一阶段做冻结BC vs同起点受限actor自身小预算RL，并在已验证域稀疏Teacher查询。不要继续独立Expert PPO、扩大技能/接触族或强求混合BC先完美。若需严格补齐标签价值对照，可复用本轮旧数据模型/批次，仅再授权正确交互臂；当前不自动执行。

## 身份、位置与成本

- 长期审查分支 `luoxinyuan/compliance_BFM:gaoe05/research`；实际最终实验源码 `7617d96298c4d19ec94adbfc90a11a3a7140fc14`，文档最终SHA见Issue53结项，不把文档HEAD冒充runtime。各环节runtime表在报告。
- stage：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/interactive-contact-pilot-20261108`，所有owned作业退出，无自动续训/Bridge/watch。SSH1016已验证NVIDIA环境，NVML命名差异保留，下一发车重新核查权限/资源。
- S4000 SHA `8f5e7287e974deedba1a8d841dd65d7692e944e278bd4a091f1ec224f167a479`；BC500 SHA **`a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`**，`runs/bc-fit-01/old_data_only-500.pt`。新S-init f414ea3...是旧正式step0，历史S0 0ef3d43...另存，不互换。
- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`、原PD/Direct256D/10-50-200Hz、K80D8/10N不变。Student1639D仅关节/IMU/动作/真实z历史/ideal-force及已知安装；无anchor、伙伴真值、clock、active、搜索phase或特权hidden，G1真实力传感未知。
- 实际96查询/40FB额外序列/341760候选物理步，监督2000、RL0；约4375.2/12000 job秒含300CPU预留，约1.037/8GiB输出。失败和中断计成本，不作新成功样本。
- 正确34标签：`runs/final-analysis-01/corrected-new-labels.pt`，未训练；数据/原始预测/真实执行/反事实/接管分档。受影响混合模型保留，不作为baseline。
- Mac：`~/Projects/compliance-bfm-diagnostics/videos/interactive-contact-pilot/`，**3个1×真实qposMP4＋INDEX**，SHA、时长、全解码和画面均通过。[视频说明](INTERACTIVE_VIDEO_INDEX.md)。

## 跨会话接力

先读本页、OVERVIEW、Issue53最新结项、报告与erratum，再对实际ledger/源码/模型SHA核对。I/C0/T30/S0/D2、M8–M10、M38转腰、M41启动因果及Issue47–52全部保存；不恢复消费完的旧任务。Library2026-10-28是历史快照。只个人worktree/研究分支，不改main/共享BFM/PD/权重/SoftSONIC/他人任务、不做真机，不上传weights/raw轨迹/凭据。新阶段需独立预算/当前资源核验。
