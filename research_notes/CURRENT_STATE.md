# CURRENT_STATE — 新会话先读

最后核对：2026-11-04。**Transition-centered privileged latent RL已完成有界训练、48回合封存评价和视频交付；未获得优于初始化的开发选定模型。当前本阶段无训练，未启动下一实验。**

入口：[完整结项与归因边界](TRANSITION_RESULTS.md) · [Issue41](https://github.com/GaoE05/robot-research-bridge/issues/41) · [预登记](TRANSITION_PROTOCOL.md) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。旧“执行中”记录为历史。

## 保留的事实与基线

总目标Contact-Conditioned Whole-Body Motion Transitions。T30→S0已建立有限域右腕直行/停止Teacher→ideal-force Student链路；ideal-force不等于G1已确认有精确外力传感。M38有限域双向转腰成立，不等于骨盆/脚步转身。M39-A/B保留直行与左右能力，组合进入/撤销仍不协调；[M39A](M39A_RESULTS.md)、[M39B](M39B_RESULTS.md)全部成功/失败保留。M39-B已结项，不原样续训，不自动推广新Student。

可靠模型（均不覆盖）：
- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- T30 SHA `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，runs/stop-sensing-20261103/B1-train-02/job/best.pt。
- S0 SHA `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`，runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt。
- D2 SHA `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`，runs/m38c-history-20261103/C-train-05/job/D2.pt。
- 本轮共同初始化M39-A best64 SHA `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`，runs/m39a-demand-20261104/B-ppo-03/job/best.pt。
- M39-B A64/B56 SHA和结果见M39B_RESULTS；未替换T30/S0，未删除。

以上runs前缀：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/`。个人长期审查分支`gaoe05/research`；具体最新文档提交由Issue41链接记录。源码发布不等于训练成功。

## 本轮实际结果

共同站立课程A与完整源运动接管课程B各64×768=49152 RL决策；A219/B242优化更新。相同初始化、Direct绝对z、1088D特权输入、5N物理、奖励、raw保持与seed；fresh Adam。源前缀保留q/qdot、支撑、BFM/动作/z/力历史，额外物理成本单列。没有新Student。

两组开发最佳均iteration0，best SHA `e1e33b521e829647e7aaac3a04b7e019f0ed99eb94883371f4437ae3d6da1fcb`；与来源SHA差异是序列化元数据，不是学习成果。最终48回合包括重复模型身份，不能计为48个独立效果证据。开发训练候选保存，不按终检重选。

主要失败是站立启动能力退化：初始化开发误差5.34cm；A32/B32为48.03/50.03cm；14s root仅前移4.54/1.12cm。B48在已行走条件反而5.37cm/3.85N→4.84cm/3.36N，但站立启动变14.23cm。不是所有能力消失，也不能拼接不同候选指标冒充成功策略。

同初态启动首个z差异仅约0.1°，随后约6.5s真实root开始明显分叉；支持闭环启动响应被更新损坏，不支持大初始指令突变或不同qpos作为解释。尚不能因果区分任务更新干扰、探索/确定性目标差距或启动反馈敏感性。A32/A64及B32按预定规则回滚，不是连续64批无中断优化。

封存初始化仍能直行：移动8.001cm/3.936N、保持0.997cm/0.796N；组合撤销后继续平移左4.645cm/4.736N、右2.860cm/4.282N。释放后不把旧anchor误差计为失败。已有T30/S0及有限双向转腰结论保留。

5/10/20N冻结诊断：10N改善部分组合跟随，直行误差和负担未改善；20N也会明显拖动standing。不能归结为5N是唯一原因。本轮主课程均5N。原生简单walking/goal混合失败不代表BFM所有组合都不可表达。B距离仅辅助诊断。

## 版本与数据

实际训练runtime `a8afaff087f535124d9e1b28be7ada57caa10bdf`；准备 `8d1a02754807eda84fc2842921c9577c56a9dafb`；封存/渲染 `5fd2798185212ec65afa56753051900305bb28a5`。结项提交见Issue41，不回写历史运行worktree。

服务器：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/transition-curriculum-20261104`。A-ppo-03/B-ppo-04保存候选、开发、回滚；C-final-02准备；C-final-05封存；final-physics.json分阶段物理，compute-ledger.json资源。紧凑归因见transition_artifacts/saved-attribution.json，可用transition_saved_audit.py只读复算。

Mac：`~/Projects/compliance-bfm-diagnostics/videos/transition-curriculum/`，3个MP4与INDEX，SHA/时长/完整解码已验证。**A/B选回初始化，三个面板相同；视频不是新训练改善证据。** 本结项不声称已完成视觉QA。

## 资源与下一决策

本轮上限24000累计job秒/15GiB/196608RL决策；实际98304决策，账本计算及渲染约11296.6s，加单列保存数据CPU分析；全部账本job结束。首次准备缺物化源码失败3.43s/0物理，保留。没有追加训练、Bridge、watch或下一阶段任务；不使用旧GPU/PID快照发车。

唯一推荐：以同真实启动状态做有限初始化/失败候选接管干预，定位启动丢失的控制与优化原因，再决定是否改课程、保持或探索；不是继续无界训练，也不设置“所有旧任务完美”新门槛。此建议尚未执行。

新会话先读本文件、RESEARCH_OVERVIEW、TRANSITION_RESULTS与Issue41。当前是已完成但无新优胜模型的负结果，不要重复启动。共享BFM/PD/权重/main、SoftSONIC和其他用户任务保持不变；Library2026-10-28仍为历史快照。
