# CURRENT_STATE — 新会话先读

核对2026-11-03。**Issue36已结项；用户新授权Contact Geometry→Translation/Rotation阶段现已启动，Issue37。** 保留S0/T30，不原样续训旧HeadingPPO10。M11 continuation仍停止。

[Issue37](https://github.com/GaoE05/robot-research-bridge/issues/37)：单腕曲线与刚性torso左右双点；每点2.5N，总sum norms≤5N。先真实几何/原生资格，再Direct privileged学习，成功部分才做共享传感表示迁移。阶段root `bridge-v2-owned/runs/contact-geometry-20261103`，worktree `gaoe05/contact-geometry`；预算21600累计job秒/15GiB/最多3张实时空闲GPU，预登记RL≤61440决策、源监督≤8000及Student≤8000更新。GPU1物理准备已发车，当前没有新Teacher成功或Student迁移结论。新授权优先于下方上轮“建议”。

长期方向：单/双点平面引导→多部位分别/组合→高度与移动→未见接触配置和身体状态泛化。不建设Bridge，不改共享BFM/PD。

入口：[研究纲领](RESEARCH_OVERVIEW.md) · [本阶段完整结果](HEADING_TRANSFER_RESULTS.md) · [Issue36](https://github.com/GaoE05/robot-research-bridge/issues/36) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。上一阶段[T30迁移](T30_TRANSFER_RESULTS.md)/Issue35完整保留。

## 已固定能力与当前结论

- 有限域Direct Teacher→ideal-force Student直行、减速、保持及释放链成立。S0仍是旧能力基线，Teacher30是较稳特权基线；S1matched/S2和本轮新模型都是研究对照，不能全面替换。准确外力真机来源未知，不称已实机部署。
- 本轮A：6个独立Student回合、36次完整状态接管，13有效局部窗口/5回合取得65标签。两组相同数据/初始化/seed/各3000更新，仅访问标签8/128与32/128比例不同。正常S0/低/高为6.505/7.006/8.441cm，新早6.456/8.331/6.921cm，新晚6.785/7.006/5.941cm。低比例较保旧，高比例较改善晚停止，均非全面升级；不能将旧S2失败唯一归因于重采样。
- 本轮B：两种M10来源的Direct初始化都未优于T30，负结果保留。转用T30权重+新优化器，20×768窄曲率PPO，开发预定规则选batch10；并未根据终检重选。执行仍是特权当前反馈→学习z→冻结BFM→原PD，无在线手写force-to-motion/M10 Student。
- 新右弯T30→PPO10误差6.711→6.340cm、力4.173→4.153N、heading4.13→3.74°；同前缀转右5.925→5.543cm、3.713→3.350N，但heading3.20→3.64°。新左5.220→5.914cm、heading10.28→13.34°，局部请求也退化。故存在真实反馈及部分收益，不宣称可靠对称转向或general策略。
- 正反曲率分支在8.6s之前qpos/qvel/force/anchor完全一致，之后真实伙伴变化产生不同响应；变化后状态也不同，不能称隔离出的同状态actor因果效应。
- 释放后旧anchor距离不是任务误差；正常收步允许。仍保留释放前牵拉、自由恢复root/手腕速度/支撑及q_PD连续性。旧右踝观察反馈大跳变与0.1s平滑退化不重复修补。本阶段离线分期边界bug已修正，只用physical-metrics-v2；物理、训练和选择未受影响。

## 执行、资源与下一决策

累计4844.92 job秒、约2.80GiB；Student6000更新，另Expert初始化8000监督更新，PPO15360决策/306572physics步/76643control步，初始化另50400physics步。实际GPU1/2在实时资源检查后使用，owned均退出；不沿用空闲快照。共享代码/BFM/PD/权重/他人进程未改。

**唯一建议**：下一授权阶段以冻结T30为主，做有限曲线引导ideal-force Student迁移与同前缀方向变化，PPO10仅作其已证明有益域的独立比较，保留S0直行/局部回归。不继续无界停止打磨，也不把有退化的PPO10推广成全局Teacher。此建议尚未执行。

本阶段源码runtime：A `094d4492f7097d9f7ccabb64fefd6b359dcb8a6c`，B `ee45292eb204e370b002470400b6f443c9f0c65c`。文档提交不等于runtime。
结果根：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/heading-transfer-20261103`；完整SHA、选择、曲线、资源与物理分项见报告和`heading_artifacts/`。Heading选中SHA `31963f604b56a500b004a1ed6019f3784a44871c6d840eba254ef06b4762ebc9`。
Mac仅2条新视频与INDEX：`~/Projects/compliance-bfm-diagnostics/videos/heading-turning-20261103/`；同前缀转右收益和新左局限，实际qpos、1×、完整接触/6s恢复，SHA/时长/解码已验证。

## 精确身份与位置

长期个人分支：[gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)。以下为上一T30阶段身份；本阶段runtime见上方报告。上一阶段物理runtime主A/C `8af5a87cf6684469d28a99a11e156ecdb0ba9575`，质量D `23309461185f0b10cdc6dfc291c8665f6213fa80`，此前源码已推送`290c458e3a8c5861f3c13a1a532840300108ecea`。不以文档commit替代runtime。

- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- Teacher30 SHA `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，`bridge-v2-owned/runs/stop-sensing-20261103/B1-train-02/job/best.pt`。
- PPO15 SHA `92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe`；原r0 SHA `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`。
- S1warm SHA `2a4e8babb81cad35a127d8feeb4bfe594bb93915d476375b37d87bcc2e8e9cff`；S1matched `11b82954b10f68961d05f32a74b16fb30e79efc58f428b9852572268974744ad`；S2 `2ecfff8e3ff8ee70e2b8889cea85820552747a9c69f47f616294401339a27b1d`；N30 `49d9aa94114b3e5eadb679796b43160725d72830da5353a63b4d4db2d809bf83`。
- 完整服务器root `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/t30-transfer-20261103/`。前三Student在`A-prepare-01/job/`、N30在`C-final-02/job/`；协议、阶段分项、所有失败见报告和`t30_artifacts/results.json`。原B505/N505等均保留，不覆盖。
- Mac视频：`~/Projects/compliance-bfm-diagnostics/videos/teacher30-transfer-20261103/`，3条完整1×真实qpos对照＋INDEX。旧视频目录原样保留。

## 成本与接力规则

上一T30阶段完成14000监督更新，0RL；76回合/236888记录物理步/11866高层决策（非RL样本）；含视频2293.69累计job秒，约1.39GiB。授权上限18000秒/10GiB/16000更新/最多3张实时空闲卡，**剩余额度不自动授权下一阶段**。执行结束核对ledger和ownedPID，不沿用GPU空闲快照、不重放队列、不启动Bridge/watch。

新会话先读本文件、纲领、Issue36及协议，再只读核对ledger/Git/模型SHA。普通工程在用户授权内自主处理；不改共享BFM/PD/权重、团队main、SoftSONIC/他人任务，不下载大资产，不做真机。推送仅个人分支、无force/main/模型大数据。2026-10-28 Library PROJECT_STATE/SESSION_HANDOFF是历史快照，实时状态以GitHub/本文件为准。
