# CURRENT_STATE — 新会话先读

核对：2026-11-03。**Teacher30阶段已结项；新授权Contact-Guided Heading / Turning阶段已启动，Issue36记录实时进展。** M11 continuation A160/B114仍停止，不重启、不当等预算终检。

## 当前入口

[研究纲领](RESEARCH_OVERVIEW.md) · [本轮完整结果](T30_TRANSFER_RESULTS.md) · [Issue35](https://github.com/GaoE05/robot-research-bridge/issues/35) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。上轮[STOP_SENSING_RESULTS](STOP_SENSING_RESULTS.md)/Issue34、[TRANSFER_RESULTS](TRANSFER_RESULTS.md)/Issue33及所有旧模型/失败完整保留。

## 已验证与边界

- 干净Direct特权Teacher及ideal-force最小Student迁移成立；无在线CyclicFeedback/M10 Student。Teacher30停止窄分布PPO的真实学习收益保留，但不保证任何新任务或Student偏离状态都更好。
- 本轮Teacher接管前qpos/qvel/force完全一致；S0早停止2/5、晚停止5/5满足筛选。S1warm20个端点只有4个成对恢复窗口，20条有效局部标签用于S2，未泛化成整条轨迹标签。
- S1从r0混合旧数据和943条Teacher30标签学习；S2追加有依据的局部访问状态聚合，与S1matched均7000总更新。S2四个开发接触条件优于匹配S1，但不是全面升级：新早停止S0/S1/S2误差6.594/6.762/8.295cm，力3.645/3.578/4.509N；新晚S2由S0 7.682cm/4.201N改善至6.741/3.878。正常S2仍差于r0，保留r0作通用基线。
- 释放与平均力并非同一结果：新晚S2保持有收益，但释放root路程6.00cm vs S0 2.20cm；新早S1释放1.24cm vs S0 8.00cm。局部请求、无接触保持保留；76回合无记录的物理安全终止或双足同时失去支撑。
- 同4000更新/数据/seed有力S1warm vs无力N30：新晚6.934cm/3.718N vs12.019/4.646；新早7.679/4.259 vs11.385/3.881。无力不是直接置零，删除300力列重新训练725维；更好标签没有自动解决差距，不代表无力不可行。G1准确外力硬件未知，不宣称真机可部署。本轮正式物理均5N，旧7.5/10N仅历史诊断。
- q_PD大跳变主要是移动期右踝pitch。保存状态分解中只换z贡献约0.4–1.5°，只换BFM观测约41–53°；实际关节没有20ms转50°。本轮目标峰值仍可超过50°。固定0.1s高层平滑使整体跟随更差，未采用，不改BFM/PD。

## 当前授权与执行状态

[Issue36](https://github.com/GaoE05/robot-research-bridge/issues/36)：新主线为已有缓弯/heading的Direct特权反馈获取；并行科学工作包是有限访问状态覆盖/采样比例对照，保留S0，不直接续训S2。实时仅GPU1空闲，因此物理任务串行；旧GPU5/6已被其他任务使用。

预算18000累计job秒/10GiB/12000 Student更新，Heading PPO最多20×768决策；已完成两类M10行为来源及各4000步Direct初始化，未优于Teacher30，负结果保留。Heading PPO已启动，runtime `ee45292`，从Teacher30权重、新优化器开始，最多20×768决策；GPU1，run `heading-transfer-20261103/B-train-04`。A访问覆盖/比例对照已准备，因单卡串行尚未运行。资格与训练分别使用独立contact-heading、contact-heading-next、contact-heading-train worktree；普通推送仅个人research分支。不在线引入force-to-motion规则；M10仅作初始化/对照。历史20标签高重采样是待检验假设，不是已证实唯一原因。

释放后不再把旧anchor几何距离或合理收步判为主动跟随失败；保持期仍有接触的偏差/牵拉必须保留。有限域Teacher→ideal-force Student成果已固定；S2早停止/正常退化不抹去旧成果，也不阻塞缓弯资格。原0.1s平滑负结果保留，不重复调参。

## 精确身份与位置

长期个人分支：[gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)。本文件随结项文档提交；物理runtime主A/C `8af5a87cf6684469d28a99a11e156ecdb0ba9575`，质量D `23309461185f0b10cdc6dfc291c8665f6213fa80`，此前源码已推送`290c458e3a8c5861f3c13a1a532840300108ecea`。不以文档commit替代runtime。

- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- Teacher30 SHA `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，`bridge-v2-owned/runs/stop-sensing-20261103/B1-train-02/job/best.pt`。
- PPO15 SHA `92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe`；原r0 SHA `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`。
- S1warm SHA `2a4e8babb81cad35a127d8feeb4bfe594bb93915d476375b37d87bcc2e8e9cff`；S1matched `11b82954b10f68961d05f32a74b16fb30e79efc58f428b9852572268974744ad`；S2 `2ecfff8e3ff8ee70e2b8889cea85820552747a9c69f47f616294401339a27b1d`；N30 `49d9aa94114b3e5eadb679796b43160725d72830da5353a63b4d4db2d809bf83`。
- 完整服务器root `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/t30-transfer-20261103/`。前三Student在`A-prepare-01/job/`、N30在`C-final-02/job/`；协议、阶段分项、所有失败见报告和`t30_artifacts/results.json`。原B505/N505等均保留，不覆盖。
- Mac视频：`~/Projects/compliance-bfm-diagnostics/videos/teacher30-transfer-20261103/`，3条完整1×真实qpos对照＋INDEX。旧视频目录原样保留。

## 成本与接力规则

本轮完成14000监督更新，0RL；76回合/236888记录物理步/11866高层决策（非RL样本）；含视频2293.69累计job秒，约1.39GiB。授权上限18000秒/10GiB/16000更新/最多3张实时空闲卡，**剩余额度不自动授权下一阶段**。执行结束核对ledger和ownedPID，不沿用GPU空闲快照、不重放队列、不启动Bridge/watch。

新会话先读本文件、纲领、Issue35及协议，再只读核对ledger/Git/模型SHA。普通工程在用户授权内自主处理；不改共享BFM/PD/权重、团队main、SoftSONIC/他人任务，不下载大资产，不做真机。推送仅个人分支、无force/main/模型大数据。2026-10-28 Library PROJECT_STATE/SESSION_HANDOFF是历史快照，实时状态以GitHub/本文件为准。

当前进度补充：Teacher30在统一条件移动期左/右弯yaw变化+5.49°/−8.66°，左弯幅度不足。旧M10报告较好的左右弯数字来自其共享Student，不应笼统当作原规则专家证据。本轮两种来源迁移失败不抹去旧条件成功，也不宣布整个BFM不能转向。Student迁移本轮尚无新结果。运行期间本文件状态以Issue36/ledger为准。
