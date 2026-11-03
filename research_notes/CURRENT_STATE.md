## 新授权执行覆盖｜2026-11-03

Teacher30迁移/恢复与质量阶段已授权，正在准备；尚未启动。Issue35、T30_TRANSFER_PROTOCOL.md/t30_protocol.json为当前计划。上轮结果如下完整保留。新预算18000累计job秒/10GiB/16000监督更新/最多3GPU，不RL。计划S1warm4000+匹配续训3000+条件成立的S2聚合3000+无力4000。不得将下方上轮已结束状态当作本轮无授权。

# CURRENT_STATE — 新会话先读

核对：2026-11-03。**停止覆盖、Teacher停止PPO、传感约束及追加力上限诊断全部完成；无本项目任务仍在训练。** 旧M11 continuation A160/B114仍停止，未重启，不作为等预算终检。下一研究阶段尚未启动。

## 入口与科学状态

纲领：[RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)。当前报告：[STOP_SENSING_RESULTS.md](STOP_SENSING_RESULTS.md)、[FORCE_CAP_DIAGNOSTIC.md](FORCE_CAP_DIAGNOSTIC.md)，[Issue34](https://github.com/GaoE05/robot-research-bridge/issues/34)。完整历史与决策：[Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。上轮[TRANSFER_RESULTS.md](TRANSFER_RESULTS.md)/Issue33的Teacher→ideal-force Student最小迁移成功保留。

- **停止数据覆盖有价值，但新Student不是全面升级。** 同原r0/4000更新，B混合6条停止Teacher回合，比A只增加旧数据更新更好，两个更新seed一致支持总体趋势。新晚停止B误差6.929cm/力3.872N，原r0为7.015/4.313；新早停止B7.874/4.312反而差于r0 6.587/3.580。保留r0，不按终检挑统一赢家。
- **Teacher本轮有限域训练有真实收益。** Direct初始化、原reward/BFM/PD/5N保持；40批30720RL决策/123优化步，75%停止窄分布。开发预选30批，三个封存条件均改善：新早7.491/4.085→6.486/3.866，新晚7.113/4.400→5.980/3.716，新正常8.424/4.145→5.638/3.435（cm/N）。开发综合改善1.51%不足预定2%续训标准，因此结束40，不以新测试延长或重选。
- Teacher保持质量也改善，但q_PD连续性未全面解决：新早最大单控制步37.94→53.92°，局部25.18→41.75°；未见双足同时失去支撑或物理安全终止。此模型尚未查询Student偏离恢复域，也未迁移给本轮Student。
- **无力路线不能判死刑。** 删除300力列重新训练725维Student，在部分时段可跟随，但5N下总体弱于力输入，停止/释放有再动作。输入没有新增本体测量噪声；“弱接触被自身步态掩盖”是解释假设，不是已确认传感噪声根因。
- 用户授权冻结5/7.5/10N诊断：同一个已暴露条件12回合。两个无力seed误差5→7.5N分别12.223→7.629、14.684→9.721cm；力4.531→4.443、4.430→5.924N，负担不一致改善。10N不一致更好、力更高，起步不一致提前。Standing无主动迈步，但不能完全分离被动动力学辅助与可观测性。正式训练cap仍5N，不自动提高。
- 传感压力仅改输入：B505新晚强噪声/偏置/100ms/20%丢测变10.348cm/4.250N；单噪声实现不足统计结论。G1实际力传感硬件未知；ideal-force与无力模型均未获真机验证。

## 当前瓶颈与下一研究决策

主要瓶颈是停止/释放访问状态恢复、不同观测权限下闭环分布，以及控制连续性；不是缺少可训练Student，也不是已证明BFM或无力方案不可行。新Teacher30改善了有限任务域，可作为后续独立冻结标签源；先验证少量代表Student访问状态的恢复，不以Teacher自己轨迹成功推广全部状态。

**建议下一阶段优先围绕新Teacher30做有力/无力Student的停止恢复迁移；若纳入力强度，将5/7.5N作为明确训练/评估因素并保留5N基线，避免把加力收益误记成学习收益。尚未授权或启动新的追加训练。** 平面行为扩展最有依据为现有缓弯/heading；下沉/恢复为备选高度技能。侧移/后走/低姿态行走尚无可靠接触组合证据，见[能力清单](STOP_CAPABILITY_REVIEW.md)。椅子/多接触/真机仅远期候选。

## 身份、源代码与证据位置

长期个人分支[gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)；该文件提交是文档集成身份，不替代运行SHA。

- A及Teacher runtime `32c87b4db3aba007e4d6cf15ed5495cdffdfcb75`；C `a9f80be24f32175a41d7c36fd67549a0b82445c1`；冻结Teacher终检 `619837f564f09441a3169d793e0bd662fe5a8d59`；加力诊断 `52f5b398078295b865a3dc542a0d6ac7306865c5`。
- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- 本轮主标签Teacher旧PPO15 `92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe`；r0 Student `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`。
- 新Teacher30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，与Student采集期间旧Teacher身份分开。
- B505 `78c818478b059cfd8aeec868918c275b9429a3863aac53dd1eb84e08dd62d4ff`；N505 `038c218829f85ce0b661e38397c19121d5e49ec3ac38fde8c483d87f151893d6`。全部模型/末轮身份见[清单](stop_sensing_artifacts/model-identities.json)。

`P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`；`R=$P/bridge-v2-owned/runs/stop-sensing-20261103`。
A-prepare-03/job（A/B模型、标签、物理）；C-final-05/job（N/传感）；B1-train-02/job（Teacher初始/最佳/末轮/曲线）；E-final-08/job（冻结终检）；F-final-07/job（cap）；analysis/（分阶段XYZ、身体/支撑、完整恢复曲线）。原始轨迹和权重不进Git。

Mac视频：`~/Projects/compliance-bfm-diagnostics/videos/stop-sensing-20261103/`，4条MP4+INDEX。01覆盖vs额外更新，02理想力/退化力/重训无力，03Teacher初始化/30/40，04同N505真实5/7.5/10N。已本地完整解码、时长、SHA和画面核验，实际qpos1×。

本轮实际6101.35累计受监督进程秒、约2.81GiB输出；24000监督更新，30720RL决策、612996物理步另107400预填步，179完整训练回合。全部完成；18000秒/10GiB是本轮封闭预算，不能当下次新授权。两次启动工程失败保留，未改运行中源码。无需为了满预算继续计算。

## 可靠旧能力与跨会话接力

M5–M8有限LPP/共享策略、M7连续转换、M8躯干下沉、M9行走/有限转弯、M10任务合适行为及M1–M4负结果均保留。当前右腕直行新模型没有证明保留全部旧部位/技能，不替代M8/M9B/M10。旧Library2026-10-28文件仅历史快照。

1. 先读本文件、纲领、当前报告及Issue34/18；核对ledger/result和精确SHA，不重启已消费任务。
2. SSH pro5000，`source $P/env.sh` / `$P/.venv/bin/python`。每次GPU前实时查GPU/CPU/RAM/I/O；无GPU不擅自改CPU跑BFM，共享空余不是个人独占配额。
3. 仅个人worktree/输出。不可改main、共享BFM/PD/原权重、SoftSONIC/他人任务，无Bridge/watch、真机或自动大资产下载。
4. 重要方法/里程碑将获准源码普通push至个人research，记录runtime/模型/失败/视频；不强推、合main、覆盖历史。下一阶段需用户科研决策，不自动扩展。
