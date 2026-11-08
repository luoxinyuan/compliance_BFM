# CURRENT_STATE — Issue53 路线A交互pilot已授权，正在执行

2026-11-08更新。[Issue53](https://github.com/GaoE05/robot-research-bridge/issues/53) · [预登记方法](INTERACTIVE_PILOT_METHODS.md) · [协议](interactive_protocol.json)。当前仅准备有限Teacher恢复/成本与FB影子测试，未产生新BC/PPO结论。Issue52维持结项，不恢复独立Expert PPO。

独立stage `experiments/interactive-contact-pilot-20261108`，启动前GPU无进程、CPU负载0、可用RAM199GiB，使用现有验证NVIDIA环境。总预算12000 job秒/8GiB/96完整搜索/40额外FB序列/2000监督更新/12288 Student RL决策。S4000失败参考SHA与新S-init/历史S0分开。先从实际Student完整历史确认Teacher恢复域；不合格标签不蒸馏。FB不改变正式7候选Teacher。监督和Student自身RL均按协议条件启动，不自动扩技能。

## 上阶段结项快照（事实保留）

# CURRENT_STATE — Issue52完成：PPO有限收益，独立critic不足以解决均值反馈学习

最后核对：2026-11-08。当前**无训练、无排队自动续训、新seed或Student**。[Issue52](https://github.com/GaoE05/robot-research-bridge/issues/52) · [完整结果](P4000_PPO_RESULTS.md) · [方法](P4000_PPO_METHODS.md) · [指标/模型清单](p4000_final_evidence.json) · [Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。先读[长期目标](RESEARCH_OVERVIEW.md)，不要把旧Library快照视为实时状态。

## 当前可信结论

目标仍是冻结BFM上的Contact-Conditioned Whole-Body Motion Transitions / General Compliance，原生能力→真实反馈Expert→受限观测Student及自身闭环学习。当前不是修复某条固定walking动画。

Issue50原生goal/motion＋因果滚动MuJoCo搜索取得有限域有效反馈，不是新神经PPO Teacher；约23秒/100ms命令，reward-z未成为有效主来源。Issue51真实执行数据训练fresh特权BC P4000部分有效，fresh受限ideal-force S4000失败；两者不能混淆。历史Teacher30→旧S0有限域迁移成功继续保留。

Issue52 P4000热启动真实PPO：原5201完成32批，原5202完成30批更新、第31批采样后FP32 batch-shape logprob断言退出；不是完整双seed结论。保存失败和真实采样成本，已用canonical-row FP32 actor统一采样与评分验证修复，原概率/KL断言不放宽。

根据用户新授权，随后固定seed5201，完成同数值路径joint与independent critic各32批/24576真实RL决策。独立critic不再随actor KL提前停止，每批18拟合步；网络、reward、std=.008、任务比、10N/K80D8及BFM/PD不变。它不是第二seed，只有单seed开发证据。

| 同条件确定性接触期3D RMSE cm / 实际力RMS N | P4000 | joint32 | independent32 |
|---|---:|---:|---:|
| 8cm小请求 | 2.511 / 2.076 | 2.516 / 2.079 | 2.488 / 2.056 |
| 24cm中距 | 6.646 / 5.515 | 6.691 / 5.556 | 6.782 / 5.622 |
| 较快72cm | 13.636 / 7.299 | 14.074 / 7.407 | 13.995 / 7.363 |
| 正常72cm | 12.306 / 7.939 | 12.999 / 8.337 | 11.752 / 7.722 |
| 同前缀停止 | 8.350 / 5.559 | 8.478 / 5.600 | 8.443 / 5.558 |

**正常长牵引有真实行走，独立critic有小幅收益；没有多条件共同改进，不替换P4000。** 独立方法开发best仍0；joint best8仅极小分差。无接触稳定、无安全终止不等于动作质量全部合格。新封存条件未使用，不宣称未见泛化。

## 主要瓶颈与诊断结果

- 原小σ与KL合同确实约束每批均值方向变化约.035°；32批固定状态累计仍约.46–.50°，不是完全没更新；径向只占差能量7–10%，不支持主要浪费在raw缩放。
- 两臂实际24576决策均为移动6218、减速343、保持4401、接触前2587、无接触2342、释放8685。移动25.30%、减速1.40%，关键反馈覆盖偏少是有依据假设，不是已证实唯一原因。
- 3072保存决策GAE复算误差<2e−7；移动/减速force成本约91%，当前没有Gaussian force reward sigma。限幅使部分区间force项恒定，但其他reward仍有信号；不自动提高cap。
- 14个完整同状态2.4s物理分支全部完成，前缀features/qpos/qvel逐值一致。学习修正×3在停止略好、行走不及原独立均值；std翻倍在这两个窗口没有改善。不能简单说“只要放大动作/噪声就能解决”。
- 正常长牵引原P4000三种保存随机流均比确定性更好，但其他任务结果混合；新PPO未保留该随机收益。短窗reward与critic尾部bootstrap排序有分歧，但未观测更远未来，不能直接认定critic错误。
- 旧姿态/footstep/q_PD问题保留。释放后原anchor误差不是任务失败；接触仍在时的偏差和负担照常评价。q_PD跳变不同于真实关节20ms跳过相同角度。

## 用户视频反馈补充（2026-11-08，仅保存轨迹复核）

[完整视频分析](https://github.com/GaoE05/robot-research-bridge/issues/52#issuecomment-6720848432)。66停止分支anchor实际前移约51.5cm，非完整66cm；独立32保持期1.725cm/1.423N，后段有效，不由整段均值否定。72独立32在9.0→9.8s腕肩距离34.7→22.9cm，手向内收，误差2.65→18.54cm、力达10N；实际解除要到11.8s，故不是释放后的正常恢复。P4000/joint32也有同类问题，初始化已存在而PPO未修好；尚未区分高层z/BFM反馈/耦合主因。三行都是特权Expert，不是Student差距。

下一建议聚焦约9s移动结束附近的完整真实状态，少量因果搜索接管验证稳定控制方向，再做关键移动末段/减速/保持覆盖的匹配反馈训练；同时保住66好保持。未启动新物理或训练，不更改结项/选模，不自动调整reward/std/KL/cap。

## 下一步决策：未启动

推荐同P4000＋独立critic下做一次**关键移动/减速实际覆盖**的匹配课程对照，固定reward/std/KL/物理/网络；跟踪有效随机反馈是否真正进入均值。若仍受更新几何限制，再单独研究探索/步幅，不能一次改多个因素。没有授权自动新增训练；当前阶段完成后停止，先由用户审查下一方案。未开展新Student、大型F/SF planner或reward候选扫描。

## 源码、模型、资源、视频

- 服务器：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM`，SSH1016当前已验证NVIDIA环境；NVML L20Z/CUDA9.0硬件命名差异保留，不重装，不沿用旧GPU空闲快照。stage `experiments/p4000-feedback-ppo-20261108`，ledger和各job launch是准确运行身份。
- 原训练runtime `93383a406e4b4923418af0b47aca2077ca68e40c`；新配对runtime `e1d6929d297442ad2ab4ebb7912306f292750354`；最终物理诊断/视频runtime `92ec566c554bfa8e29238c9cb1f68d0ac84b6169`。文档HEAD不替代运行SHA。
- P4000：Issue51 `runs/bc-fit-01/P-4000.pt`，SHA `9f560903072b355e6d2af537af725b9e61210ecea87a56151aa5d59dff45d164`。
- independent candidate32：`runs/single5201-independent-32/candidate-0032.pt`，SHA `e9d0a5989096e87a2ca402aabbfca889578119069ead31d84a96da87c9aa2d72`。
- joint candidate32：`runs/single5201-joint-32/candidate-0032.pt`，SHA `2fc142549a390c8ce8624e7457c0f2080072751d808d9d34a2697f7ef6f8fdc6`。best/last完整身份见结果，不以文件名判断。
- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`，原PD冻结；1088D策略当前是特权Expert，没有Student部署权限结论。
- 总实际RL采样97536，包含失败批；累计job20183.05秒含CPU预留，上限24000；输出<10GiB，详见[p4000_closeout_ledger](p4000_closeout_ledger.json)。诊断6720分支physics步单记，不算RL训练。没有新增第二开发seed或封存评价。
- Mac实际目录：`~/Projects/compliance-bfm-diagnostics/videos/p4000-feedback-ppo/`，仅**2个MP4＋INDEX.md**。正常72cm12.8s、停止11.8s，25fps/1×；每条上/中/下为P4000/joint32/independent32，每行世界斜视＋俯视。真实qpos、未重新挑seed、完整保存接触/释放，SHA/全解码/画面验证通过。它们全是特权Expert，不是新Student。
- 长期审查分支 `gaoe05/research`，最终文档提交见Issue52结项评论。训练/诊断worktree及失败保留，不改正在运行的源码。

## 接力保护

I/C0/T30/S0/D2、M8–M10、M38双向转腰及M41启动因果结果保留。本轮fresh S失败不推翻历史S0成功。M11初轮与continuation A160/B114提前停止分别归档；不重启旧任务。原Issue47/48/49/50/51已结束，不按旧计划恢复。

新会话先读本页、OVERVIEW、Issue52最新结项、实际ledger与用户新授权。只个人worktree/gaoe05/research；不改main、共享BFM/PD/权重、SoftSONIC或他人任务，不做真机；不上传模型、raw轨迹、视频或凭据。Library2026-10-28为历史快照，Project Master保留完整决策历史，本页为精简入口。
