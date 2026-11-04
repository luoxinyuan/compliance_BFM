# 当前新授权：Transition-centered privileged latent RL

2026-11-04：Issue41。M39-B保持结项；本轮独立worktree正在实现源策略真实前缀接管与共同站立课程对照，尚未启动训练。5/10/20N冻结诊断已获本轮明确授权；上一版力范围A/B扩展训练不执行。两臂统一5N、绝对z、共享Direct；24000累计job秒/15GiB/196608RL决策上限，当前两张空闲卡。协议见TRANSITION_PROTOCOL.md。下方历史的“待授权”已被此次用户指令更新，不改旧结果事实。

# CURRENT_STATE — 新会话先读

最后核对：2026-11-04。**M39-B完成并停止：两臂各64批，52新封存回合与真实视频已交付；尚未获得可靠连续平移/旋转反馈，action保持不全面优于raw。** M38双向转腰和直行成果保留；下方M39-A为历史基线，最新结论见本文件末段M39-B及最终报告。新增力上限实验仅提出、尚未执行。

入口：[研究纲领](RESEARCH_OVERVIEW.md) · [M39-A完整报告](M39A_RESULTS.md) · [Issue39](https://github.com/GaoE05/robot-research-bridge/issues/39) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。旧[M38-C](M38C_RESULTS.md)和所有阶段报告/失败保留。

## 可靠基线及真实新结果

T30→S0仍是有限域右腕直行/停止Teacher→ideal-force Student基线。ideal-force不代表G1已确认具有精确外力传感。D2及本轮模型为1088D特权Direct Expert，不是新Student；冻结BFM/PD、合法256D z、50Hz actor/200Hz物理、右腕5N或rigid torso双点各2.5N。没有在线force-to-yaw/M10 Student/按方向选模型。

- 覆盖监督从D2开始3000更新，10独立有效源回合/1426有效标签。零需求、小请求、平移和新组合跟随较好，但新左+9°实际仅+.78°、旧停止退化；不能全面替代。相对无力仍有+5.09°增量，准确说是左向弱化，而非完全无响应。
- 两独立PPO从相同D2 actor/normalization/std开始，各新critic/Adam。seed3901完成96批73728真实决策，三次开发严重退化后按规则停，选batch64；seed3902完成64批49152决策，选batch16。共122880决策，不是监督更新次数。两个开发best在封存前锁定。
- **新±9°一次封存：**PPO3901实际torso+8.48/−11.11°，相对匹配无力增量+14.03/−5.78°；PPO3902+7.68/−8.17°，增量+13.26/−2.80°。主要腰相对骨盆，不等于迈步转身；仍存在无力偏转。D0在新左失败，保留旧有限成功但不宣称普遍泛化。
- **新零需求：**D2误差1.644cm/负担2.649N/偏转−9.57°；PPO3901 .783cm/1.244N/−4.44°，PPO3902 .674/1.067/+.05°。均有实际收益，未清除所有无请求动作。
- **旧能力取舍：**开发PPO3901直行7.086→5.67cm，但新79cm直行为8.138cm/3.647N，劣于D2 7.548/3.484和T30 5.359/3.427。PPO3902新6.416/3.971，误差与力取舍；不能只看单指标。局部13cm仍附带约+12–13°转腰。
- **首个主要边界是组合反馈：**新19cm/−8°组合PPO1/2实际+4.04/+9.16°方向错误。连续27cm过程中加入−8°再撤销，PPO开始阶段移动/旋转偏晚，后续追赶至root约32–34cm，yaw需求撤销后仍偏转。误差13.284→7.802/7.569cm改善但负担仍约4.57N，不能称可靠过渡。覆盖组5.373cm/4.011N较好，但撤销后yaw不回正且左转弱化。
- 54封存回合、总414独立评价；无记录安全boundary，封存无支撑丢失/关节越界。直行/组合q_PD峰仍约38–51°每20ms，实际q峰约1.7–2.5°每5ms，二者不能混称。释放后只看自然恢复，不把旧anchor距离算任务失败。完整XYZ/力/支撑/滑移/恢复见报告数据。

## 当前研究判断与唯一下一建议

已有可执行左右能力，训练也能降低零需求负担；问题不再是是否能左转。一个模型对步态、转腰、进入/撤销的共同反馈仍不可靠，覆盖监督与PPO表现出明确取舍。奖励能给已知坏行为较差回报，但不能由此证明当前优化与保旧约束充分；没有证据将唯一原因归为数据不足或BFM不可表达。

**建议下一轮仅做真实组合PPO中raw-z保持 vs 同状态完整50Hz BFM action保持的有界对照**，可靠旧域保留动作，组合域仍按真实任务反馈学习。沿用已暴露条件为开发，另封存；不加方向状态机、不只再做3000离线监督。M39-B已获新授权，执行该匹配对照；不先蒸馏尚未统一有效的专家，T30/S0不替换。

## 精确身份、数据和视频

服务器项目`/mnt/data/temp_user/yi.gao/projects/compliance_BFM`，本轮结果根`bridge-v2-owned/runs/m39a-demand-20261104/`。

|身份|本轮相对根路径|SHA256|
|---|---|---|
|覆盖|A-train-05/job/coverage.pt|ad452ad5efb69bf57e7c121a83a62aef7f1143cc576e7839f39a6d13b1b322be|
|PPO3901 best64|B-ppo-03/job/best.pt|72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a|
|PPO3902 best16|C-ppo-04/job/best.pt|d4c2fedf27d0bdf3f7a4fa44e5863e99bb9acfa7039f690623a64c899aacc76d|

D2来源`m38c-history-20261103/C-train-05/job/D2.pt` SHA9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5；T30来源`stop-sensing-20261103/B1-train-02/job/best.pt` SHAc71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33；S0来源`direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt` SHA0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12。三来源相对项目runs根。BFM SHA33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。

PPO实际runtime6f1789269de53a428aa5f5383eff0adcf247c04d；准备修复/监督3a4ff2759c15a1d9b4289e3124d76ef6e5c4370e；封存/视频956b1cf，完整SHA见[resources](m39a_artifacts/resources.json)。最终文档commit由Issue39精确链接指向。个人长期分支gaoe05/research；原两PPO worktree固定，不回写运行源码。

[结果](m39a_artifacts/results.json) · [物理质量](m39a_artifacts/physics.json) · [接触对照](m39a_artifacts/contact-contrasts.json) · [学习曲线](m39a_artifacts/development-curves.png) · [模型与选择](m39a_artifacts/models.json)。原始qpos/权重仅服务器。

Mac：`~/Projects/compliance-bfm-diagnostics/videos/m39a-demand-allocation/`，4MP4+INDEX：零需求、同模型左右完整回合、连续组合失败、T30/新PPO直行取舍。真实qpos1×，SHA/时长/完整解码/抽帧核验，不为视频重跑。

## 失败、资源与接力

准备首轮29回合后JSON bool错误，复用后补9回合，共38，不计67。PPO3902在dev64保存平台决定时报同类错误：failed日志保留；独立重建本来应平台停止，无需重训。没有将异常退出伪改为正常完成，也没有用终检挑模型。3901 last保存回滚后best，candidate96才对应末次退化评估。

累计受监督job-wall15504.635秒（含失败/渲染），reward复算54.443秒；CPU离线分析/验证另保守占用1200秒预算余量而非冒充实测，仍小于24000。约8.5GiB/15GiB；3000监督/6000，122880决策/196608；未新增Student。所有owned任务退出，GPU释放；0/5/6他人任务不操作。

新会话先读本文件/纲领/报告/Issue，实时核查Git、SHA、ledger和GPU，不能按旧PID重启。main、共享BFM/PD/权重、SoftSONIC不改，无Bridge/watch、无真机。Library2026-10-28为历史快照。完成阶段后停止，下一研究建议不等于新授权。

## M39-B最终接力（本段优先于历史M39-A结果）

[Issue40](https://github.com/GaoE05/robot-research-bridge/issues/40) / [最终报告](M39B_RESULTS.md) / [预登记](M39B_PROTOCOL.md)。**2026-11-04完成训练、一次52回合封存、视频及Mac核验；停止，不自动新增训练。**

同起点M39-A best64，seed3911，各64×768=49,152真实RL决策，A446/B472接受更新，各3次预定开发退化回滚。A raw best64 SHA `cce7e0fedcd4bb3c6e821d933c06d29c96ebb12d4473a64cd03f5df072cd3aac`；B action best56 SHA `49cc22b8df7f08ea4b57125f5366a19e32856f9ec6d38d14f5aea7c4364427fe`。先开发选择再锁定一次终检，未按终检补训。

**已保留：**新±10.5°实际torso A+9.42/−11.15°、B+9.26/−11.13°；相对同刻自身无力增量约+14.5/−5.9°。有限转腰成立，不等于迈步转身。正常直行可执行：初始/A/B误差5.698/5.854/5.732cm、力3.509/3.592/3.454N。原T30/S0等不覆盖，新模型均是特权Expert，非新Student。

**主要边界：**正向连续组合全接触误差初始8.895→A7.376/B6.727cm，但进入阶段root仅移动0.62/0.52/0.34cm且torso仍反向。B撤销后仍接触的yaw RMS10.15°；反向组合B误差8.047→8.527cm。A新早/晚停止退化，局部请求仍附带+12°，无力约−7°偏置。动作保持没有全面优于raw。保持梯度中位数仅PPO的A0.293%/B0.402%，单seed弱保持实验不能否定该机制。T30同5N平移更及时但旋转差，不能简单归为BFM不能移动或5N唯一原因。

52新终检无安全boundary/支撑丢失/关节越界；q_PD峰与真实q变化严格分开。释放后自然收步不按旧anchor误差判失败。4条真实qpos1×MP4+INDEX已同步Mac `~/Projects/compliance-bfm-diagnostics/videos/m39b-action-retention/`，SHA/全解码/画面验证。四格左上初始、右上A、左下B、右下T30。

运行源码：训练`11b40db7897580916d55da0afaaea7e8ab646d33`，封存`568a8e5e664dd12d9c7b1c6ac60c75b189196d11`，渲染`c6d448cef4745d891570120a0c3e973ef47b7a1e`。服务器run根`bridge-v2-owned/runs/m39b-retention-20261104`；模型`A-ppo-04/job/best.pt`、`B-ppo-05/job/best.pt`；终检`C-final-06/job`；全分析`final-analysis`。源/模型/视频摘要见`m39b_artifacts`。原训练worktree不动。

成本：累计job-wall12,084.92秒，另实测CPU分析178.92秒，文档/本地验证保守占用600秒余量而非实测；仍低于24,000。约6.61GiB/15GiB；98,304RL决策、768校准、smoke副本4更新、0新Student。所有本阶段GPU进程结束，其他用户任务不动。

**唯一下一建议/待决策：**用户提出5N可能过小。保存轨迹组合约80%接触tick饱和，限幅前总弹簧需求RMS9.5–15.7N；但首次限幅在onset后1.7–1.94秒，特权误差/速度及辅助奖励仍可用，不能解释全部迟滞。建议独立冻结模型5/10/20N短程物理对照并配原生BFM/T30，区分限幅、主动反馈与被动拖动；已提问，**截至结项未获确认且未执行**。不默认20N是标准，不变更原5N正式结果，不自动继续PPO/Student。

新会话先读本文件、M39B_RESULTS、Issue40，再刷新Git、模型SHA、资源/ledger与用户新授权。Library旧快照不作实时依据；不按旧PID重启，无Bridge/watch，不改main/共享BFM/PD/权重，无真机。
