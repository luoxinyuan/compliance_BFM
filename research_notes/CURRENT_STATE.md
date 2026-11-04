# CURRENT_STATE — 新会话先读

最后核对：2026-11-04。**M39-A完成；M39-B已获新授权，校准与实际PPO更新测试通过，两臂训练进程已启动，初始16条件开发及首批真实PPO更新已通过，正在训练；尚无M39-B科研结果。** M38-C有限域双向转腰成立；M39-A进一步降低零需求牵拉，但连续平移/旋转需求分配与旧能力泛化仍未统一解决。不得回到重复BFM左转资格，也不自动续训。

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

## M39-B active handoff

[Issue40](https://github.com/GaoE05/robot-research-bridge/issues/40) / [预登记](M39B_PROTOCOL.md)。共同初始化为上表PPO3901 best64，actor/critic/normalization/std完全继承，两臂fresh Adam。seed3911，各64×768；原奖励/物理不改，56%组合课程共同改变。A raw保持，B同状态5子步q_PD保持；正式训练前固定梯度标定和新封存组合。准备worktree `bridge-v2-owned/trees/m39b-action-retention/repo`；新run根`bridge-v2-owned/runs/m39b-retention-20261104`。本段优先于历史完成段的无任务状态。运行源码 `11b40db7897580916d55da0afaaea7e8ab646d33`。A raw：GPU1 / supervisor2280380 / child2280416 / `A-ppo-04/job`；B action：GPU2 / supervisor2280381 / child2280420 / `B-ppo-05/job`。各3worker，CPU8–15 /24–31，每臂墙钟预留8500秒；达到64批自动停止。启动/校准共约399.7秒，768物理校准决策，smoke仅副本4优化步，无新正式训练结果。数据身份首轮失败保留。校准B系数4845.037013683694（qPD rad²），A .002；初始梯度规模约PPO的0.2–0.6%，不可先宣称强保旧。实时状态请读新账本，不根据本快照重复启动。新上限24000累计job-wall、196608决策、15GiB；实际运行身份由新账本/Issue记录，不能沿用旧PID。

M39-B首批核验：两臂各768真实PPO决策；16初始开发条件核心指标差为0，直行3212物理tick的qpos/qvel/force逐点差为0。raw/action实际类型正确，初始tensor摘要一致；每批采样约51.5秒，优化0.19/0.22秒。预计2026-11-04北京时间约13:45–14:00完成训练（以实际账本为准），随后仍需一次封存评价/真实视频和阶段结论，不把进行中当成功。保留两臂共同比较，不因终检给单臂加训。
