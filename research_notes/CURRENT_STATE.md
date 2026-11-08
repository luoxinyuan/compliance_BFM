# CURRENT_STATE — Issue52：P4000热启动Expert真实反馈PPO（准备中）

2026-11-08新授权优先于下方Issue51下一建议。[Issue52](https://github.com/GaoE05/robot-research-bridge/issues/52)。主线回到特权Expert真实物理PPO；BC仅初始化，不要求受限Student通过。两个seed5201/5202，同P4000 actor/normalizer，新critic/Adam；BFM/PD/Direct256D/1088D/10N/K80D8/reward不变，无在线I/C0/C32/搜索覆盖。

stage `experiments/p4000-feedback-ppo-20261108`，新ledger18,000job秒/10GiB/98,304RL决策；两seed先各32×768，共同开发规则满足且预算允许才共同延至64。预登记见p4000_protocol.json；3个采样worker顺序使用同单卡，每个进程明确记录。无Student训练、reward-z扫描或未来输入。必要均值/采样和同状态分支检查后直接PPO，不恢复已结项Issue51。当前仅准备，是否启动和runtime以新ledger为准；GPU已只读核对空闲，另有资产下载进程不动。

P4000 SHA256 `9f560903072b355e6d2af537af725b9e61210ecea87a56151aa5d59dff45d164`；原Issue51的S失败/旧历史S0成功完整保留。阶段中源代码保持固定，文档另独立worktree；下一会话先看新ledger，不按旧PID重启。

---
以下为Issue51结项历史；其中后续建议已由本轮新授权覆盖，不是当前主线。

# CURRENT_STATE — Issue51已结项：Teacher扩域有效，特权BC部分有效，受限BC失败

最后核对2026-11-08。[Issue51](https://github.com/GaoE05/robot-research-bridge/issues/51) · [结果](NATIVE_TRANSFER_RESULTS.md) · [方法/runtime](NATIVE_TRANSFER_METHODS.md) · [精简证据](native_transfer_evidence.json) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。当前无训练、无待自动发车任务；建议不等于新授权。

## 本轮实际结果

冻结BFM/PD、10N/K80D8、7提议/2.4s因果滚动搜索Teacher，在4个新完整回合：8cm2.551cm/2.207N，72cm平均.22m/s7.665/5.714，新66cm继续5.957/4.697、停止6.346/4.964。数值仅接触期3D RMSE/实际力RMS。短请求腕相对肩X+9.40cm、rootX+1.44cm；长请求有真实脚步。不是reward-z成功或新PPO Teacher；搜索仍约23秒/100ms命令，偏航/q_PD峰及严重超前恢复边界仍保存。

fresh特权BC P4000，在新66cm继续8.004/5.480、停止8.350/5.559，优于旧I但落后Teacher；旧已见正常牵引12.306/7.939，较I退化。P有特权，不能称受限Student成功。

fresh受限ideal-force BC正式开发选回**新S-init/step0**，不是训练成功，更不是历史S0。固定S4000已见正常55.219cm/9.218N、开发58.270/9.245，没正常跟上。已见轨迹持续执行z差异4.24s出现，早于5.0s接触；Teacher状态上平均拟合低，但关键步态动作有尾部误差，自身历史越走越偏。该时序不是同状态因果证明，尚未证明信息不足或force太小。单轨P/S各1000更新仅诊断；S能前进约35cm但25.369/9.079，仍不合格，不重选正式模型。

## 科学瓶颈与下一决策

优先解决受限Student接触前自身历史闭环漂移及关键过渡命令拟合/覆盖。建议少量完整真实访问状态Teacher接管，界定可靠域，再做有限数据/历史反馈修复及冻结BC vs同初始化自身RL。**尚未启动或授权新PPO/大规模Student**；不再以增加reward候选/C0修补代替当前问题。P缺少搜索内部真实phase/上一计划，因此P/S差距也不能全归为观测权限。训练仅3整回合/2独立前缀族，单seed，不宣称general。

## 权限、数据与成本

P1088D；S1639D=465原BFM本体/动作历史＋256实际上一z＋900真实力历史＋18已知安装；不含anchor/error、伙伴真值速度、目标yaw、time、任务ID、active真值或搜索kind/phase。原obs保存的privileged_state/time未拼入S。ideal-force仍非已验证G1实机传感。

642真实首命令查询→641完整有效标签；预测标签0。train311，dev104，新prefix final226；按整回合/前缀族划分。真实重放输入/上一z/qpos/qvel/q_PD全0误差。P/S主各4000更新＋单轨诊断各1000=10000；RL决策0。419搜索、1,407,840候选physics步、8376实际Teacher步；累计job12,622.77秒（含300秒CPU预留），新增实验输出约1.02GiB，低于18000秒/10GiB。所有正式回合无安全提前终止，不代表质量问题消失。

## 模型、源码、服务器与视频

服务器项目 `/mnt/data/temp_user/yi.gao/projects/compliance_BFM`；stage `experiments/native-transfer-20261107`；完整ledger、私有权重、raw轨迹均保留。后端SSH1016，CUDA9.0/132SM/~80GiB，NVML L20Z名称冲突保留；不将旧空闲快照当下次资源许可。

- Teacher物理runtime `443f02c4fc22fd84e59f2cee4e25750d838fe527`；重放/主BC/评价 `0b5135574719f41d888f9f77fcf2d32e951868ca`。
- 冻结/单轨诊断runtime `d800eb0b90d00526053cc983b33abb39ba6e5d26`；最终分析/视频 `73c437e81f2aca617bc432ddf859dddd2a959b4e`。文档提交不是物理runtime。
- P `runs/bc-fit-01/P-4000.pt` SHA256 `9f560903072b355e6d2af537af725b9e61210ecea87a56151aa5d59dff45d164`。
- S4000 `runs/bc-fit-01/S-4000.pt` SHA256 `8f5e7287e974deedba1a8d841dd65d7692e944e278bd4a091f1ec224f167a479`；新正式S-init `S-0.pt` `f414ea3fd12a4b90ae8a03ad273672dfd6c18e3d848a6d6e2e0efde5384561ba`。
- 冻结BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- 长期个人审查分支 `gaoe05/research`，本轮结项commit见Issue51；精准运行SHA以上表为准，绝不以文档HEAD替代。
- Mac实际目录 `~/Projects/compliance-bfm-diagnostics/videos/native-teacher-transfer/`：**仅3个MP4＋INDEX**。01短请求、02较快长牵引（Teacher/I/P4000/S4000），03新共同前缀Teacher及同P4000的继续/停止；最后一条不是受限Student成功。SHA/正常1×时长/全解码/画面已核对，详见native_transfer_video_verification.json。

## 保留历史与接力

旧有限域Teacher→ideal-force Student迁移成功不被本轮fresh S失败抹去。保留T30/S0、M8–M10、M38双向转腰、I/C0/D2及所有失败；旧I/C0仅冻结参照，不是本轮actor初始化或在线输入。Issue50/49/47不恢复；M11初轮与continuation不同，A160/B114提前停止不代表等预算结论。

新会话先读本页、OVERVIEW、Issue51和实际ledger，确认模型/环境身份与最新用户授权，不重跑已完成任务。只在个人worktree维护，Git无权重/raw数据/视频/凭据；不改团队main、共享BFM/PD/权重、SoftSONIC/他人任务，不做真机。Library2026-10-28为历史快照。
