# CURRENT_STATE — 新会话先读

最后核对：2026-11-05 CST。本轮Issue43已获授权：初始化去偏/reward审查，先来源资格，匹配PPO有条件执行（最多49152决策、16000job秒、10GiB）。当前尚未启动PPO；Issue42以下为已完成历史。

**Arm-to-Locomotion任务合理性审查（Issue42）完成并停止；43个新冻结物理回合，0 PPO/Student训练。没有后台或下一轮训练排队。**

入口：[本轮结果](ARM_TO_LOCOMOTION_RESULTS.md) · [M41启动因果结果](M41_STARTUP_CAUSAL_RESULTS.md) · [原Transition课程](TRANSITION_RESULTS.md) · [Issue42](https://github.com/GaoE05/robot-research-bridge/issues/42) · [Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。纲领见RESEARCH_OVERVIEW。旧Library2026-10-28只是历史快照。

## 大目标与保留成果

Contact-conditioned whole-body behavior selection/modulation/transition：局部接触与身体状态决定合理的手臂、躯干/root、支撑和步态响应，不以root走远或demo数量作为成功。有限域直行/停止Teacher→ideal-force Student、小幅双向转腰等旧成果保留；未证明general或真机精确力传感。

冻结BFM/原PD；I/T30/S0/D2、M41 A/B和全部失败不覆盖。本轮没有新checkpoint，不把原生goal资格播放称为新学习型反馈策略。

模型身份（runs前缀`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/`）：
- BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- I M39-A best64 `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`，runs/m39a-demand-20261104/B-ppo-03/job/best.pt。
- T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，runs/stop-sensing-20261103/B1-train-02/job/best.pt。
- S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`，runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt。
- D2 `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`，runs/m38c-history-20261103/C-train-05/job/D2.pt。
- A32/B32/其他节点身份见m41_causal_artifacts/identities.json；本轮直接分析保存轨迹，未重训或重新选模型。

## 本轮真实发现

1. 旧82cm任务确有先局部响应：I/T30腕相对肩2cm发生约onset后1.3/1.2s，脚明显移动约2.1/2.0s；当时目标已前移约13–14cm。初始anchor实际偏差约毫米及以下量级，不支持大预载冲击解释。
2. 当前肩固定、腕Y/Z保持时继续前伸的宽松几何上界约16.75cm。80cm需要身体参与；15cm已近边界，不应强制不走。IK只是独立几何参考，未接真实控制。
3. BFM通过锚定原生goal可局部伸臂：5cm/3s时Standing误差/力2.75cm/2.23N→goal1.42cm/1.15N，root约0、未明显调脚。无力时仍相对肩前伸2.97cm，证明主动动作来源。15cm仍有Y/Z和幅度不足，不宣称任意目标精确可达。
4. 同7.8s慢引导：5cm时I不调脚(2.15cm/1.74N)，T30调脚并前移约4.69cm(1.72cm/1.42N)；15cm时I不走但误差5.01cm，T30走约13.79cm、误差2.74cm。30/80cm均可行走。不同需求存在任务收益/运动代价，不能仅凭脚动判失败。
5. 同峰速.15m/s的短快请求明显更难；例如15cm误差I/T30为8.66/8.89cm。尚无证据证明连续距离/速度/可达性泛化。
6. Teacher不是force-only：有当前anchor error/relative velocity。关闭实际力但保留目标时未走出原距离；干净遮蔽几何而保持真实力时I误差16.66cm、T30 6.96cm。四组反事实前缀qpos/qvel与anchor差均0。它们是已训策略敏感性/OOD诊断，不代表无力Student理论不可学。

## M41仍然成立的边界

原Transition A/B各49152决策但开发best均iteration0，没有新学习胜出。A32/B32从standing失败；I先执行到6.5s再交失败模型仍能走，晚失败状态交I却不能救。单100ms替换不足，状态/动作历史积累重要。旧stand_walk实际决策约15–17%，奖励把失败评为更差，噪声有高方差救援但不可靠；未唯一定位优化根因。**本轮不为保住walking而压制合理局部响应，也不抹去历史退化。**

## 下一决策（未执行）

建议一次匹配reach-to-walk课程：同一I初始化、Direct/BFM/PD/5N/std/reward/预算，两臂共享旧域保持，比较旧长请求与近/中/远+有限真实源状态覆盖。完整32×768/臂建议、距离/速度分层、真实goal源前缀、预封组合与解释边界见本轮报告。源状态不能qpos-only重置，启动保持只用于其有效长请求域，不能把小请求强制学成walking。必要的少迈步偏好以后独立评价，不写force→motion规则。本轮不追加PPO/Student；0/1/2/4/8/16帧force-history消融仍未执行。

## 代码、数据、视频、接力

个人分支gaoe05/research；主runtime6ccfedf57b7bd7a546bbec9d86b69e0ccdc4f86f，补充runtime在reach_artifacts/ledger-02.json。最终源码/报告commit见Issue42链接。

run `runs/arm-to-locomotion-20261105`：physical-01/02真实qpos与完整控制、timelines、offline-ik、saved-local、逐点final-metrics、validation、ledger；旧run不变。43回合/102116suffix物理样本/25529 BFM控制步，另有准备/前缀；0训练。物理777.86+229.17s、视频27.39s，另有短CPU审查；约720MiB。全部owned进程结束，GPU5/6释放，他人任务未改。

Mac `~/Projects/compliance-bfm-diagnostics/videos/arm-to-locomotion/`仅3条1×真实qposMP4+INDEX：01小5cm I/T30差异；02Standing与原生伸臂；03大80cm先局部后脚步。非新训练，完整交互/保持/4s恢复；SHA/全解码/画面验证通过。

新会话先读本文件、纲领、Issue42；查实时资源与准确身份再按新授权行动。共享checkout仍原6023c6e clean；不改共享BFM/PD/权重/main、SoftSONIC及他人任务，不启动Bridge/watch、真机或未授权训练。
