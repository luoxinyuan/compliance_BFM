# E/I v2 — 本轮三条实际qpos视频已交付

2026-11-17。按Yi指定位置交付到Mac的compliance-bfm-diagnostics/videos/ei-v2-dev-20261117。不是新的物理采样，不改变既定Dev或模型。

1. 01-near-FOLLOW-four-methods.mp4：同小请求，左上E、右上I、左下Goal-only、右下冻结原生standing。
2. 02-mid-FOLLOW-tracking-shortfall.mp4：同中距FOLLOW与相同四格；保留跟随不足的负例。
3. 03-common-prefix-STOP-versus-CONTINUE.mp4：上排E、下排I，左停右继续；每排固定同一终点checkpoint及同父组共同前缀，公开Goal同为HOLD→FOLLOW。

全部是存档实际qpos、25fps、1×，拼接同回合已存原生源前缀与完整任务/4s真实release恢复。记录从t≈3s开始，未记录的0–3s共同初始化不虚构补帧；短分支结束后保持其最后一帧并标注。真实力/anchor/腕部、当前公开Goal与实际命令显示；released后anchor几何误差明确是non-task geometry。arm15是原生候选来源名，不保证实际移动15cm。

源角色actual_frozen_policy_independent_dev_not_training；终点模型、physics、reward、判据不变。严格复用本轮12个实际回合，没有搜索预测动画或重新采样。

## 验收与修复

本地三视频与远端SHA逐个一致；全部帧在远端和Mac完整解码，ffprobe核验1280×1280、25fps、268/300/290帧及10.72/12.00/11.60秒。中段截图已实际视觉检查，保留头部到足部全身和可读HUD。精确video SHA/来源见contact_goal_v2_video_delivery.json和contact_goal_v2_video_local_verified.json。

第一次渲染因eval隔离目录未含可视化XML退出，0新物理，62.541814秒；失败与导出的真实源全保留。修复为显式只读训练runtime中的同一XML，与原资产SHA相同，并在渲染前校验29D关节顺序。第二次92.101939秒完整完成；视觉检查后仅将camera lookat高度0.60→0.82m、把伙伴phase和公开Goal分栏，第三次34.904324秒复用同SHA导出源完成。两版源片段与旧视频均保留，无科学判据修订。

合计189.548077job秒，物理/训练新增均0；已进入原总账，当前新增stage38895.018698/56000job秒。源码f9a5119700b258fe306f1b08a6b0b7c7aa965d9a，原训练/评价runtime保持。最终GPU0空闲，锁释放。

本地INDEX说明每格、Goal语义与失败边界。视频不将本轮未达行为门重判为成功；完整分阶段科学结项另按原合同继续处理。
