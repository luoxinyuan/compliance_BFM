# F1真实执行视频

Mac：`~/Projects/compliance-bfm-diagnostics/videos/f1-l1-l2-20261111/`。三条均已同步、校验SHA，完整解码；25fps、75帧、1×真实qpos，完整3s诊断窗。没有将候选预测、reference或q_PD当真实运动；不声称完整回合或实时计算。每次100ms控制的离线搜索中位耗时17.75s。

|文件|上下画面|解释|
|---|---|---|
|f1-p1-13-feedback.mp4|固定BC500 / 因果特权重规划|同状态同条件，跟随和力小幅改善，yaw略增；两者均过窗口参考|
|f1-stop-continue.mp4|停止需求 / 继续需求，均同一因果planner|同前缀后真实分叉；停止有效，继续分支跟随与滑移失配。两行不是相同需求，不作为同条件优劣比较|
|f1-tail-hold-release.mp4|固定BC500 / 因果特权重规划|同状态同条件，减力明显，但yaw/滑移代价保留；覆盖保持到释放，释放后不考核旧anchor|

每行自身左右视角为同一条实际轨迹，字幕含时刻、阶段、真实XYZ力、接触误差及pelvis/torso yaw；黄箭头为力。中帧图已人工查看。SHA与源轨迹身份见[f1_video_manifest](f1_video_manifest.json)。不剪除窗口内失败，不拼不同checkpoint为统一策略。

## L3匹配C-P/O-F实际执行（2026-11-12）

本地 `~/Projects/compliance-bfm-diagnostics/videos/f1-l3-20261112/`：f1-l3-stop.mp4、f1-l3-continue.mp4、f1-l3-tail.mp4。每条上下同条件C-P/O-F，实际qpos三秒25fps、1×，不是搜索预测动画。服务器runs/video-l3-01；全帧解码核验。视频短窗不能冒充完整回合。
