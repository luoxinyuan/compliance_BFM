# M9C 本地视频交付索引

全部为已保存真实 MuJoCo qpos/qvel 离线重放，25fps、正常1×；不是参考 motion、q_PD动画或新挑seed。每段包含固定世界斜视和俯视，保留完整记录的交互和约6–8秒结束恢复。轨迹约从物理t=3s准备完成后开始，前置准备未重新模拟。旧视频的clean/与根目录渲染历史保留。

优先观看：

- [原专家 vs 原M9B Student，同一开发直行](final/original_expert_vs_old_student.mp4)：上方专家，下方Student。
- [新学习型特权专家 vs 新Student，同一封存直行](final/new_expert_vs_new_student.mp4)：显示Student仍有高受力边界。
- [旧 vs 新Student，小幅腕部退化](failures/old_vs_new_wrist_regression.mp4)：上方旧版，下方新版；新版不应替换已有共享策略。
- [原有限左弯正例](final/held_left__student.mp4)、[原位置跟上但heading错误的右弯反例](final/held_right__student.mp4)。

C青色=实际接触点，A红色=伙伴anchor，黄色=实际施加三维力（视觉尺度0.065m/N），蓝色=root轨迹及heading；显示足部法向载荷。力记录是刚完成的5ms积分所施加的wrench，画面为积分后qpos。渲染器只在内存中隐藏原XML静态调试site，没有修改资产或真实物理。单视频1280×640，并排比较1280×1280。

下表RMS只计真实接触窗口；不把释放后零力当改善。各项condition/source/source_sha256/video_sha256完整保存在对应manifest.json。所有14个单视频、3个合成对照均完成非空、SHA/来源、时长与全文件ffmpeg解码验证；另做了新专家、躯干与退化例截图目视检查。

|视频|误差cm / 力RMS N|条件与已知边界|模型来源|
|---|---:|---|---|
|[held_straight__expert.mp4](final/held_straight__expert.mp4)|13.643 / 4.581|完整记录，详见阶段报告; onset=4.25, ramp=7.8, release=14.05|original|
|[held_straight__student.mp4](final/held_straight__student.mp4)|12.235 / 4.380|完整记录，详见阶段报告; onset=4.25, ramp=7.8, release=14.05|old-memory|
|[held_left__student.mp4](final/held_left__student.mp4)|10.989 / 3.640|有限左弯，不代表普遍左转; onset=4.55, ramp=8.0, release=14.55|old-memory|
|[held_right__student.mp4](final/held_right__student.mp4)|13.614 / 4.703|旧右弯heading方向错误; onset=4.65, ramp=7.9, release=14.55|old-memory|
|[down_test__student.mp4](final/down_test__student.mp4)|2.775 / 2.259|完整记录，详见阶段报告; onset=6.6, ramp=3.5, release=12.1|old-memory|
|[held_straight__arm.mp4](final/held_straight__arm.mp4)|10.760 / 3.763|直接关节正控制，非latent-only Student; onset=4.25, ramp=7.8, release=14.05|original|
|[held_straight__projected.mp4](final/held_straight__projected.mp4)|10.162 / 3.837|合法z投影物理执行，heading/连续性仍差; onset=4.25, ramp=7.8, release=14.05|original|
|[development_straight__rl.mp4](final/development_straight__rl.mp4)|12.436 / 3.708|开发条件；不冒充留出结果; onset=4.25, ramp=7.8, release=14.05|rl|
|[new_straight__expert.mp4](final/new_straight__expert.mp4)|13.035 / 3.776|完整记录，详见阶段报告; onset=4.4, ramp=8.4, release=14.8|original|
|[new_straight__learned_expert.mp4](final/new_straight__learned_expert.mp4)|13.543 / 3.761|完整记录，详见阶段报告; onset=4.4, ramp=8.4, release=14.8|rl|
|[new_straight__transfer.mp4](final/new_straight__transfer.mp4)|12.272 / 4.752|力4.752N/饱和86.9%，未保留专家降力; onset=4.4, ramp=8.4, release=14.8|transfer|
|[new_right__transfer.mp4](final/new_right__transfer.mp4)|9.293 / 3.633|跟随改善但q_PD峰值69.75°; onset=4.6, ramp=8.4, release=15.0|transfer|
|[wrist_early__ablation.mp4](failures/wrist_early__ablation.mp4)|1.945 / 1.586|完整记录，详见阶段报告; onset=5, ramp=4.0, release=12.0|old-no-memory|
|[wrist_early__transfer.mp4](failures/wrist_early__transfer.mp4)|9.173 / 4.185|小任务误调用行走，root约+30cm; onset=5, ramp=4.0, release=12.0|transfer|

## 模型与数据

- original: `frozen BFM + native motion + handwritten feedback`
- old-memory: `57b346158c9ce1bbc18dc0be59991c5f4710bd4077746bf913d4a0c4d3f12634`
- old-no-memory: `ad81e5736f165353a3d45afa3dc5f8a11b76c153f4ac7ddc6db5660f8dcbac78`
- rl: `7f45da6785e49d34851a1236e1cc66898455adb43527494c658e5ff8c3efb4cf`
- transfer: `01ba43aa305496ac63bd25dba621920c48d1b2d8be2a977fdefc0bf940949f1a`

BFM SHA256: `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。新专家是本轮实际训练的特权高层residual；Student部署不输入anchor、未来轨迹或专家相位。M9C服务器数据目录：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m9c-expert-20261101/`；旧5条物理来源为`m9-feedback-continuation-20261101B/final-attempt-01/outputs/`。源码分支`gaoe05/m9c-expert`。本地原始SHA验证见`final/local-validation.json`、`failures/local-validation.json`及`paired-validation.json`。
