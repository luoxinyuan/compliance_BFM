# M39-A：看这4条就够了

全部是保存的真实MuJoCo qpos，正常1×，不是目标motion或q_PD动画；没有为视频重跑/挑seed。每段从ready状态t≈3s开始，含接触前、完整接触和解除后5s恢复。先前站立初始化不重复播放。新封存条件：onset4.4s、ramp8.1s、release14.5s、结束19.5s，总力上限5N。

1. **01-zero-demand.mp4（16.52s）**：上D2，下PPO seed3901开发最佳batch64；两点连接但伙伴不移动/不转。看不必要偏转与牵拉是否减小。负担RMS 2.649→1.244N，误差1.644→.783cm；仍不是完全无动作。
2. **02-left-and-right-9deg.mp4（33.04s）**：前半+9°、后半−9°，两段均完整独立回合。四格始终为左上D2、右上覆盖监督、左下PPO3901 best64、右下PPO3902 best16；没有按方向换最佳模型。PPO两者保住左右转腰；覆盖监督左向减弱。
3. **03-translation-rotation-return.mp4（16.52s）**：同四模型，躯干双点前移27cm，过程中加入−8°再撤销旋转、继续平移。展示真实取舍/失败：位置跟随改善不代表转向与恢复正确。误差D2/覆盖/PPO1/PPO2 =13.284/5.373/7.802/7.569cm，负担4.579/4.011/4.576/4.567N。
4. **04-straight-regression.mp4（16.52s）**：上冻结T30，下PPO3901 best64，右腕新79cm直行。T30 5.359cm/3.427N，新PPO8.138cm/3.647N；保留旧技能的反例，不能只看开发直行改善。

画面：青C=真实接触点，红A=伙伴anchor，黄箭头=真实力，绿=目标朝向，蓝=pelvis朝向/root轨迹。左右双点逐点XYZ力、净yaw moment实时显示。角度是世界绝对角；报告中的转角是相对ready状态，两者零点不同。解除后标为non-task geometry，此时anchor距离不计任务失败。四格使用固定俯视，双模型采用侧视+俯视；不改相机来隐藏失误。

## 身份与可复现入口

阶段Issue：https://github.com/GaoE05/robot-research-bridge/issues/39
服务器真实轨迹根：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m39a-demand-20261104/`，封存`A-final-08/job`、`B-final-07/job`、`C-final-06/job`。`videos/manifest.json`记录各片的实际输入轨迹与SHA。封存/渲染runtime为`956b1cf`（精确SHA见阶段报告）；训练runtime `6f1789269de53a428aa5f5383eff0adcf247c04d`。

- D2：9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5
- 覆盖监督：ad452ad5efb69bf57e7c121a83a62aef7f1143cc576e7839f39a6d13b1b322be
- PPO3901 best64：72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a
- PPO3902 best16：d4c2fedf27d0bdf3f7a4fa44e5863e99bb9acfa7039f690623a64c899aacc76d
- T30：c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33

这些均为特权Expert，不是本轮训练了新Student；不宣称迈步转身已成功。两PPO在看封存结果之前按开发规则固定，封存后没有重选或补训。

本地文件均非空、25fps，ffprobe时长与预期一致，完整ffmpeg解码通过。源文件复制SHA一致，左右视频只无损串联同一组模型的完整回合；抽帧检查已确认接触点、力、足部和模型标签可见。
