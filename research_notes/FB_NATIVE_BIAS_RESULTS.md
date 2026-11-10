# 三个小工作包结项：更新分叉、无接触能力、B均值

2026-11-10，起点 `55cfe85928365ed4933c40a4b62ebc18e6324e12`，
事实基准为 D1_D2_RESULTS、D2_FB_STOP_RECOVERY_RESULTS 和获批合同。
三个包已完成并停止；**正式PPO/Student/FB长跑均0**。
未改reward、物理边界、概率断言、容差、共享BFM/PD/权重或main。
旧结项、所有失败与erratum保留。

|工作包|实际结果|仍不能宣称|
|---|---|---|
|[一：更新分叉](FB_UPDATE_FIRST_DIFFERENCE.md)|相同真实batch/RNG下，首个更新数值分叉定位到discriminator反向FP32加法累加；改变三个真实梯度分量的括号顺序精确复现两条最终梯度SHA|恢复已修复、严格一致性通过、可以重启1M|
|[二：原生能力](NATIVE_NO_CONTACT_INVENTORY.md)|12个真实无外力实例完整执行；双向小转腰、抬臂保持、下沉/恢复可用；motion能推进/停住但偏航/滑移质量不足，可靠后退未找到|所有native prompt均可调用、左右迈步转向/低速接触已合格、BFM不可能后走|
|[三：B均值](B_MEAN_BIAS_DIAGNOSTIC.md)|均值norm1.001 vs单状态B norm16；reward z公共方向明显，中心化角度中位74.9→97.0°；goal间角度几乎不变|均值已解释prompting全部失败、中心化会改善物理控制|

## 包一关键证据

参考和恢复各5443个算子，最终分别复现旧不间断/恢复model和Adam tree digest。
前向相同；反向索引270首先出现调度顺序区别，不能误称同张量数值错误。
discriminator pre-step四个梯度张量不同，F/B/auxiliary自身梯度差0；
后续critic77、actor46个不同。

保存真实梯度操作数后：`((A+C)+B)`与`((A+B)+C)`精确匹配两条首层最终梯度SHA，
最大梯度差4.768e−7、182830/801792元素不同。
这不是重训/修复对照，原严格恢复门继续阻塞；也不能据微小差异推断行为质量崩溃。
为何Autograd调度不同的更深执行状态来源尚未完全定位。

## 包二能力范围

上半身左右腰相对pelvis约+10.23/−10.21°，随后回中性；
下沉root持续均值.634m，standing .779m；手臂抬升使腕相对肩高约8.48cm并保持。
前进motion真实root前移3.068m，持续vx .424m/s、之后能停；
但slip峰.811m/s、持续末pelvis−22.53°，不称低速接触质量合格。
后退reward仅持续−.0138m/s，速度goal仅−.0010m/s，有限候选未找到可靠后走。
左右motion不对称，不能拿左右文件名代替方向实证。

无接触意味着**外部Tether=0**，不是没有足底地面接触。
所有轨迹都由原生真实前缀接上原BFM/PD；目标q仅供私人goal推断，未qpos-only摆姿。
“完整执行且无越界”与“目标行为/质量合格”分别报告，负例全部保留。

## 包三解释边界

7个机器人状态reward、5个goal；30k样本/29898不同replay索引，非独立回合数。
reward方向对均值的平方投影14.84%–61.68%，goal只有.191%–.525%。
原reward z彼此已经明显不同，不是完全坍缩。中心化是分布相关的离线诊断，
**中心化z没有进入任何真实控制/训练或正式reward inference**。

## 精确身份与账本

- 原BFM SHA：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- 原replay SHA：`46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b`。
- 参考trace runtime：`a1834af33ac136bbac046dfed26448b43656c670`。
- 首次恢复trace：`570cbf002c770ccb3db61453974a1a3b4972f429`。
- 捕获操作数 / 原生盘点 / B诊断：`aa62fe541037bb6a59f453fe4b9c366e10977523`。
- 私有stage：`experiments/fb-native-audit-20261110`；每job独立PID/目录/日志/结束戳，
  `trace-analysis`为CPU离线算术，不是新增agent.update。

|成本|实际|
|---|---:|
|5个受监督GPU job墙钟合计|1371.82s|
|准备/未单独计时CPU分析保守预留|900s（不是实测CPU job时间）|
|保守计费|2271.82s / 7200s|
|新增私有输出（汇总前快照）|约.256GiB / 12GiB；最终核验另见ledger|
|完整工程agent.update|4；到本轮自登记上限|
|新增工程native transition|1118；重建前缀另计、非新学习样本|
|原生盘点受控5ms tick|31200；reset/prefill7200＋实际源suffix2400另计|
|离线B-state evaluation|240005（含官方API重复计算），30k sampled rows|
|正式PPO / Student / FB长跑|0 / 0 / 0|

运行前确认单RTX PRO5000 GPU0独立可用，CPU cgroup210核、MemAvailable约992GiB，
I/O压力低；共享盘约77GiB可用，故只保存小型新诊断，不复制全模型包。
其他GPU/用户任务未操作。编译/导入/IO/实际重建均包含job墙钟，不拿纯更新耗时代替总成本。
所有五job完成退出；完成后另核GPU0释放。原始数据/latent/权重不上传。

[成本/runtime](fb_native_bias_artifacts/cost-runtime.json) ·
[SHA/来源](fb_native_bias_artifacts/identities.json) ·
[预登记协议](fb_native_bias_artifacts/protocol.json) ·
[真实物理图](fb_native_bias_artifacts/native-actual-timeseries.png)。

## 接力与停止

Issue57继续追踪**尚未修复**的严格更新恢复门，本轮诊断任务完成。
下一单一主要工程干预建议是固定discriminator梯度累加顺序，再做精确更新回归；
原生运动质量/后退来源与表征诊断分别保留，不用它们替代工程修复，
也不因工程阻塞否定已验证的小腰转/下沉/抬臂能力。
未执行修复、中心化物理A/B或下一训练阶段；不自动重启1M/5M。
