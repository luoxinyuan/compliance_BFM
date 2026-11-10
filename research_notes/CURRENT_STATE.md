# CURRENT_STATE — 三包诊断完成，D2严格更新恢复尚未修复

最后核对2026-11-10。[Issue57](https://github.com/GaoE05/robot-research-bridge/issues/57) ·
[本轮总报告](FB_NATIVE_BIAS_RESULTS.md) · [更新分叉](FB_UPDATE_FIRST_DIFFERENCE.md) ·
[原生能力](NATIVE_NO_CONTACT_INVENTORY.md) · [B均值](B_MEAN_BIAS_DIAGNOSTIC.md)。
三个小包完成即停，无正式PPO/Student/FB长跑，无排队或自动重启。
Issue57仍打开追踪严格更新恢复；**定位不等于修复，不发长跑资格**。

## 当前保留的模型与旧结论

BC500仍为基线：a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e。
原BFM：33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
KL.02 iteration0：1fa6e0703bc8e5bb9e06ec34cf72d7ecab016c672cc5899a83e5ec8626763332；
batch48：1c136654af650e60e8d5deb7fedef40818af698de2cb2df4a01390b76360f9f7。
两KL各98304正式高层决策均开发选iteration0；没有新模型晋级。
T30/S0/M38与全部旧模型、失败保留，不按任务拼接。

旧FB最后完整日志33792 transitions/512 updates后按用户意图早停，
下一部分采样计数未知；512更新后权重/六Adam/replay已丢失，只有step0。
[旧结项](DUAL_TRACK_RESULTS.md)、[batch48](DUAL_TRACK_BATCH48_RESULTS.md)、
[FB早停](DUAL_TRACK_FB_EARLY_STOP.md)、[永久保存勘误](DUAL_TRACK_FB_CHECKPOINT_ERRATUM.md)保留。
新诊断不能补回旧丢失模型，更不能拿重训假冒原模型。

## 本轮三个实际工作包

1. **D2更新分叉**：同真实mini-batch/RNG和更新前状态，追踪参考/恢复各5443算子。
   前向一致，discriminator backward先改变调度；首层矩阵梯度的
   `((A+C)+B)`/`((A+B)+C)`分别精确复现两条最终梯度SHA，差最大4.768e−7。
   discriminator4个最终梯度不同，F/B/aux自身差0，随后critic77/actor46个不同。
   三次最终状态精确复现旧两条model/Adam不一致；只完成算子定位，未改原更新实现。
   为什么Autograd调度不同的更深执行状态尚未全部定位。
2. **无接触原生能力**：12候选完整原BFM/PD物理执行，外部实际force=0，
   双向小腰转约±10°、抬臂保持、下沉约14.5cm及回standing可用。
   motion能前进/停住，但slip峰.811m/s、偏航等质量代价仍在，不称低速接触合格。
   可靠持续后退、双向迈步转身在有限来源/调用范围未找到；不等于基座不可能。
3. **原B均值**：30k replay successor、7 robot-state reward/5 goal。
   μ norm1.001 vs B norm16；reward z公共方向明显但非坍缩，
   两两角中位74.9→97.0°（中心化诊断）；goal两两角约40.27→40.33°。
   中心化候选从未用于真实控制/训练；没有证明其解释prompting全部失败。

三个包分别报告，不用数值工程问题否定原生能力，也不用表征角度代替物理收益。
来源为原生接口/既有goal-motion资产，未在线调用I/C0/T30或继承新高层actor。

## D1/D2此前证据保持

[D1_D2_RESULTS](D1_D2_RESULTS.md)与[D2保存报告](D2_FB_STOP_RECOVERY_RESULTS.md)事实不变。
D1六轨迹精确复现、24接管、2812实际BFM子步差0；batch48站立退化有接触前历史传播，
5.3s初始化→48改善跟随但增滑移，反向恢复不可靠。564开发输入独立对36864训练输入核查。
未证明唯一梯度冲突/覆盖不足/critic原因。
D2旧198边界均真实踝roll越界>.02rad，143次接触前；不等于198次摔倒。
完整不可覆盖停止包、读回与继续27步采样逐值一致，更新精确一致仍失败。
[验证erratum](D1_D2_VALIDATION_ERRATUM.md)保留，不放宽边界/概率/恢复容差。

## 资源、身份与接力

本轮单RTX PRO5000 GPU0串行，runtime从55cfe85隔离；重作业均退出，无正式训练。
完整工程agent.update4次、新采样1118；原生受控31200 tick，源/reset另计；
离线B-state evaluation240005（含重复），完整原replay不改不下载。
5个job实测1371.82s＋准备/CPU分析保守预留900s＝2271.82s/7200s；
新增输出约.256GiB/12GiB，最终账本保留。旧轮成本不混入新轮。

私有stage `experiments/fb-native-audit-20261110`；运行代码：
参考a1834af，首次恢复570cbf0，操作数捕获/原生/均值aa62fe5。
精确runtime/SHA、协议、分包成本见总报告及fb_native_bias_artifacts。
个人长期gaoe05/research只集成获准源码/小型汇总；发布HEAD不等于runtime。
旧runtime/数据/模型原样保留；本轮无新MP4，交付实际物理曲线，不向Mac塞重复视频。

下一单一工程建议：只固定discriminator梯度累加顺序，复跑同batch新进程严格回归。
该修复、中心化物理A/B、PPO/Student/FB长跑均未启动，也没有自动启动授权。
新会话先读OVERVIEW、本页、本轮总报告、Issue57/Master；实时核验资源。
不改main、共享BFM/PD/权重、SoftSONIC、他人任务，不真机。
旧Library2026-10-28仅历史快照。
