# CURRENT_STATE — Issue57 D1完成，D2更新恢复仍阻塞

最后核对2026-11-10。[Issue57](https://github.com/GaoE05/robot-research-bridge/issues/57) ·
[总报告](D1_D2_RESULTS.md) · [D1](D1_FROZEN_BATCH48_RESULTS.md) · [D2](D2_FB_STOP_RECOVERY_RESULTS.md)。
本轮测试停止，所有本任务GPU进程退出，核验GPU0=0MiB/0%。无排队训练或自动重启。
Issue57保留未关闭追踪D2；**长跑禁止，下一诊断仅建议、未启动**。

## 保留的结论与模型

旧#56已结项；两KL各98304决策，均按开发规则选iteration0，BC500不替换。
FB旧最后完整落盘33792 transitions/512 updates后提前停，下一部分采样计数未知。
旧512更新后model/六Adam/replay丢失，只有step0；新包不能补回。
[旧结项](DUAL_TRACK_RESULTS.md)、[batch48](DUAL_TRACK_BATCH48_RESULTS.md)、
[早停](DUAL_TRACK_FB_EARLY_STOP.md)、[原保存erratum](DUAL_TRACK_FB_CHECKPOINT_ERRATUM.md)保留。

BC500：a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e。
原BFM：33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
KL.02 iteration0：1fa6e0703bc8e5bb9e06ec34cf72d7ecab016c672cc5899a83e5ec8626763332。
batch48：1c136654af650e60e8d5deb7fedef40818af698de2cb2df4a01390b76360f9f7。
末轮、T30/S0/M38等历史全部保留，不按任务拼模型。

## D1：新物理证据

- 三个已暴露父组×两模型，完整轨迹精确复现旧误差/力/饱和（差0）。
- 退化standing62.07cm/.22：4.4s同初态约.095°同状态q_PD差；5.08s各自实际q_PD差
  持续>1°，早于5.2s接触；随后支撑与滑移分叉。
- 24条真实完整前缀接管：初始化5.3s→48改善跟随但增滑移；反向仍不能恢复。
  不当作全部质量合格标签。2812实际BFM子步owner动作复算差0。
- 564实际开发输入对36864训练输入分块核查，不再用训练近邻代替开发状态重叠。
  没有证明唯一梯度冲突、覆盖不足或critic错误。

## D2：保存部分通过，更新门未过

旧198条完整边界：194左踝roll/4右踝roll，均实际越界>.02rad；143次接触前，
104次交接后100ms内，不是198次摔倒。不放宽边界。
新runtime实现完整不可覆盖停止包、rolling/块内游标、hash与原子发布，
包含online/targets、六Adam、normalizers、z_buffer、replay、全部RNG及真实物理历史。
新进程读回与继续27步采样逐值一致；后续1次完整FB更新model/Adam有可重复差异，
最大model差2.384e−7。确定性设置、统一factory均未解决，严格验证未过，不能长跑。
CPU篡改/半写入/覆盖/信号/外部stop返回gate检查通过；eager测试不覆盖compiled恢复。
[本轮验证erratum](D1_D2_VALIDATION_ERRATUM.md)保留三次失败与未关闭项。

## 成本、代码与接力

单RTX PRO5000 GPU0，原BFM/PD/10N/K80D8/频率/reward不变。
正式PPO/Student/1M/5M均0；D2新采样2209、完整工程update共7。
D1受控49312物理tick/12328 native控制，源前缀另计。
本轮保守计费3226.80s（含900s准备预留）/12000s，唯一输出约27.10GiB/48GiB。
所有失败计入；相同新包仅hash核验后hardlink去重，原字节/身份保留。
无精确退出戳的job显式采用最后自身产物+清理预留，详见账本，不冒称实际精确耗时。

私有stage：experiments/dual-track-diagnostics-20261110；repo固定D1 c99d576，
fb-runtime为独立工程/文档集成，实际运行完整SHA见总报告。runs与ledger分账。
旧正式runtime147b22b始终不动。公开长期分支gaoe05/research，文档HEAD不等于runtime。
本轮无新MP4，交付真实物理诊断图；历史Mac视频不动。

下一单一干预：D1 source/precontact稳定域函数保持；D2固定batch拆六优化器定位数值分叉。
两者均未启动。本轮不是多接触共享Student训练，历史有限Teacher→ideal-force Student成果保留，
不等于原生来源→可靠神经Teacher→受限Student新链已完整建立。

新会话先读OVERVIEW、本页、总报告、Issue57与Master18，实时核验资源/进程/权限。
不沿用旧PID，不改main、共享BFM/PD/权重、SoftSONIC、他人任务，不真机。
旧Library2026-10-28仅历史快照；本轮不自动下一阶段。
