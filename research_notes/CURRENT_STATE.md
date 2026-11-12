# CURRENT_STATE — Issue62 Pilot 已停止：A物理硬界＋监督器故障

2026-11-12。T0三门通过结论保持，用户已接受；每臂3,000提案被否决，原草案另存。此次仅完成获准工程准备，没有启动Pilot、Student/PPO或新机制。

[修订锁](PILOT_BUDGET_AND_SPLIT_LOCK.md)建议单seed A/B/C各32,000真实50Hz transitions，共96,000；8h/8GiB总框架，准备成本也扣除。实测存盘/吞吐保守投影最大约33,500/臂，建议32k留余量。34训练父组/44回合第一遍约21,650步；dev16父组、sealed28父组整组锁定，stop/continue不跨split。新清单尚未物理执行，哈希去重不冒充统计独立。

[准备结果](INTERACTION_PILOT_PREPARATION_RESULTS.md)：A/B完整update+IO warm吞吐9.565/9.549 transitions/s；C7.869。common网络初始权重/实际action相同；C增加表示模块容量，不能唯一归因Factorization。delta-v1完整checkpoint新进程恢复两轮逐值一致；中断保留最近提交轮。独立完整评价C/BC500及共同前缀检查通过；参考入口错误/汇总错误和全部失败保留。最终runtime `6ee2dd283ca532862ca190c3df7b543895470680`，实际运行SHA逐job见账本。

成本2560.4251s（含300s保守CPU记账），新资产约0.316GiB；项目累计48.8623h/240h。原BFM/BC500/所有历史不改。首读仅dev方向性+B_I探针；§5.1正式判据尚不可满足，不读sealed、不作显著性/收敛/晋级声明。

**用户已确认32k/臂修订锁。** 原草案JSON保持不变；批准版本为 `interaction_pilot_lock_v2_approved.json`，引用Issue62 comment6816360266。独立stage `interaction-fb-pilot-20261112`，计划A→B→C串行训练，再独立开发评价及CPU probe；当前尚无Pilot行为结果。累计预算包含准备2560.4251s，另保守预记当前启动/报告CPU600s；不重置8h/8GiB。运行PID和每轮账本以stage/jobs及ledger为准。以后启动仍须实时核GPU0与CPU/IO、共享free≥100GiB；三臂单卡串行，完成轮必须完整存盘。

[Issue62](https://github.com/GaoE05/robot-research-bridge/issues/62)、[Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)、[T0报告](R1_T0_TRIAL_RESULTS.md)、[方法](INTERACTION_FB_METHOD_SPEC.md)、[历史入口](CURRENT_STATE_BEFORE_ISSUE62.md)。不得将代码推送当新行为成功，不自动开启后续阶段。

运行快照：2026-11-12，A臂pilot-A-r1已完成至少round1（250真实50Hz transitions、2完整更新组及事务checkpoint）。runtime `cf247d5f87beab89ff8be1a6eeaca764f0f93599`；queue PID2349347、A PID2349359。B/C及统一开发评价依次排队，尚无行为结论。首尝试缺少stage-local状态布局.so，采样前失败105.2319s，精确复制既有helper后重启，见[启动记录](INTERACTION_PILOT_LAUNCH_ERRATUM.md)。实时进度必须读取独立stage/job sidecars，不能把本快照当最终轮次。

最新实时核对（2026-11-12，覆盖上方运行快照）：A-r1最后完整round65=16,250 transitions/130更新；下一轮125条采样后右踝roll真实q=0.285557rad，XML上限0.2618rad，越界0.023757rad>原0.02保护，停止，未新增更新。round65和未提交失败包均保留。监督器更早在扫描.pending-18目录时与原子重命名发生竞态，异常退出，未终止/回收子进程；子进程之后自行触发物理保护停止。B/C/dev均未启动，GPU0空闲。旧ledger stopped/FileNotFound及缺失wall不能当完整成本；独立status-reconciliation保留日志活动下界与首次确认进程已死的墙钟上界，退出时刻无法精确恢复。此为需独立记录的监督缺陷，不能仅描述为日志延迟。不得自动重启、放宽硬界或宣称Pilot方法胜负。
