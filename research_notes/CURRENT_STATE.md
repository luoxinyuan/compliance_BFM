# CURRENT_STATE — Issue62 全维降噪冻结验证未通过，训练未重启

2026-11-12最新授权：共同std=.01/clip=.06先验证，通过后A从原共同初始化重跑、B/C同合同；Pilot累计上限15h，仍计入项目240h，输出8GiB不变。

[新结果](INTERACTION_PILOT_NOISE_RESULTS.md)：同一A65完整状态/RNG、冻结权重3s。匹配窗口误差14.42→11.17cm、力8.49→8.11N，关节越界消失；加载滑移p90 .0855→.0349、超.45 tick5→2，但峰值 .5822→.6197m/s（右脚接触点146N）。因此未通过运行前登记的峰值不恶化规则；未事后改门，未启动A重跑/B/C。该局部负结果不等于新噪声无法学习。

实际runtime ecf8856，新增训练0；GPU0已释放。累计保守成本6.15446/15h、聚合输出约1.41/8GiB。原A65和全部失败保留，旧A不充当新合同匹配对照。下一决定涉及是否接受这一Pareto修订，不能默默放宽本次验收规则；本轮未扫其他噪声、未读sealed、未改BFM/PD/reward。

## 之前快照（历史保留）

# CURRENT_STATE — Issue62 工程修复完成，采样安全修法尚未通过

2026-11-12最新纠正：用户要求解决后继续；监督器改名竞争/异常回收、Pilot清单恢复及两个存档MuJoCo数组写回已修复。五项CPU测试通过，round65新进程两次恢复的全部后续命令/action/control与500个physics ticks逐值复现原失败，未放宽断言。见[修复与反事实结果](INTERACTION_PILOT_REPAIR_RESULTS.md)。

冻结反事实：原采样两次均在7.80s右踝越界0.023757rad；mean完成3s无越界，但质量仍不足。另测仅两踝noise×0.2：无越界，但同窗口加载滑移峰值0.582→0.852m/s，未采用。原Agent默认std/reward/PD/边界均未改，不能把这个诊断冒充已修好行为。

A仍为round65=16,250真实训练transitions/130updates；旧未提交125条保留。B/C/dev未启动，sealed未读。本次六个冻结诊断进程（含两次恢复失败）新增优化更新0；GPU0已释放。Pilot累计保守job上界21,046.1843s=5.8462h/8h，输出约1.392GiB；保留原无法精确确定退出时刻的账本上界，不以活动下界替换。

下一建议只有全动作探索尺度的有限冻结对照（全29维std.01/clip.06；未执行）。正式采样修改须明确新三臂共同合同，不能仅改A、静默跳过失败父组、放宽真实物理边界或将旧/新噪声产物当严格匹配。余量不保证完成剩余三臂与评价，不自动扩8h或重新计账。原BFM/BC500/所有历史保持。

## 之前状态快照（保留历史，以上方纠正为当前状态）

# CURRENT_STATE — Issue62 Pilot 已停止：A物理硬界＋监督器故障

2026-11-12。T0三门通过结论保持，用户已接受；每臂3,000提案被否决，原草案另存。此次仅完成获准工程准备，没有启动Pilot、Student/PPO或新机制。

[修订锁](PILOT_BUDGET_AND_SPLIT_LOCK.md)建议单seed A/B/C各32,000真实50Hz transitions，共96,000；8h/8GiB总框架，准备成本也扣除。实测存盘/吞吐保守投影最大约33,500/臂，建议32k留余量。34训练父组/44回合第一遍约21,650步；dev16父组、sealed28父组整组锁定，stop/continue不跨split。新清单尚未物理执行，哈希去重不冒充统计独立。

[准备结果](INTERACTION_PILOT_PREPARATION_RESULTS.md)：A/B完整update+IO warm吞吐9.565/9.549 transitions/s；C7.869。common网络初始权重/实际action相同；C增加表示模块容量，不能唯一归因Factorization。delta-v1完整checkpoint新进程恢复两轮逐值一致；中断保留最近提交轮。独立完整评价C/BC500及共同前缀检查通过；参考入口错误/汇总错误和全部失败保留。最终runtime `6ee2dd283ca532862ca190c3df7b543895470680`，实际运行SHA逐job见账本。

成本2560.4251s（含300s保守CPU记账），新资产约0.316GiB；项目累计48.8623h/240h。原BFM/BC500/所有历史不改。首读仅dev方向性+B_I探针；§5.1正式判据尚不可满足，不读sealed、不作显著性/收敛/晋级声明。

**用户已确认32k/臂修订锁。** 原草案JSON保持不变；批准版本为 `interaction_pilot_lock_v2_approved.json`，引用Issue62 comment6816360266。独立stage `interaction-fb-pilot-20261112`，计划A→B→C串行训练，再独立开发评价及CPU probe；当前尚无Pilot行为结果。累计预算包含准备2560.4251s，另保守预记当前启动/报告CPU600s；不重置8h/8GiB。运行PID和每轮账本以stage/jobs及ledger为准。以后启动仍须实时核GPU0与CPU/IO、共享free≥100GiB；三臂单卡串行，完成轮必须完整存盘。

[Issue62](https://github.com/GaoE05/robot-research-bridge/issues/62)、[Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)、[T0报告](R1_T0_TRIAL_RESULTS.md)、[方法](INTERACTION_FB_METHOD_SPEC.md)、[历史入口](CURRENT_STATE_BEFORE_ISSUE62.md)。不得将代码推送当新行为成功，不自动开启后续阶段。

运行快照：2026-11-12，A臂pilot-A-r1已完成至少round1（250真实50Hz transitions、2完整更新组及事务checkpoint）。runtime `cf247d5f87beab89ff8be1a6eeaca764f0f93599`；queue PID2349347、A PID2349359。B/C及统一开发评价依次排队，尚无行为结论。首尝试缺少stage-local状态布局.so，采样前失败105.2319s，精确复制既有helper后重启，见[启动记录](INTERACTION_PILOT_LAUNCH_ERRATUM.md)。实时进度必须读取独立stage/job sidecars，不能把本快照当最终轮次。

最新实时核对（2026-11-12，覆盖上方运行快照）：A-r1最后完整round65=16,250 transitions/130更新；下一轮125条采样后右踝roll真实q=0.285557rad，XML上限0.2618rad，越界0.023757rad>原0.02保护，停止，未新增更新。round65和未提交失败包均保留。监督器更早在扫描.pending-18目录时与原子重命名发生竞态，异常退出，未终止/回收子进程；子进程之后自行触发物理保护停止。B/C/dev均未启动，GPU0空闲。旧ledger stopped/FileNotFound及缺失wall不能当完整成本；独立status-reconciliation保留日志活动下界与首次确认进程已死的墙钟上界，退出时刻无法精确恢复。此为需独立记录的监督缺陷，不能仅描述为日志延迟。不得自动重启、放宽硬界或宣称Pilot方法胜负。
