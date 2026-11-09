# CURRENT_STATE — Issue56双轨正式stage：合同与迁移准备已验证，正式训练未启动

最后核对：2026-11-09。[Issue56](https://github.com/GaoE05/robot-research-bridge/issues/56)
· [获批范围合同](DUAL_TRACK_CONTRACT_V1.md)
· [已获批P1–P9](DUAL_TRACK_SCIENTIFIC_DECISIONS.md)
· [实际准备进度/证据](DUAL_TRACK_FORMAL_PROGRESS.md)
· [Project Master](https://github.com/GaoE05/robot-research-bridge/issues/18#issuecomment-6739473031)。

**本阶段尚未完成；正式Expert/Student PPO=0，正式FB transition/update=0，无队列。**
没有把计划、源码push、step0测试或#55工程测速写成学习成功。

## 当前批准与科学边界

- 原腕部接触/三类真实原生源，30开发/60新封存parent目标；共同前缀整组划分。
- 高层同初始化KL.02/.08、std.008；Expert总196608决策、Student后续98304。
- FB独立原权重warm-start，fresh optimizer/replay/计数，首次仅1M真实50Hz
  transition，保存0/250k/1M；5M必须另批。250k非任务质量自动停止门。
- 总240累计job小时：冻结112/挑战80/共同32/缓冲16；单Pro5000重任务串行。
- M1质量线保留作参考，取消小样本全门槛对研究方向的一票否决。
- 本会话新增明确批准：统一gamma高层.99/100ms、FB.99**.2/20ms及原逐点
  物理成本积分；FB97D当前接触/伙伴输入、固定尺度/原统计冻结/零新权重、
  robot-only discriminator；三源等比例、近中长停零无接触20/20/25/20/10/5%，
  高层不额外BC保持，FB新真实replay与原生无接触CPR原则。
- **用户已整包批准7354e0a的P1–P9**：两KL臂各98304决策；具体范围、父组、选模容差、优化节奏、Identity辅助reward归一化、fresh辅助critic、native-z在线先验、新replay与恢复配方均已明确批准。当前在做必要入口验证，无待确认科学参数。

## 本轮实际完成

原#55草案单独归档；版本化获批范围与JSON决策表分开存储批准值和建议。
实现97D输入迁移，保留原input LayerNorm域，避免零补齐仍改变旧函数。
真实BFM checkpoint的actor/F/B/两critic/discriminator step0逐值一致；
新权重梯度和fresh optimizer参数归属通过。三类真实源各作原/新.5s物理
分支，25控制步/分支，实际动作/qpos/完整物理历史逐值一致。
新阶段终止/截断回报测试通过；父组泄漏/错误后继/预测样本入库拒绝测试通过。

这些是工程正确性结果：**0科学dev/sealed新组、0封存评价、0新行为收益**。
新的真实采集、终止后继、跨reset完整状态恢复及物理/训练RNG隔离测试已通过；高层正在运行可丢弃的768决策端到端测试。FB真实online collector、修改后模型/6Adam/新replay/实际动作物理前缀读回已通过。入口验证已完成，尚无正式学习结果；下一步从全新初始化串行启动正式两KL臂与1M FB。
原有#55原拓扑1M估时不是本轮真实完成时间。暂不重复生成视频。

## 精确身份与运行位置

- BC500 `a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`，
  仍是冻结可靠基线；5403b7未晋级，原M38及其他历史保留。
- BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；
  原共享权重/PD/main/SoftSONIC未改。尚无新正式model SHA。
- 97D模型迁移runtime `82e84807779c2bf295b3b3f63e44c81960cdbe3d`；
  真实物理回归 `d3b137f5d23e08845c8db439b74e91013e75c607`；
  returns回归 `7d1ae5f38cb7a278874021e266fc587d1e20d4af`。
- 私有stage `experiments/dual-track-formal-20261109/`，独立detached worktree，
  未新增研究分支；`jobs/`、`runs/`、ledger保留实际身份/原日志。
- 个人审查入口`luoxinyuan/compliance_BFM:gaoe05/research`，发布SHA在Issue56中
  单独记录，与上述实测runtime区分。原#55起点a097774，不变更旧运行环境。
- 本地少量验证JSON：`~/Projects/compliance-bfm-diagnostics/dual-track-formal-20261109/evidence/`；
  没有新checkpoint/大数据同步Mac。既有视频保持原路径。

## 成本、安全与接力

209.95802s实测job；另3600s明确准备/发布保守预留。为避免漏账，#55保守
4076.20309s先并入共同额度，总共同保守计费7886.16111s，仍在32h内。
只有明确隔离的可丢弃工程更新：高层768决策；FB每个工程尝试4个full agent update。没有正式学习收益或Student训练。GPU0当前由本阶段工程检查独占；可用性必须实时核对，
下一启动仍需复核GPU/CPU/RAM/I/O。CPFS约257GiB自由/99%使用，不假定存储无限。

新会话先读本文件、OVERVIEW、获批范围合同、P1–P9和Issue56；恢复前核对
实际ledger/进程，不能把工程验证写成正式训练，不能把缺失四格填成失败/成功。
既有理想力Student不是已验证真机感知。无真机、无非公开公司数据。

历史入口：#55工程结项、SOURCE_COVERAGE_M1_RESULTS、RETENTION_VALIDATION_RESULTS、
RETENTION_RESULTS、INTERACTIVE_INCREMENT_RESULTS及对应Issues。原CURRENT_STATE全文
仍可从a097774提交读取，未删除历史实验或改写任何结论。旧Library2026-10-28
仅为历史快照。

## 最新入口验证补充

P1–P9已获批，**不再等待任何科学参数确认**。
高层768决策完整采集/更新及独立评价通过（6ab8b6e…）；FB完整读回及
实际29D动作前缀重建通过（b9862178a6355043a19a809b08c27c5aa6a2fe94）。
见FORMAL_PROGRESS与FORMAL_ERRATUM；失败保留，未放宽原断言。
当前累计实测job2140.3241s，保守共同计费9816.5272s。短样本FB采样
58.6 native transition/s不含全周期更新/IO；新compiled稳态吞吐仍未知。
3个开发父组已用于工程闭环检查；封存未查看，90父组仍是目标而非已获资格。
正式训练启动后的运行SHA与状态将在Issue56和本文件另行记录。
