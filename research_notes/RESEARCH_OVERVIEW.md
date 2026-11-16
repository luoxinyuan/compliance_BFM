# 当前更新：E/I v2正式运行准备

2026-11-16：Yi已授权按OR1实测完成正式全程预算修订并开始长训；[预算修订](CONTACT_GOAL_EI_V2_FORMAL_BUDGET_AMENDMENT.md)。OR1科学与恢复资格复用，正式数据/初始化与工程scratch分离。当前尚未启动，具体以CURRENT_STATE及新阶段账本为准。旧已结项判读保留。

# Humanoid Compliance — 长期主线与当前执行入口

2026-11-16：OR1局部优化规则已完成有界工程资格及两臂完整恢复；[最终报告](CONTACT_GOAL_EI_V2_OR1_FINAL_RESULTS.md)。真实接触行为与BI增量未验收，OR6正式全流程成本尚未放行，当前停止。

2026-11-15最新：Yi已签[OR1分段秒帽补充](CONTACT_GOAL_EI_V2_OR1_SEGMENT_SECONDS_SIGNED_AMENDMENT.md)。仅分段秒数超支改报告制，stage47000/240h/输出/8192新采样/五α与科学门不变，OR0按决定接受现状继续。已获得E/I各首P2块2次真实Actor接受和各3项保持PASS；正在复用已有完整状态补齐原D1/D2，不再因原分段秒帽等待。原成本停点/IO/SIGKILL不重判，正式32k和Dev/Sealed仍未放行。[CURRENT_STATE](CURRENT_STATE.md)为当前执行状态。

长期目标：Contact-conditioned whole-body behavior selection, modulation and transition；由真实身体、接触历史与公开目标共同决定可持续响应。原生运动先验提供可追溯能力，真实闭环学习分配和转换，之后才验证受限观测Student及自身RL。不能将固定force→skill规则、Goal查表或物理搜索结果冒称已学会通用交互。

## 当前：已获批 Contact-Goal E/I v2

[CURRENT_STATE](CURRENT_STATE.md)为唯一当前执行状态；[v2规范正文](CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_APPROVED.md)、[签署](CONTACT_GOAL_EI_V2_SIGNOFF_20261115.md)、[不可变锁](ei_v2_locks/protocol.json)。Yi整体批准并追加授权自行修正实现问题继续；独立新6,000job秒已有批准，旧R2帽/项目240h不改。

两臂都以原BFM/PD为先验、同等Actor-tail与零初始化contact-FiLM执行适配；E普通有限结果预测，I有限交互后继编码。公开FOLLOW/HOLD目标有明确且不同的实际reward。先Goal-only原生命令和50Hz Gaussian低层PPO，后最低0.5s dwell的有限option/低层联合PPO。两臂无旧I/C0/T30高层actor继承；不是自由256D U探索、标量动作Q考试或完整基座FB重训。自身理想里程计/理想接触测力是仿真假设，不能声称已验证G1感知。

阶段顺序锁定：P0合同/身份→P1完整mean/sample/reward/工程更新恢复及成本→P2各12k低层→通过既定门→P3各20k联合→固定终点统一完整Dev→有限probe/消融→五分支结项。不用任务尚未完美作为额外门；科学/安全/权限/恢复/预算失败按原合同停，不放宽数值。

P0源重复实现已授权修复。P1完整mean/sample各12/12、有限reward及已有状态新进程恢复通过；但原8192工程采样（含失败1024）已满，仅E4/I3块。Actor0/32、0/24，执行参数逐值初始化；更新的是预测/表示/value。原joint KL门正确拒绝.16–.26的Adam提议，一提议.188249已精确复现，不能把辅助loss下降当RL学会。G07因采样/覆盖缺格暂停，正式P2/P3/Dev/Sealed均未运行；[P1报告](CONTACT_GOAL_EI_V2_P1_ENGINEERING_RESULTS.md)。下一先科学审可接受优化步，不能自行放宽KL或加样本；尚不能判断E/I或B_I行为价值。

## 已完成事实与边界

H0 P-E4/4、P-I3/4有限Goal局部物理效应保持有效；零学习Goal查表能复用E四组结果，窗口未跨真正stop，不能升级为接触驱动walking或BI独立行为收益。旧QA2/G17/Return/Envelope的失败保持，但不用其独立Q门阻塞不含该Q的新算法。BC500/BFM/PD与全部历史模型、失败、erratum保留。

[本次P1之前的完整overview存档](RESEARCH_OVERVIEW_BEFORE_EI_V2_P1.md) · [H0之前历史](RESEARCH_OVERVIEW_BEFORE_H0.md)。历史“草案未签/源阻断/需下一批准”文字只说明当时状态，不覆盖上述最新签署和实测，不再次启动旧路线。
