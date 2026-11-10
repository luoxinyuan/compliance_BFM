# Issue58：1/2/4进程真实采样加速资格（2026-11-11）

用户在“尽量加速”及后续“可以”中授权有界工程验证。正式A/B仍固定19b3117，
FB工程固定d757bee，不改其源码、actor、任务、reward、物理、更新数、概率断言或任何容差。
原单卡重任务串行。原科研stage仍执行中，本工作不训练、不扩行为族。

## 第一版实际架构与限制

- 每个独立进程各自持有完整MuJoCo、solver、本体/BFM/动作/force历史及真实last-z，连续运行原Task；
  不做qpos-only转移，也不把不同轨迹隐藏phase串接起来。
- MuJoCo物理在CPU执行。**第一版worker仍保留原CUDA观测张量、未调用的原冻结模型副本及CUDA context**，
  不是纯CPU worker、不是最低显存实现；这会有上下文/传输开销，可能抵消并行收益，实测后判断。
- 一个中央进程统一调用相同冻结BFM与同一冻结特权高层策略。命令用私有继承Pipe传输，
  不是网络Bridge/服务平台。每个worker一次仅一个实际请求，连续序号和policy SHA严格校验。
- worker冷启动按顺序，全部真实环境ready后才进入并行采样。启动/校准与warm采样时间分别记录，
  总job成本包含所有进程启动、等待、物理、记录、失败及读回。
- 实际执行保持原batch-one FP32推理；聚合不改变实际策略。每条返回action立刻clone，避免
  compiled actor复用输出buffer导致前一worker命令被下一次推理覆盖。
- 仅额外做BFM批量推理shadow（最多32组），不执行其输出。报告batch-one与batch推理动作差。
  高层Gaussian joint logprob小诊断沿原3e-4断言，随机探针仅诊断，未执行/不当PPO数据。
  当前没有优化器、策略更新或actor版本切换；不能据此宣布完整并行PPO已通过。

## 固定配对与正确性判读

原A初始化SHA112b3da596960e7064ba3a31f55f7c9da976101adcd866bedb15038501bbff1f；
冻结BFM33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
仅原已暴露dev父组0/8/16的near/mid/stop-main/stop-continue四个子回合，三个父组，
不当作四个独立父组、不触新封存，不按性能增删条件。

按1、2、4个worker依次执行，固定同一策略/源/伙伴/原控制链。每个worker只创建一个Task，
避免重建Task导致callback包裹叠加。每回合原随机种子/完整原生源流程不变。
与先前原始独立进程完整轨迹的**全部8份解压档逐字节SHA**比较，包括q/qdot、完整BFM观测、
动作/previous-z、高层raw、q_PD、force/contact、支撑与真实source prefix；只要不一致立即停止本加速项。
不放宽容差、不自动修改原A/B，不以“误差很小”冒充严格工程等价。

分别报告冷启动、warm采样、完整job墙钟、高层决策/s、50Hz控制transition/s、200Hz physics步、
reset/source物理额外成本、GPU利用率/显存、Pipe实际请求数和批量机会。
**这些是冻结策略采样指标，不是完整训练FPS。** 未含更新，不可宣传为训练吞吐。
若吞吐无提升，如实保留负例，不继续扩大worker数或换物理实现找补。

## 资源与排程

- 先完成原A/B与其原评价，再记录profile/物理一致性，再运行本独立job；GPU0不并行重作业。
- 本job最多1800秒、2GiB输出；计入原F-RL-v2≤48h及总240h，不增RL决策/Teacher搜索/训练更新。
- 启动前重新检查GPU0身份/占用、100GiB自由盘、CPU quota与允许CPU的实际空闲量、RAM、IO压力。
  1/2/4目标要求至少6个有效且抽样空闲CPU与48GiB独立可用RAM；不足则保留阻碍停止此项，不争抢。
- 本job结束即退出，无自动启动新的PPO/Student/FB长跑。

## 已完成与尚未完成

4项CPU回归已通过：4个真实CPU子进程Pipe往返/序号，跨版本拒绝，错误响应拒绝，
模拟compiled重复输出存储的独立命令保持。第二次完整CPU测试21.791秒，GPU/物理/更新=0；
第一次较小3项测试0.989秒，均保留日志。源文件语法检查通过。

**GPU并行物理与吞吐尚未执行**，未宣布加速成功，未替换当前num_envs=1可靠训练链。
旧history旋转CPU微基准：24个存档窗口约5.1×，最终FP32相同，但整批估计不足1%，未采用；
原metadata的source_groups表示8份文件，不是独立组，已有qualification-note澄清。
完整资格失败须保留负例；不把未来能力写成已具备。
