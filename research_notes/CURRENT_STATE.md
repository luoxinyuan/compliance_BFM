# CURRENT_STATE — 原生动态运动时间连续探索（Issue49）

核对：2026-11-07 CST。**新阶段已登记/正在实现准备，尚未启动新PPO。Issue48已完整结项，不重启其目录或修改原选模。**

[Issue49](https://github.com/GaoE05/robot-research-bridge/issues/49) · [协议](TEMPORAL_EXPLORATION_PROTOCOL.md) · [机器配置](temporal_protocol.json) · [Issue48结果](NATIVE_CONTACT_RESULTS.md) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)

## 当前目标与状态

原生walking进入/持续/低速接触/退出先做有限证据审查；新增正控制最多3回合，复用旧full-rate自由证据。比较同单步边际噪声协方差下，A独立探索与B时间连续AR探索；先NVIDIA真实输入条件概率/梯度检查，再两配对seed4901/4902各16×768。是否全部延至32由固定共同开发规则决定，不按封存调参。当前只是准备，不把用户提示或源码提交写成已开训。

固定Issue48原生初始化SHA d399174cc2ddbd3efb858953653c9134495070948341381565f5f918f58da5cd，两个方法同coordinate head/原normalizer/MLP/covariance/课程/reward/历史。新阶段fresh critic/Adam；不继承I/C0/T30 actor。C32/I/C0仅冻结比较，不再C0修补。原BFM SHA33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361；PD/10N/K80D8/5N回归/256D/10-50-200Hz不变。1088D特权Teacher，不是新Student；不输入未来/任务ID/clock/最终距离，不真机。

AR条件均值含mu(current)+rho*(actual_previous_u-mu(previous_observation))，更新两个mu均需当前theta梯度；完整保留实际历史，噪声边界只重置探索记忆。相同的是stationary噪声边际协方差，不是反馈导致的全部动作/状态分布。actor/critic网络输入仍1088，2433D replay wrapper只是概率上下文；critic不读精确AR residual是已记录的逼近限制。原3e-4/KL .02/Adam守卫不放宽。

## 资源与身份

新阶段硬上限：18000累计experiment-job秒、12GiB、98304新RL决策，Student/C0更新0。只使用实时检查空闲的Pro5000 GPU0，三个单线程worker；GPU1–6被其他任务占用，不动。无PPU、大下载、共享环境修改。首次49152决策，资源上限不是必须耗尽。

RUN_ROOT=/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-temporal-20261107；ledger.json记录实际作业/PID/start_ticks/预算，尚无PPO job。隔离树bridge-v2-owned/trees/native-temporal-learning/repo，分支gaoe05/native-temporal-learning；源码commit以launch.json为准，当前基础3049ef513aa49f87dde5ffca4d98e693b26d37cd，准备中的代码不是已验证结果。不要重复启动同名目录。

## 已有结论与不可丢失基线

Issue48三组73728决策/97冻结物理执行全部完成。C32零需求0.985cm/.812N、6.8cm局部2.823cm/2.300N，69cm仍34.920cm/8.756N，moving新源67.236cm/9.843N。保留best=0及其近零基线选择偏置，不否认固定末轮实际部分收益，也不宣称已解决及时walking。C32 SHA39cd0db55fddb174de3f77104acbd8d7574fb3a5051f743e4ca39bb6b7d24ad0，旧I/C0/T30/S0/D2及全部失败与模型SHA见native_artifacts/models.json和历史报告。旧原生视频在Mac videos/native-contact-learning/，4MP4+INDEX已验，不是本轮新成果。

[Issue48精确结项快照](https://github.com/luoxinyuan/compliance_BFM/blob/3049ef513aa49f87dde5ffca4d98e693b26d37cd/research_notes/CURRENT_STATE.md)

接力先读CURRENT_STATE/OVERVIEW/Issue49/protocol/ledger，核实实际runtime、资源、result和模型SHA。没有空卡不换CPU，不根据旧PID/历史快照重跑。不改main、共享BFM/PD/权重、SoftSONIC或他人任务。完成本有界阶段后统一报告、更新Master/research并停止，无自动Student或追加PPO。
