# F1 L0 预检 — BLOCKED，未进入L1

2026-11-11；[源合同Issue60](https://github.com/GaoE05/robot-research-bridge/issues/60)。用户本次仅要求启动L0。已实际执行并按严格正确性门停止；不把阻塞写成能力失败，不自动启动L1/L2/L3。

## 一句话结论

资源可用，但首状态的完整portable状态重复一致性未通过：**两条1秒真实physics、BFM controls、输入、raw、评分逐值一致；末态25个MjData数组不同**。差异全部位于`state_after/mjdata_arrays`，没有发现本对照的q/qdot、外力、控制、输入、RNG/历史或积分状态分叉。严格门仍失败，不删字段、不放宽容差、不自动重跑。

## 实时资源与实际工作

- 端口1017，单RTX Pro5000 GPU0；启动时0MiB/0%，共享盘641.79GiB可用（100GiB保护＋2GiB预留通过）；报告时641.32GiB。先前“77GiB”历史快照不作为当前事实。其他用户GPU3–6有任务，未触碰。
- 主机可用RAM约763GiB、CPU quota210核、load约5.25、IO PSI avg10=0；这些是启动快照，不保证后续资源。
- 原v3 coordinator/training PID均已结束，GPU0最终释放；没有训练、Teacher标签生产或重跑280分支。
- 独立runtime `a39be585f54b71a213ac65b1171aa6a58f6d7354`。完整模型hash见[f1_state_manifest.json](f1_state_manifest.json)。BC500原模型与零特权支路包装checkpoint分开标识；BFM固定33f410c1…。
- 锁定12个暴露主父组24状态，另索引3个原生无接触来源（standing、arm_ready_goal、forward_motion），合计计划15父组27状态；**实际只重建1父组1状态，其余未执行，不计为资格通过**。别名组额外1父组2干预分支，仅预登记，不计能力统计；旧sealed未读取。

## 首失败边界与有限通过项

首状态P1-1，父组`00be6bacd16f55262bc9c376fa4a31cba715b3a0d5364a3bf354d948f4308ad3`，BC500包装策略，实际t=5.400s。由原source与保存raw前缀真实重建，当前2727D输入、上一z和mjSTATE_INTEGRATION与P1原档逐值一致。原CrossReset/TrainingBranch指纹恢复3次通过；这些是其现有覆盖范围内的通过，不等于所有MjData字段已通过。

两个1秒冻结策略续跑各10个高层命令/50控制/200tick，全部physics/control/input/raw/评分逐值相同。比较末态完整portable对象时25个数组不同：含`dof_island`、`efc_island`、`map_*`、`iefc_*`、`iM_*`、`iLD`以及`efc_AR_rownnz/rowadr`。具体路径、形状、索引与差值见[f1_l0_first_difference.json](f1_l0_first_difference.json)。本轮不是D2的discriminator更新问题，根本没有optimizer更新。

本地MuJoCo头文件`mujoco/include/mujoco/mjdata.h:410–471`将多项标为mj_island生成的岛/约束结构，AR项属于约束矩阵缓存；出现非物理量级索引/极小浮点值，与未激活或未初始化工作区解释相容。**尚未核实当前配置下每个数组的有效长度/读写依赖，不能据此认定全部无害或删除这些字段。** 已证实的只是这对1秒实测没有物理分叉。需下一项独立修复/合同证据，不自动修改本轮runtime。

## 已测成本与不能估算的部分

完整内存恢复3次为[0.013469251978676766, 0.01073220098624006, 0.01028630998916924]s，中位数0.010732s；1个完整快照1296598bytes，保存0.110585s，读回0.055193s。

1秒执行wall原设计暂存内存，严格断言失败发生于写timings.json之前，因此没有可信的独立候选wall记录；**不从mtime、整个113秒job或历史23秒/query推造吞吐**。3秒候选、100ms重规划、独立冷启动分解均未完成。L1/L2/L3成本投影因此标未知，不能发车。

实际受监督job 113.187s（含失败/初始化/保存/退出）；准备reserve900s、收尾reserve100s，保守合计1113.187s=0.3092job小时<0.5h。只读差异定位0.488s在收尾reserve内，不重复计账。新输出约7.12MiB；模型和原始快照不公开上传。额度记入原240h，不重置；剩余L0余额不是自动修复重跑授权。

## 信息权限审查与后续门

1639D受限路径为465D机器人state/last_action/history_actor＋256D真实last-z＋918D真实力历史/固定安装；`geometry_env.py:26–48`与`m9_learning.py:37–42`明确，不能输入anchor、partner真值速度、active或未来。理想力只是simulation proxy。

1088D特权附加路径含当前anchor误差/相对速度/active/目标朝向等，`clean_env.py:85–91`和`geometry_env.py:43–46`；B64/B128/v3不能冒称受限Student。完整真实未来Task.spec用于仿真排名时应标O-F，即便候选网络不读未来。此轮两个冻结测速分支也保守标O-F环境回放，不是C-S搜索成果。

旧`native_rolling.py:37–41,156–186`有当前伙伴位置/速度预测与未来字段擦除，可作后续C-P实现来源；本轮未移植/验收F1因果权限。合格C-S planner目前缺格，不能借真实隐藏状态/未来模拟补上。饱和别名只预登记构造及交叉响应条件，未构造，更没有控制相关信息不足结论。

**状态：L0 BLOCKED；资源门通过，完整状态严格门未通过，测速/计划状态恢复不全。** 下一项最小建议是只读核对上述25数组的有效域和动力学依赖，建立可审查的状态保存/比较合同及回归，再决定修复；本轮不自行修改或继续L1。Issue59结项与BC500基线不改。
