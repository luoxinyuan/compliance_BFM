# CURRENT_STATE — Issue58正式F-RL-v2运行中，FB独立工程验证排队

最后核对2026-11-10。[Issue58](https://github.com/GaoE05/robot-research-bridge/issues/58) ·
[本轮进展](ISSUE58_PROGRESS.md) · [锁定协议](FRL_V2_FB_REPAIR_PROTOCOL.md) ·
[P0-B执行合同](FB_SAFE_EXECUTION_PROTOCOL.md)。本轮是用户新授权，不是恢复旧结项。

## 科研决策与边界

冻结线以BC500＋零初始化特权支路，standing/右腕near-mid-stop/zero-none，
实际on-policy PPO自主学习；A原RL，B只加接触前函数保持。
各128×768，配对总196608。stop/continue来自mid16–45cm，禁止旧stop→long50–80cm。
旧standing62cm/.22m/s仅暴露退化哨兵。新dev32/封存56父组清单预定；初始开发实测32个不同真实源状态/40子回合，封存尚未执行。
原BFM/PD/Direct256D、std.008、jointKL.02、10N/K80D8、原时间积分reward不改。
物理搜索退出主训练数据生产，仅独立能力测量/抽验。无Student、正式FB1M或5M。

P0-A恢复数值修复、P0-B踝交接修订、F-RL-v2彼此独立；P0-B不通过不阻塞冻结RL。
单RTX PRO5000 GPU0重任务串行；P0-A≤4h/P0-B≤12h/F-RL-v2两臂≤48h，原240累计job小时不重置。
100GiB共享自由空间保护不降低。最初空间不足，用户清理后已恢复。

## 本轮实际状态（不将实现当训练成果）

- P0-A：按expert/unlabeled/原GP分量固定累加梯度。在同真实mini-batch/RNG/模型/六Adam下，
  新进程恢复后继续27步采样、一次完整更新逐值一致；进一步连续16次固定输入完整更新，
  每步模型/Adam/RNG/六optimizer有效梯度hash一致，无容差。runtime dcb3fae/3377edb。
  **compiled、同状态action及额外完整物理回归尚待完成，不发FB长跑资格。** 旧D2门不改判。
- F-RL-v2：新课程/函数保持/完整记录已实现，CPU合同回归通过；
  017c271的B臂一批工程smoke已完整完成：768真实采样、3接受actor步、18critic步；前后各6子回合评价与父状态/RNG隔离均通过。
  **正式A臂实测到第106/128批、81408决策；B臂同runtime串行排队。最新完成开发节点96，按原保护规则仍选iteration0；暂无新模型晋级结论。**
- Reward：三组已有匹配轨迹实际优化reward重建误差0；一组局部反馈的error/force/质量共同改善，
  return−12.86→−.71，排序正确。另两组有Pareto取舍，不宣称全域reward已证明正确，不改reward。
- P0-B：安全action/实际q_PD/单次采样/metadata传递CPU检查通过；三源30父组真实配对尚未跑，
  工程8192 transitions只有物理资格成立才启动。无新正式FB训练。

## 冻结身份与历史事实

BC500：a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e。
原BFM：33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
旧KL.02初始化1fa6e0703bc8e5bb9e06ec34cf72d7ecab016c672cc5899a83e5ec8626763332；
batch48 1c136654af650e60e8d5deb7fedef40818af698de2cb2df4a01390b76360f9f7。
旧两KL各98304决策均选初始化，无新模型晋级。T30/S0/M38/5403及全部历史模型与失败保留。

旧FB33792transitions/512updates后早停，512后权重/Adam/replay丢失，只有step0。
[正式旧结项](DUAL_TRACK_RESULTS.md)、[早停](DUAL_TRACK_FB_EARLY_STOP.md)、
[永久保存勘误](DUAL_TRACK_FB_CHECKPOINT_ERRATUM.md)不改，不能重训伪造。
[D1/D2](D1_D2_RESULTS.md)、[旧恢复验证](D2_FB_STOP_RECOVERY_RESULTS.md)及erratum保留。
旧198边界均真实ankle roll越界，143接触前；不是198次摔倒。

上一轮8dd48a0三包结项：[总报告](FB_NATIVE_BIAS_RESULTS.md)。
原生小腰转/抬臂/下沉与恢复有有限物理证据；前进/停止有行为但slip/yaw质量有限；
有限来源未找到可靠持续后退，不等于基座不可能。
B公共均值影响reward聚合，z未坍缩；中心化只做离线，未执行/训练，不当唯一根因。

## 运行与接力

私有stage `experiments/frl-v2-fb-repair-20261110`，各job sidecar记精确PID/runtime/成本。
正式A/B固定runtime19b3117，不改其正在运行的源码。FB独立工程与有限队列在另一worktree准备；不得把发布HEAD当A/B运行commit。原运行worktree不覆盖；代码集成仅个人gaoe05/research，不改main或共享BFM/PD/资产。
下一会话先读OVERVIEW、本页、本轮进展/协议、Issue58和Master18，实时查job/GPU/账本；
不要凭本页的暂时状态重复启动任务。工程smoke不计科学模型，发布commit不等于运行commit。
新阶段进行中，完成各包或到额度即保存停止，不自动开始后续里程碑。

## 用户追加：保护正式A/B的工程加速

[记录加速实测与预登记](FRL_EFFICIENCY_RESULTS.md)：CPU无损压缩写盘基准已完成，8.3315→1.6897秒/批两档，解压字节完全一致，文件约+10.6%；不是完整训练4.93×。4项CPU回归通过，失败原记录保留。独立worktree准备原生实际物理前后对照/cProfile与推理batch缩放基准，等待原A/B全部完成后单卡串行；正式A/B固定19b3117及FB工程d757bee不变。当前并行接触采样仍未实现，不把未来潜力写成已有吞吐；新增成本从原48h/240h扣除，不加RL决策。
