# DUAL_TRACK_RUNTIME_RESULTS — Issue55, 2026-11-09

工程里程碑完成；无正式Expert/Student PPO或大规模FB训练，无Teacher查询、排序器训练、UFO/mjlab安装。BC500仍冻结基线，5403b7不晋级。所有计时更新仅保留为不可推广的disposable副本。科学合同为待Yi批准草案，未自动开启下一阶段。

[共同合同V1](DUAL_TRACK_CONTRACT_V1.md) · [完整聚合实测/身份](dual_track_artifacts/measurements.json) · [准备失败与勘误](DUAL_TRACK_ERRATUM.md) · [Isaac最小缺口清单](ISAAC_BATCH_CONTACT_GAP.md) · [Issue55](https://github.com/GaoE05/robot-research-bridge/issues/55)

## 1. 已实现与验证

- `dual_track_runtime.PrivilegedBranch`：原BC500完整1639D输入、center/scale、actor路径保留；另接当前1088D特权输入，支路1088→128→256，只有最后一层零初始化。初始mean/normalizer逐值复现BC500；最后层与随后前层梯度均非零。raw相加一次，再用原direct256→norm16；Gaussian只采一次。它是Teacher侧工程实现，不是部署Student新基线。
- `PPOContract`＋新入口`dual_track_ppo/dual_track_critic`：统一KL/回滚配置，保留独立critic拟合。默认与旧耦合更新器、旧独立critic更新器分别在相同状态/RNG下得到逐值相同权重与日志；错old-logprob被原断言拒绝。limit=.02、stop=.015、assert=.02001、logprob<3e-4、clip=.15、7次重试/.5回退均未放宽。测量runtime使用同值dataclass默认；最终源码另加同值版本化JSON读取，未把文档提交冒称运行版本。
- `TrainingBranch`：跨reset恢复callback/spec绑定、完整MjData/solver、BFM/动作/last-z/力历史。新增Python、NumPy全局、Task Generator、Torch CPU/所有可见CUDA、显式采样Generator状态。复原后的下一步真实物理逐值相同。
- checkpoint保存真实native reset＋已执行raw前缀重建配方、模型/optimizer/显式计数和全部RNG。独立新进程实际重建验证physics/history摘要、模型与全部随机流逐值相同，counter恢复为保存值；没有qpos-only reset。此测试覆盖该真实前缀；不夸大为任意环境/未完成rollout缓存的完整中断续训保证。
- 独立子进程评价四条旧回归，父Task物理/历史及所有RNG逐值未变，四回合无boundary。它是已暴露回归，不是新封存。当前Pro5000的normal72为11.163cm/7.462N；不强称与旧H100长时闭环逐值相同。
- 原FBcprAux完整回路在独立副本strict weights-only warm-start；6个Adam初始均无状态、z_buffer空、replay空、新run env_steps/agent_updates从0开始。模型里的预训练normalizer统计属于权重，不冒充旧run计数。
- 新实际控制3825条中取得3817对连续非terminal20ms transition、3761个可抽样真实8帧块，填入fresh replay；仅用于计时，**不是资格化的正式FB expert数据集**。完整256次update执行discriminator、F/B、critic、aux-critic、actor及targets，六优化器均有状态、六在线模块参数探针均有变化、所有Tensor metrics有限。
- base与高层checkpoint存盘读回验证；base模型和六优化器完整逐值相同。FB benchmark未做伙伴/contact输入扩展，不等于接触条件FB科学训练已跑通；也未提供完整FB simulator/replay/RNG恢复保证。

## 2. 单卡与计时口径

仅GPU0 RTX PRO5000 72GB Blackwell（73415MiB），driver580.126.09；project venv PyTorch2.7.1+cu128、MuJoCo3.3.3。启动时卡0空闲，其他卡有任务，未操作。容器CPU配额210核是共享配额，不是独占；采样/数值测试Torch单线程、编译workers上限4。无共享环境安装/变更。各GPU测试顺序执行，无本任务并发干扰计时。

高层物理：原10/50/200Hz、K80/D8/10N、原BFM/PD，可靠链num_envs=1。高层FP32关闭TF32并沿用canonical-row概率合同；FB使用官方train原`float32_matmul_precision=high`（FP32存储/允许TF32matmul）、AMP=false、原2048架构/1024batch。两者是各自现有配置的运行能力测量，不是公平算法优劣比较。高层gamma=.99/100ms与base .99/20ms并非同一时间折扣；FB空auxiliary-reward列表也不是当前contact reward。共同科学目标和折扣匹配已列入合同待Yi批准项，测速未静默修改它们。

|高层完整batch项目|实测|
|---|---:|
|真实高层决策 / 实际50Hz控制|768 / 3825|
|采样（含途中reset）|79.350s|
|其中reset|18.714s|
|完整PPO更新（原KL guard＋独立critic）|0.852s|
|接受actor步 / 独立critic步|1 / 18|
|old-logprob最大重算误差|0.00012207 < 0.0003|
|采样数据保存|1.537s|
|checkpoint保存 / 读回|0.076 / 0.065s|
|采样速率|9.679 decision/s；48.204 native transition/s|
|含更新、上述保存/读回|9.380 decision/s；46.714 native transition/s|
|四条件独立评价（含子进程初始化）|187.864s|
|若每batch做上述评价的合成速率|2.847 decision/s（情景估算，非已选正式频率）|
|PyTorch allocated / reserved峰值|0.217 / 0.242GiB|

高层未启用JIT，编译时间为不适用。actor接受步少是原合同真实结果，未为吞吐数字绕过guard。实际控制数为3825而非理想的768×5，按实计数，不虚增为3840；不从该计数单独推断具体失败原因。reset/source额外physics成本包含于wall，但未从旧wrapper补造完整独立step账本。

|完整FB agent.update|eager段|compile段|
|---|---:|---:|
|完整更新数|128|128|
|段wall（含进度IO、冷启动）|26.851s|343.320s|
|首调用|6.171s|283.109s|
|第二次调用|0.148s|42.261s|
|排除前8次的中位数|0.145856s|0.122453s|
|PyTorch allocated峰值|11.696GiB|10.849GiB|
|PyTorch reserved峰值|11.871GiB|13.455GiB|

compile后更新计算约快16.0%，但首次两调用含trace/compile/autotune/capture/更新/缓存成本，**没有单独剥离出纯编译CPU秒数**。每次更新实际抽train1024与expert1024，不把262144次train replay抽样冒充新增环境transition。计时包含抽样、CPU→GPU、所有更新和同步；per-update timer不含JSON进度写入，整段wall含它（分别2.129/2.498s）。不能把1024/update_time当环境训练FPS。

base模型构建4.069s，warm-cache权重载入0.362s；模型＋optimizers保存11.502s；optimizer读8.805s、模型读7.550s，完整身份比较另有CPU开销。是应用可见读回，不是冷存储/断电持久性测试。独立资产初始化与整job耗时见账本。

## 3. 单卡预算：实测项与推导项分开

现有原FB低层训练默认比率16updates/1024新增50Hz transitions，即每1M需15625完整agent updates、每5M需78125。按本轮含reset真实采样48.204transition/s，串行算：`N/r_sample + (N/64)*t_update`。

|新增50Hz transition|采样|eager采样＋更新|compile稳态采样＋更新|
|---:|---:|---:|---:|
|1M|5.762h|6.396h|6.294h|
|5M|28.812h|31.978h|31.470h|

这是**两项实测拼合的预算推导，不是完整在线FB训练FPS实测**。若沿用本benchmark逐update进度IO，1M约6.47h(eager)/6.38h(compile)，5M约32.34h/31.89h；compile另加本次约5.4min冷调用超额，另计模型初始化、checkpoint、replay保存及独立评价。实际base actor更新后的reset率/物理状态可能变化，不能从这一次样本承诺学习曲线。

正式run可承诺范围：已证明原拓扑可完整执行两段128更新和当前链768真实决策。**正式run尚未批准，不能承诺1M或5M已获资源/科学授权**。1M可作为Yi审核的单卡小时级预算候选；5M约32h以上且replay/存储/评价另计。带伙伴扩展的FB拓扑、批量接触采样、全replay存盘、正式run评价频率均未知，不能用当前数字替代。无已实现的Isaac批量Tether吞吐或num_envs承诺。

## 4. 版本与成本

- 高层实测runtime：`a2efe6cee4c9a4cca974f862affa3b2fe16fdc66`。
- FB / 新进程恢复runtime：`3fa9b8e66d8d72311438196ef7440970c355f7af`。
- BC500：`a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`。
- 冻结BFM：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- disposable高层：`186a8c10ea52b2c7f68836df8a7bca900b34c0a0b5a7ee5b2ba4ae4101413a10`；disposable base：`6b99d00d5be140bbf79ec15dc7293b704b9b09a3e7c7ab942be65f3494e34e97`，均非新基线。
- 实际控制数据SHA：`e49e7d9299dbe7006f6345e331eb3eb64b73aad9a297b23c731b1276f281dc73`。
- 私有stage：`experiments/dual-track-contract-v1-20261109/`，`runs/high-06/`、`fb-02/`、`resume-01/`；完整job sidecars、失败及checkpoint留在该目录，不入Git。最终保守计费4076.20s（含父子双计及2300s独立测试/分析/预检预留）；输出6.801GiB，低于工程自限14400s/14GiB。精确分账见私有ledger及聚合证据resource段。

只有工程测试，未产生可推广行为增量，所以未新增演示视频。原生/旧策略结果与失败均保留。阶段结束停止；待Yi批准共同科学合同后才能另开正式研究运行。
