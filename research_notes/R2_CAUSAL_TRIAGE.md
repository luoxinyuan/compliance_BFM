# R2 / Issue63 — D0只读机制裁决

2026-11-14。依据最新[Issue63](https://github.com/GaoE05/robot-research-bridge/issues/63)、Yi本轮明确授权和源码 **f6bd9eb606975202b34e693968784308bad3745c**。本轮只读旧模型/回放/真实轨迹，在CPU做前向、自动微分和统计；**新物理0、优化步0、CUDA初始化0、sealed读取0**。只新增研究文档和可复算分析脚本，不修改任何运行Agent、reward、Envelope、共享BFM/PD或旧判定。

**结论：停止旧B19固定Q的adapter-only修补有充分依据。** 独立恢复域上第一条执行命令已经发生函数漂移，随后本体/BFM反馈放大；Return降低训练参考域的raw代理，却未约束这些状态的实际执行。Q有可直接证明的绝对值校准误差；其候选排序是否在共同续跑下错误仍待D1，不能把不同策略3秒回报当精确Q真值。没有发现足以独立解释全部退化的确定性实现bug。

推荐并行准备真实rollout的部分Actor适配强基线与Factorized Interaction FB实验方法，见[R2训练设计](R2_INTERACTION_FB_DESIGN.md)；[D1最小物理合同](R2_D1_MINIMAL_PROTOCOL.md)只是待批准草案。本轮不运行它们。

## 1. 身份与证据范围

|对象|精确身份/范围|
|---|---|
|原BFM|33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361，原PD|
|B19|core 0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984；4750条50Hz回放、38次Q/联合更新；当时尚无Envelope|
|V1|同B19权重加已批准Envelope V1；不是额外训练模型|
|Control64|c4e1c1525b37701929b0a0b33b3add420d2d5ca52155f80cf1b0ff5ba112d284|
|Return64|f2056e30087eb6748602ffd4df3815486dbfda23566d73e431d3a28c2de20fec|
|fit / physics runtime|1c06170c8932af7fbd95b8d843cc505fc814ab01 / 8afe076c9362c74cdb37fc6e936780a959cc6567|
|恢复父组P2|fea32998…，11.9s，mid已释放1.0s；同起点3模型×mean/sample的存档|
|恢复父组P3|eb777384…，9.7s，stop已释放0.3s；同起点同规格|
|参考A128|core d97f8c515fb76f34b7ff3d75b7a29208a68987afbf75de5cf727f8227180114b，runtime62d55b0d26127440253dba2b5d0baa8bd1e457fa；保留，不宣称已完整行为验收|

本轮详细再计算用6条**mean**完整轨迹；另6条sample结果引用原报告，未冒充12条重新计算。父组在B19原训练中曾访问、但从Return新监督池排除，所以是**历史已暴露、对新监督独立的恢复组**，不是全历史未见泛化。旧Envelope V1 B19资格失败、Return64结项、active/none缺格都不重判。

机器可读入口：[r2_reward_q_execution_audit.json](r2_reward_q_execution_audit.json)；每关节完整表：[joint-trajectory.csv](r2_audit/data/joint-trajectory.csv)；每个首状态29分量值/梯度：[first-state-jacobian.json](r2_audit/data/first-state-jacobian.json)。原始轨迹只在私人存档，公开的是必要数值统计，无权重/完整观测/solver。

## 2. Reward：哪些被优化，哪些只在评价中出现

实际优化入口是`dual_track_task.Task.step → native_terms`，而非`geometry_env`中保留的legacy100ms统计；`NativeActor.take_step`将每20ms真实积分放入`aux_rewards.contact_physical`；`Agent.update:82–83,109–111`直接使用它，无正向reward归一化或常数偏移。

`r_t = -Δt × Σ cost_rate`，通常Δt=.02s；`γ_t=.99^(Δt/.1)`；真正terminated屏蔽bootstrap，truncated仍bootstrap。终止还可增加负吸收尾项。释放本身不是终止。模型Q没有任务clock；固定回合时长截断后的尾项因此尤其不能当成完整已知回报。

|物理阶段|真正进入训练的目标|没有单独表达的目标|
|---|---|---|
|active移动|逐点实际力、误差、限幅前需求、相对速度、heading，以及slip/tilt/continuity|无“先手后脚”目标；无显式能耗或不必要步数项|
|stopping/仍连接holding|同一组项；伙伴当前速度/位置变化间接形成减速和保持信号|没有新的stopping loss、停止响应时延或保持窗积分专门项|
|released事件|active几何/heading项关闭；本存档force/raw_force均实际为0|没有释放事件bonus或特定恢复Goal状态机|
|recovery/无接触|瞬时root **xyz线速度平方**idle、slip、tilt、q_PD连续性|无显式累计root位移、yaw/角速度、脚步次数、恢复到公共Goal、支撑重排或手臂最终姿态成本|

关键数值语义：
- slip率=`.05*min((v/.15)^2,16)`，v≥.6m/s后封顶**.8/s**；每20ms上限.016。用的是原`contact_tangent_speed_max_mps`，不能与另存的`stage_loaded_slip_mps`或新法向力筛选口径混称。
- inactive heading=0；active heading按距离和20°尺度加权。idle率=`.2*min(||vroot||²/.04,16)`，v≥.8m/s封顶3.2/s；它不直接罚yaw或累积位移。
- continuity率按逐关节q_PD差平方均值，封顶.8/s；这不是实际q/dq成本。tilt只在15°以上起罚；本6条轨迹tilt成本都为0。
- 关节真实越界/失稳主要由硬物理检查/终止控制，不等同上述连续奖励已覆盖所有动作质量。

### 2.1 实际3秒折扣成本复算（越小越好；return为负和）

|父组|模型|idle|slip|q_PD连续性|真实return|
|---|---|---:|---:|---:|---:|
|P2|V1|.035023|.000981|.000883|−.036887|
|P2|Control64|.880297|.124928|.032582|−1.037806|
|P2|Return64|.873098|.102651|.030158|−1.005907|
|P3|V1|.021545|.000467|.001139|−.023151|
|P3|Control64|.891594|.106442|.035757|−1.033794|
|P3|Return64|.868032|.099647|.034645|−1.002324|

其它分项全0。Control/Return的slip封顶控制步数分别P2=4/5、P3=4/4（每条150步）；V1=0。**reward确实反对坏恢复，约85–87%的坏分支成本来自idle**，不能说“没有任何恢复奖励”或把全部问题归给slip封顶。复算与存档FP32 reward最大误差1.78e−9，是序列化精度差，不修改任何断言。[分项数据](r2_audit/data/reward.json)。

## 3. Q：已证明校准不足，尚未隔离排序病因

Q目标是`r+γQ_target(s',π_target(s'))`；target网络每联合更新EMA .01。它不是当前Return策略的三秒价值，也不是原BFM预训练价值。B19的128宽随机新Q只做过38次更新；Return阶段Q和target都固定。

**直接校准反证**：全部物理reward非正、终止罚也非正，所以任何真实无限折扣或完整终止return上界为0。最终Q在4750真实行为回放中**3058条(64.38%)为正**，恢复1850条中**1471条(79.51%)为正**。这独立于后续策略和3秒时域，证明绝对值高估；不证明相对动作梯度必然错误，更不直接定位软件bug。[Q-sign](r2_audit/data/Q-sign.json)。

|同起点|V1动作Q|Control64动作Q|Return64动作Q|
|---|---:|---:|---:|
|P2|.198134|.201406|.201415|
|P3|.173614|.176095|.176087|

真实首20ms reward也已经区分三者：P2 V1/Control/Return为−.001460/−.002674/−.002549，P3为−.001396/−.003073/−.002783。这只说明即时成本较差，仍不能排除Q预期的更远收益。

这些~.0025–.0033的差值不得直接对比~1.0的不同策略3秒return差：3秒后剩余折扣**.739700**，折扣半衰期约6.90s、连续时间常数约9.95s。老Q的后续是EMA target、旧无Envelope执行；V1/Control/Return物理续跑是不同当前策略且都用V1 Envelope。两种错位都存在。

本轮在原回放按五阶段各确定性等距取12条，CPU复算**原B19无Envelope target policy**的一步TD残差，均值绝对值：precontact .01352、active .03196、stopping .01124、holding .01804、recovery .01896。目标bootstrap/即时reward/逐条残差完整保留；这是最终模型事后样本，不是历史每次优化残差，也不是置信区间。[原target政策TD](r2_audit/data/target-policy-td.json)。另3849条完整相邻状态一致记录的“saved behavior下一动作”残差单列，不能与target-policy TD混合。[behavior-TD](r2_audit/data/behavior-td.json)。

**证实**：绝对校准不足、Q从未看到新64步策略的后继、Q分数改善与独立闭环退化并存。
**相容**：固定Q动作外推被actor利用；早期bootstrap偏置未消除；数据覆盖/阶段不足；演员更新离开可估值支持域。
**未证**：Q对同状态共同续跑的动作排名错误率、某个critic实现bug、哪个Q输入或网络大小是主因。没有用critic EV代替上述证据。

## 4. Raw→执行→真实身体：投影不能等同保旧

`Agent.action:45–60`先`base+adapter+noise`再[-1,1]；`Envelope.__call__`限制相对原BFM q_PD的方向裕量，名义上最多±.12rad、近关节边界更小。最后用原g(action)得到实际可表示q_PD，纯FP32向内舍入保留。`ExecutableValue.backward`传的是**解析投影**的梯度，不是把整个投影绕过的identity-ST；有效分量还受前面的[-1,1]限制。

因此有三种不同量：Envelope的`raw!=actual`标记；raw→执行局部Jacobian；网络参数→多个状态输出的共享梯度。某状态某关节Jacobian=0，不代表该输出不会因其它训练状态或Adam历史而变化。

|起点/模型|Envelope触发分量/29|总raw→exec零导数分量/29|raw关节等价RMS(rad)|执行修正RMS(rad)|相对V1首q_PD最大差(rad)|
|---|---:|---:|---:|---:|---:|
|P2 V1|19|19|.12548|.09894|0|
|P2 Control64|26|27|.37401|.10925|.08592|
|P2 Return64|26|27|.14529|.10899|.08592|
|P3 V1|22|23|.18691|.10272|0|
|P3 Control64|26|27|.46535|.10982|.10809|
|P3 Return64|26|27|.18622|.10940|.08839|

两新模型在各自同起点的**27/29个执行action逐值相同**；剩下两维是waist_pitch/right_wrist_pitch，故Control→Return的q_PD差RMS仅P2=.00300、P3=.00399rad。Return raw相对Control减少，不意味着相对V1保持：P2 Return raw还高于V1。

本地CPU导数（原函数，参数不更新）：两新模型只剩waist_pitch与right_wrist_pitch有非零`d a_exec/d adapter_raw`。P2 Return的`dQ/draw`分别约+5.9e−5、−2.82e−4；P3约+2.90e−4、−8.2e−5。它们是最大化Q的局部方向，**不是完整Adam实际更新方向**。所有29维及两种trigger定义逐行存档，见[逐关节可读表](r2_audit/JOINT_TABLE.md)。真实q范围/dq峰值取全部200Hz物理行；q_PD−q RMS取50Hz控制前状态，不把目标跳变当真实关节瞬转。CPU前向与旧GPU实际首action最大差5.96e−8，未称跨设备逐值；实际轨迹统计仍用原保存值，未放宽原物理/恢复断言。

相对V1，首q_PD大变化集中在左右腕pitch/yaw、waist roll/yaw，而不是只有踝：P2右腕yaw −.08592、左腕pitch −.08310、腰roll +.06584、腰yaw −.06276rad。不能据此把某关节认作唯一物理原因。

### 4.1 最早边界与状态传播

|父组|Return首输入/q/qdot|首实际物理分叉|root差>1cm|首支撑状态差|首次loaded slip>.45|
|---|---|---:|---:|---:|---:|
|P2 t=11.9|与V1逐值相同|11.905|12.175|12.365|12.485|
|P3 t=9.7|与V1逐值相同|9.705|9.975|10.065|10.600|

第一5ms非零差不单独作为“严重失败”门；后面的厘米级位置差、支撑变化与滑移给出持续传播证据。100ms时Return实际关节相对V1 RMS已约.0497/.0481rad，原BFM q_PD自身差约.0591/.0567rad：**BFM权重冻结，但其真实本体/动作历史已不同**。Guidance权重冻结也不意味着后续z固定。5子步原始q_PD及真实关节统计只做同初态后的闭环描述，不冒称同状态干预。

存档16/32/64权重的同状态CPU检查表明，Return16的首q_PD相对V1最大差P2=.04234、P3=.05485rad；因此函数漂移最迟在已保存的16步节点出现。没有16步物理分支，不能说物理崩塌发生在第16步，也不能确定1–15中首步。[checkpoint-drift](r2_audit/data/checkpoint-drift.json)、[分叉](r2_audit/data/divergence.json)、[时间摘要](r2_audit/data/temporal-summary.json)。

另外在V1和Return64各自真实访问的0/.5/1.5/2.98s状态，固定该状态实际z/base，交叉计算三个adapter，形成16个状态的同状态执行表；它隔离adapter函数差，不是新完整政策物理反事实。[cross-policy](r2_audit/data/same-state-cross-policy.json)。

## 5. 数据和实际优化覆盖

|阶段|B19回放条数|原Q实际抽样次数/唯一状态|Return共同池|64步共同实际抽样/唯一状态|
|---|---:|---:|---:|---:|
|precontact|310|89 /71|95|151 /78|
|active|1485|404 /324|425|676 /342|
|stopping|65|20 /16|0|0 /0|
|holding|1040|289 /227|285|436 /230|
|recovery|1850|414 /347|450|785 /374|

B19实际10回合、8共同前缀父组，全部standing；near3、mid3、stop/continue两对。**零需求/无接触完整回合尚未经历**。recovery1850条=37s跨8父组，不是1850独立恢复例；具体B19末父组release后50条只在原Q/联合更新中被抽1次。新监督共1255条/3父组（near2、mid1），stop/continue完全不在共同池；CPR205条也仅这3个原生standing前缀。

Return参考额外2父组：B19完整输入只有1条反复抽，near off mean/sample共300个相邻控制输入；每批每父组16，共64批，不能把300解释为独立恢复覆盖。没有新在线后继或Q重估；128个采用的adapter更新之外只有2个已记账工程步。

原Q曾见两个诊断父组，不等于见过**新策略的执行动作及后继**。以回放列标准差floor .1归一化的privileged向量最近邻距离P2=.0533、P3=.0417；最近邻也是同父组恢复行。相对该行实际动作，RMS距离V1=.0458/.0398、Control=.1255/.1148、Return=.1228/.1098。该距离无独立OOD阈值，也有记录时序/历史影响，**只作为动作支持变远的描述，不把近邻当完全同状态或严格OOD证明**。

`return_fit.losses`实际为−Q+.1CPR+.01disc+λReturn；在adapter-only且U冻结时，disc(state,z)不依赖adapter，**其对adapter的直接梯度为0**。这不是损失遗漏bug，是这次隔离配方的计算图事实。实际有效约束来自Q/CPR/Return；native CPR比较不同z总动作，不能保护本次缺失的恢复域。原Agent联合训练时disc仍可通过z训练U，不能推广为整个方法disc无作用。

## 6. 裁决、反证与缺格

|解释|证据等级|边界|
|---|---|---|
|新更新破坏原稳定恢复函数，首命令已有变化|已证实|两已暴露父组；不能推出所有状态|
|raw Return训练收益不能代表执行保持；投影/共享参数造成脱节|已证实|未证明改某一Envelope常数即可修复；不再扫clip|
|Q绝对值高估，缺新后继校准|已证实|不等同已证明同状态相对排序错误或实现bug|
|离线固定Q外推+覆盖变化驱动负迁移|相容，优先假设|缺共同后续动作反事实和真实更新梯度完整历史|
|reward存在恢复语义缺项|源码已证实|当前reward仍明显反对这批坏恢复，不能当唯一根因|
|原BFM/所有离线训练/所有Return都不能成功|未证且有反证|原V1在这两域稳定，base-only在已资格域也可恢复|
|需要B_I即可解决|未证|旧C未完成，不能用命名代替交互表征增量|

真正实现错误：本审计未找到新的确定性执行或reward接线错误。CPU跨设备微差单列；分析初稿用固定256分母算z角度会让完全相等的FP32 z出现约.02°假角差，交付版已改为逐值相等返回0、否则FP64范数归一化。这只修正**新分析**，没有改控制、训练或历史结论。旧efc_JT/IPC/no_grad等errata保留。

**下一项最小可判别干预**：两个既有完整恢复状态，仅替换首20ms实际动作，其后统一V1续跑；同时核验首真实后继上的原Q Bellman残差。D1不用于生产训练标签，不以旧Q尾bootstrap“证明自己”。R2方法设计现在交付，不等D1无限延长。

## 7. 核心代码定位（均锁f6bd9eb）

- [agent.py](https://github.com/luoxinyuan/compliance_BFM/blob/f6bd9eb/research/interaction_fb/agent.py#L39)：真实z插值、base、adapter、noise；L76–141 Q/CPR/FB/target合同。
- [return_fit.py](https://github.com/luoxinyuan/compliance_BFM/blob/f6bd9eb/research/interaction_fb/return_fit.py#L35)：固定Q实际执行action、raw Return、adapter-only、冻结digest、配对batch。
- [envelope.py](https://github.com/luoxinyuan/compliance_BFM/blob/f6bd9eb/research/interaction_fb/envelope.py#L22)：方向裕量、解析投影、向内表示、实际值和解析梯度。
- [dual_track_task.py](https://github.com/luoxinyuan/compliance_BFM/blob/f6bd9eb/research_notes/dual_track_task.py#L111)：native_terms；Task.step实际积分/终止；没有v3滑移风险新项。
- [dual_track_native_data.py](https://github.com/luoxinyuan/compliance_BFM/blob/f6bd9eb/research_notes/dual_track_native_data.py#L77)：实际行→真实后继→reward/replay。
- [schema.py](https://github.com/luoxinyuan/compliance_BFM/blob/f6bd9eb/research/interaction_fb/schema.py)：deploy与privileged及BI权限。
- [MuJoCo backend](https://github.com/luoxinyuan/compliance_BFM/blob/f6bd9eb/humanoidverse/simulator/mujoco/mujoco.py#L179)：joint顺序与robot配置一致，qpos[7:]对应29关节；未把q_PD当实际q。

## 8. 成本和复算

所有档案输入按固定manifest/SHA读取；未构造环境、未调用step、未调用optimizer.step。CPU分析秒数和代码哈希见[r2_audit/ledger.json](r2_audit/ledger.json)，另保守计入读取、工作树、文档、IO的上界，不偷用未来D1/R2额度。没有删除旧资产或修改其它任务。新分析脚本只供离线复算，环境变量`COMPLIANCE_PROJECT_ROOT`指向私有项目；输出路径不得覆盖旧stage。完整执行命令、输入manifest和缺格见总JSON。

本轮完成即停。D1、部分Actor解冻、Q新训、BI/FI训练、reward语义改动及任何新物理均待新授权。
