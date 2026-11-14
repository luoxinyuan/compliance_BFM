# R2-E/I-TRAIN-CONTRACT v1.1 — 资格门语义修订稿（REVIEW ONLY）

课题：Interaction-Steerable Humanoid Behavior Foundation Model。阶段：R2 — Online Execution Adaptation × Interaction FB。

**v1.0 于2026-11-14已获Yi批准，按当时定义的累计yaw总变差门真实暂停；v1.1 是新修订、尚未签署，不能因为原授权自动重启。** 原批准、全部失利及当时资格报告保持原判。v1.1签署后仍只在原16 job小时/12GiB许可范围内使用，并须满足本合同后附门表与失败分支授权。

事实基准：gaoe05/research `d12a0e69f2923785c6897a5527a43c8c93913173`；上游Issue63；项目240 job小时总账继续累计。

## §0 科学目标与范围

在预训练人形运动先验上，用真实交互rollout学习持续身体响应，检验新增交互后继表征相对执行适配的额外价值。

|模块|E Execution Adaptation|I Interaction FB|
|---|---|---|
|原Actor|独立副本，训练最后ResidualBlock及输出层|完全相同|
|Guidance U|全新随机隐藏层、零输出起步|完全相同|
|双Q|fresh初始化、在线更新|相同结构/规则|
|B_R、Embedding、前五Block|冻结|冻结|
|B_I/F_I|无|新增并训练|
|交互Goal|普通公开Goal Encoder|B_I编码公开交互目标|
|执行|原29D动作及PD|相同|

不使用B19 Adapter、Envelope V1、Return64或A128 U训练权重。旧A128、BC500、原生零U仅冻结参照。这是增量Interaction FB Pilot，不是从头BFM训练或通用Goal Composition结论。

## §1 严格共同初始化

完整U以同一新seed构造，隐藏层逐张量相同，输出权重/偏置全零，target精确复制。禁止加载A128隐藏层再清输出。Adam/replay/Q/Actor-tail optimizer全部fresh；tail复制原BFM，goal_delta初始输出零；B_R/normalizer/PD/合法z映射不变。E普通Goal Encoder和I B_I初始函数及32D输出相同。核验初始Q输入、Actor输出、合法z、真实q_PD一致。

复用已登记六个train-only none/near/mid/zero/stop/continue零U均值资格；正式前另核std=.01、clip=.06真实采样及E/I逐值一致。均值稳定不等于采样稳定。**异常持续旋转=真实pelvis连续unwrapped yaw相对指定窗口起点的大幅偏离，计算§3.1 yaw_excursion，绝不是总变差TV；几度小摆动须留诊断而不能用TV冒充多圈旋转。** 真实关节、支撑、执行身份和权限硬界不松。不扫初始化或挑seed。

## §2 数据与权限

沿用R2清单train34父组/44回合，dev16/20，sealed28/36。Stop/continue同父组；源/伙伴整组隔离，生成器哈希不同不等于物理独立。

腕部near/mid/stop/continue/zero/none，完整接触前、移动、保持、停止、释放、4s恢复。E/I部署输入相同：本体、动作、真实力历史、公开Goal。Actor/U/在线prompt禁止隐藏anchor、伙伴未来、active真值、任务clock和未公开阶段。Q/训练B_I可用声明当前/未来特权，不得经在线prompt泄漏。Pilot不执行sealed，不用于选模。

## §3 保持域与资格集

保护：真实原生standing/动作片段合法z连续控制；none/zero/接触前；已真实验证原生零U/原Actor可稳定恢复的状态及后继。不在未解决active域强迫复制站立。恢复标签必须同完整状态、同合法z、真实合格后继，不伪造历史。

train预固定至少none/zero/near/mid/stop父组各2个，stop/continue配对；与dev/sealed隔离。每父组先冻结零U完整资格，锁实际合格状态/来源/时段/哈希后才正式更新。失败恢复仅负例，不作Teacher。

同状态同插值z：ΔqPD=qPD_new−qPD_original。

- eps_rms=clamp(.25×parent-median(p90 native qPD step RMS),.015,.050) rad。
- eps_p95=clamp(.50×parent-median(p95 native single-joint step),.030,.100) rad。

这些是预注册工程尺度，不等于动态安全余量；完整物理核对。保持用真实输出差异，不是参数L2/静态姿态MSE。

U保持角距离：standing无接触p95≤5°，已资格恢复p95≤10°，active不加零运动约束；角度不能代替物理。

|训练保护域|实际物理门|
|---|---|
|none/zero全部online阶段|实际q相对机械界不得突破原0.02rad终止护栏、无失支撑5ms tick；原loaded slip peak≤.10m/s；**从窗口首5ms连续展开yaw的最大相对偏离≤15°**；实际root XY逐5ms路程≤.10m，net位移、TV独立报告|
|确有实际接触建立→解除的合格恢复窗|原关节/支撑硬界，loaded slip peak≤.45m/s；**从真实release时刻（或合格完整状态接管起点）连续展开yaw最大相对偏离≤15°**；root XY累计路程≤**同父组、同完整起始状态、同Goal**的冻结零U已合格参考路程+.10m。无同状态参考为UNMEASURED，不能冒称PASS|

### §3.1 实际测量、窗口与禁止替代关系

- yaw通过真实5ms pelvis旋转矩阵取连续yaw并沿实际时间顺序unwrap。`yaw_excursion=max_t |yaw_unwrapped(t)-yaw_unwrapped(t0)|`。起点是**online开始/真实release/合格接管**在该窗的第一条物理记录；不要从native前缀累计。窗口数值15°保持原数值，但变量从TV改成excursion。**大幅转出后转回**也会触线；小幅±数度高频抖动不因TV增长而被误判成持续偏航。`yaw_net=yaw(end)-yaw(start)`、`yaw_range=max−min`、`yaw_total_variation=sum |diff|`、`yaw_rate_p95`必须同时记录，TV只用于振荡/平顺性诊断，不作此旋转硬门。原终点绝对yaw≤20°的接触任务指标仍在。
- root保护统计真实5ms XY逐tick位移**路程和**，不是净位移。none/zero保护不必要走动；合格恢复相对同状态冻结参考保护多余恢复运动。伙伴要求机器人移动的active阶段**不使用standing路程门**，仍要求真实推进与接触质量。恢复参考时刻/物理历史/Goal不匹配的对照不得记通过。
- **none始终无接触，脚本尾部4s虽然可能名叫recovery，但并不是实际post-release recovery**。它按no-contact hold诊断，不允许作为release恢复Teacher或通过恢复域资格的例子。只有有实际active→release物理事件的回合才能获得post-release标签。
- 保持原真实关节硬界、失支撑、loaded slip、PD对照与所有near/mid/stop任务门。重定义仅涉及yaw目标测量，不重新判原v1.0历史资格。若真正大幅高频往返>15°，excursion仍会触门；如果只是几度摇晃但TV大，触发已预写诊断而不是直接停新门。

这是新训练保护，不替代原任务成功门。正式前逐值锁定，参考本身失败则停止报告，不移动阈值。

## §4 损失与系数

物理reward完全不变，不新增yaw/recovery，不扫权重，不改折扣/cap/真实关节界。E/I同双Q真实后继TD与min-Q、同z Actor保持、合格域U保持、实际动作Q目标。仅I有successor/防坍缩及条件性SF。

Actor/U分别非负对偶系数初值0：

`lambda_next=clip(lambda+.05×(parent-weighted normalized violation−1),0,3)`。

资格集按父组聚合的偏差/尺度，以预固定p90统计，不以密集帧压倒父组；公式、统计范围、频率、数据来源训练前锁定。不得看dev调系数/统计/重标定；系数到顶仍持续违反则停相关臂。

I successor与白化/防坍缩使用已有明确数学目标，防坍缩=.1。SF只在Q warm-up通过且首次Actor/U前，以≥6独立train父组active/holding/released真实状态标定：

`lambda_SF=clamp(.05×median_parent(||grad_Q||)/(median_parent(||grad_SF||)+1e−8),0,.01)`。

这是稳健梯度尺度，不证明SF正确。表示/梯度资格失败则SF不得驱动Actor；影子结果须注明I无完整学习资格。所有系数/数据哈希/标定曲线在首次Actor更新前提交锁定。禁止单状态设全域λ、看dev增CPR/SF、混raw/执行动作、未来输入泄漏、I额外未匹配数据/Q次数。

## §5 Q资格与节奏

同结构双Q/target/权限；新Q按非正reward上界采用共同`Q=−softplus(h)`，不替代排序校准。实际action、z、goal编码、next状态入训练；不用旧Q/Adam。

250条真实50Hz transition一块，batch64。原生prefix/reset/失败/tick独立记账。

|阶段|Q组/块|I表示组/块|Actor/U联合组/块|
|---|---:|---:|---:|
|首块|4|0|0|
|后续warm-up|4|2|0|
|Q资格后|4|2|1|

E表示更新0；E/I同Q/Actor比；U无额外组。第2块不能自动训Actor。

Warm-up退出同时满足：≥3500真实在线transition；none/near/mid/zero/stop/continue完整后继及released；train冻结配对通过非正界、动作支持、有限窗排序；无物理/权限/replay/action/恢复错误。排序≥4独立train父组，同状态首动作＋共同后续，在足够物理差异比较中方向一致≥70%；只为保守资格，不是无限Q证明。缺覆盖可Q-only到5000，仍未过则停提案。

通过后每250条新真实数据最多1组Actor/U，更新后必须当前策略采新块，禁止旧replay连续几十次Actor优化。记录行为版本、来源、Q分歧、动作支持距离、实际输出；Q分歧不是可信置信区间。

## §6 训练评价与停止

train-only预锁sentinel：首次Actor后、4k/8k/16k/24k/32k；检查采集最坏真实物理。无接触、zero和有实际release事件的合格恢复使用§3.1**同窗excursion、root path、slip、真实q与支撑**门，硬界触发立即停止当前物理并保存。失败只允按配套Gate矩阵中预批的**一次诊断/冻结对照**，诊断后仍停相关臂，不自动调整门或继续Actor更新。active可任务失败但不允许真实硬界，slip/力/位移、材料点、真实q/qPD完整报告。

Q在登记诊断连续错排明显更差后继：暂停Actor/U，最多2×250 Q-only后重验；仍失败停臂，不加容量/改reward/继续追Q。

E/I独立恢复点。共享代码/物理确定性错误两臂均停并修复重资格；单臂学习失败另一臂可继续预定规模，不能加步数改变匹配口径。

## §7 终检与判读

E/I各32000真实50Hz、单seed、方向性证据，不宣称收敛显著。固定终点后独立五方dev：E/I/共同零U/历史A128/BC500，16父组20回合，兄弟关联。

先物理：near/mid/stop成功、伙伴/腕推进、RMS误差/力、loaded slip峰/p95、**yaw_excursion/yaw_net/yaw_range/TV/末绝对yaw分列**、支撑、真实q/qPD、真实发生的release恢复与none hold分列、原生保持。

原门：near≤4cm/4N；mid≤8cm/7N；stop保持≤8cm/6.5N；slip峰≤.45m/s；绝对yaw≤20°；有效推进/真实机械界/旧相对保护保留。同父组配对，不把帧/兄弟当独立。

再表示：冻结模型、分组独立未来.5/1s标签，对比B_R/B_I/拼接/current contact-proprio；held-out AUROC、类别/父组数/bootstrap，≥.05增量参考同时报正负样本资格。固定权重I prompt disable/shuffle；无物理资格留缺格，不用probe补行为。

|结果|裁决|
|---|---|
|E成功/I无额外行为|保留E，I不晋级不追加|
|E成功/I更好且probe/消融支持|I下阶段候选，后续再验新Goal/接触|
|两者失败|停配方，按Q/覆盖/保持/执行归因，不扩模型|
|E失败/I成功|保留I，但排除容量/预算/初始不匹配后才归因|
|行为接近/probe不足|证据不足，不按loss判优劣|

固定公开Goal主要standing/hold，不宣称多目标重指定/通用组合。

## §8 预算、恢复与存储

R2累计≤57600 job秒/12GiB；截至v1.1草案，保守已用**5544.704秒**、输出约**0.814GiB**（含以前3088.840秒），不可重置。下表旧3089是初始分账快照，下一物理前从实际账本重新投影；诊断包也**包括在原16h/12GiB内**，非额外授权。D1另账但项目240小时共同累计。单GPU0 RTX PRO5000串行，共享free≥100GiB。

|分账|秒上限|
|---|---:|
|已完成工程/零U|3089|
|参数/保持/数据/Q审计锁定|4500|
|采样零U/恢复/入口资格|3000|
|E正式采样/更新/IO|9000|
|I正式采样/更新/IO|10500|
|Q/sentinel物理|5500|
|五方dev|10000|
|probe/消融/报告|3000|
|失败/冷恢复/reserve|3000|
|合计|51589|

余6011为硬上限缓冲，不要求耗尽。sidecar冷启动/恢复/IO/失败实际计账，嵌套不双计、prefix/reset不漏。

旧工程250条E18.327s/I19.401s，3组更新.084/.091s，存盘2.824/3.308s，冷单作业149.6–166.6s；21–23s/块、双臂约5637s只是乐观外推，不包括新保持/Q/完整恢复/评价。

先E/I各连续4×250 Q-only测采集/更新/表示/冷启动/完整checkpoint/replay增量/空间，p90＋保守增长投影32k/臂和评价/reserve。若16h不容纳，停报，不赌后续加速。

每250条及完整更新必须不可变可恢复逻辑checkpoint＋manifest/hash/atomic latest。正式前实测E/I四轮core/delta体积；优先无损参数/Adam增量，commands/actions/replay/physics/audit增量；新进程任意提交轮次精确恢复模型/Adam/RNG/replay/Task/solver与物理续跑。计峰值临时/失败/评价空间。不得删失败、省Adam/replay或只存weights来满足12GiB。无法保证则正式大训练前停报。

## §9 禁止修改

不改reward、不扫初始化、不继承A128/B19/Return学习权重、不用旧固定Q、无外部Teacher下载、不改PD/10N、不EnvelopeV2、不单关节改幅、不事后系数、不跳失败父组、不读sealed。普通工程可自主修复留证；Q/Actor结构、reward、采样、保持目标、权限改动须新批准。

## §10 签署与顺序

Yi此前批准的是v1.0，其首个none sampled在旧TV门下真实失败。**v1.1为新修订，未签署、不能直接从暂停位置发车。** 待明确签署并完成配套三列表审查后，才允许在E/I各32k、R2累计16h/12GiB、单GPU0的原范围内执行：用v1.1语义对现有none历史做资格回归→剩余5个事前锁定sample资格→Q/保持域/完整恢复/吞吐资格→在线E/I→五方dev→BI probe。任何失败只允预写的一次诊断，之后默认STOP_AWAIT_DECISION，不自动恢复训练。

中断条件仅：初始化资格失败；Q/执行不可信或硬界/权限/恢复错误；时间/输出/共享资源护栏；需要合同外科学改变。交付全部终检、probe、失败、runtime/checkpoint SHA、全量账与下一架构决策；不自动第二接触、额外seed或大规模FB。

正文、不可变参数锁及版本SHA须归档Issue63。批准状态不伪称运行结果；所有历史草案/失败/模型保持。


## §11 门语义与失败分支（v1.1 的规范性组成部分）

本合同所有资格、停止、信息权限、样本、预算和方法晋升门须同时参照 [R2_EI_GATE_SEMANTICS_V1_1.md](R2_EI_GATE_SEMANTICS_V1_1.md)。配套三列表（防什么/怎么算/现有PASS与FAIL例子）、证据等级、失败后一次诊断及其成本上限具有同等约束力。缺少真实例证的门必须明确标UNMEASURED，不能填成PASS或作为已经完成的科学资格。**原v1.0报告保留原失败判定；v1.1回归不追溯改判。**

## §12 v1.1签署

状态：**DRAFT / REVIEW ONLY**。只有Yi明确再次签署，并确认附表三列、数值来源、窗口语义及自动失败分支都对齐，才能开启新资格和训练。已批准v1.0的历史签署不等于本次v1.1授权。普通工程修复不能暗中改变reward、门槛、采样噪声、训练清单、原BFM/PD、Goal语义或封存访问。
