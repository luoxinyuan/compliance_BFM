# Contact-Goal E/I v2 — OR1 局部优化修订 v0.1（可签稿）

**状态：REVIEW ONLY / 未签，未执行。** 批准人Yi；Issue66；2026-11-15。原v2已签正文及全部失败原样保留，本附件只在签后覆盖下述局部条目。当前指令只授权只读审查与交付本稿，不授权本稿内Adam提议、采样、物理或新优化规则。

依据：[独立审查6893703131](https://github.com/GaoE05/robot-research-bridge/issues/66#issuecomment-6893703131)、[P1结果@02da99f](https://github.com/luoxinyuan/compliance_BFM/blob/02da99fcf26e638fd42d6516fd5aff2bdda9d821/research_notes/CONTACT_GOAL_EI_V2_P1_ENGINEERING_RESULTS.md)、[本次只读事实与代码依据](CONTACT_GOAL_EI_V2_OPTIMIZER_AUDIT.md)、[实际存档统计](contact_goal_ei_v2_optimizer_archive_audit.json)。原正文基于`37ab84ea4004cfa1f7be4efc6ce783cf4e164c51`、签署`15fc96c573deba4f01972ccdeb85e8655428622e`；body SHA `69f10a9143932351c740d5208d561f81f9aeeafe69da49c3efc9da773ac5c3b9`。

## §1 唯一问题、出口与不变项

本次只验证：**现有100ms joint PPO在不放宽KL的条件下，能否产生可测的实际执行Actor更新，并完整恢复和安全续跑。** 这是P1优化规则资格，不是E/I行为结果或B_I胜负。

共同原KL=.02、sigma=.01、Adam基准lr=1e-5/betas/eps/weight_decay=0、max_grad_norm=1、原reward、gamma/GAE、物理/PD/10N/K、Actor容量/输入许可、Goal、四命令及先验、最短option时长、split和全部物理门保持。**不改逐维平均KL、不提std、不加entropy、不改先验、不引入Teacher/Q、不碰Dev任务或Sealed。** 原PPOclip=.2、2epoch、minibatch64、4辅助更新及value更新规则不改。

仅新增：对**一次真实Adam提议的参数位移**进行固定五点回溯及预声明目标检查；其余均为记录、事务／恢复及有限新工程采样。该规则是科学优化修订，不能叫数学等价的小修复。

本稿签署后仅自动连续执行OR1-D0→有条件D1→D2并交资格报告，**不得直接启动原32k/臂正式训练**。通过后提交可运行性与真实成本；若仍失败，提交分层更新替代稿并停，不再自动追加缩放/采样网格。

## §2 实验身份与不可变起点

新stage逻辑ID `Issue66-EI-v2-OR1-20261115`，独立输出，不覆盖原P1目录或latest。源码仍从`02da99f`普通fast-forward形成独立实现SHA；开跑前归档该SHA、本附件字节SHA和参数锁、GPU/环境、checkpoint与原始账本摘要。未实现的源码SHA当前记PENDING，不伪造运行身份。

D0两个归档锚点固定：

|锚点|真实rollout|完整更新前model/Adam来源|原始重现要求|
|---|---|---|---|
|E-P3|E工程block4的1024行|`P1-scratch-E-02/checkpoints/block-mid-last`的agent；余12行从现存完整block/expected包拼接|原第一提议4个KL字段逐值复现，joint=.1882494853645253；原64个group索引已存|
|I-P3|I工程block3的1024行|`P1-scratch-I-02/checkpoints/block-mid-last`的agent；余12行从现存完整block/expected包拼接|原第一提议KL与原I/block3首step逐值相同；使用实际CPU RNG/生成的group顺序，失败不换提议|

pre-update模型/Adam、块内raw/old_mu/oldlogp、真实z/option及selector pre-draw输入逐哈希冻结；不得用post-update value替代pre-update value重新造advantage。CPU minibatch RNG、GPU采样RNG分别标明来源。若需补齐最后12条采样所消耗的RNG，只按已保存draw顺序离线重放并逐值核对原draw，不执行物理、不称新增on-policy；无法重现则记REPRODUCTION_BLOCKED，不猜RNG、不换锚点。D0中不创建Task或重跑12条物理。

原E/block4 manifest SHA `cdd398b933a50852e9597f06517669250f64e313f7fa20dd40229e8f50c7a5c8`；I/block3 `326e875f969de2875fabc564b9197a5c3eefed81d0611cee75ab110a2eacd46d`。它们指完整逻辑包，非单.pt权重SHA。旧tail与normalizer身份见原P1机器结果。

## §3 D0：一次有限真实Adam提议回溯（签后，无新物理）

每锚点仅形成**原首个minibatch的一次Adam提议**。θ0/S0为全部共同modules与Adam输入状态，梯度裁剪仍原值1；计算θA/SA后冻结，不为不同α重新反向、重抽batch、另跑Adam或更新V。候选只取：

`alpha = [1, .5, .25, .125, .0625]`

- α=1逐字节copy θA（已CPU修复的端点）；α<1按FP32 `θ0+α(θA−θ0)`构造，非浮点buffer不插值。每候选从同一θ0/S0，不串行累积候选。
- 正常有限度为2锚点×5候选=10个模型评估；最多再重读复验同一候选，不增加α/seed或状态。上一轮完整GPU网格**未通过且未运行完**，本稿不承接虚构的α=.25正结果。
- 原数学weight_decay=0下，缩小参数位移、保留这次Adam完整新moments/step，等价于当步effective LR缩放的实数公式；不等价于缩梯度、缩moments或“只训练参数不动Adam”。FP32不同表达式未必逐位同；参考实现以本参数插值表达式为唯一真值，另报告直接缩lr表达式的逐张量差异，不事后放宽容差。

### 3.1 必须测的量

每候选、整真实1024块、所有父组/Goal/阶段分别报告：

1. 原实现gaussian/categorical/joint解析KL（原分母/原.02）；逐native及逐真实draw条件KL另报；sample old−new估计只诊断、不抵消解析项；补FP64诊断，不改原gate。
2. joint log-ratio/ratio分位数、PPO surrogate、clip_fraction，明确**before / full Adam proposal / scaled candidate / final committed**四种身份。固定原minibatch及整块都测；历史的pre-step clip_fraction=0不代替新数据。
3. PPO任务loss、原λ×保持loss与合成目标；父组等权保持violation、p95以及实际q_PD容忍尺度不变。原对偶λ/更新频率不变。
4. tail/FiLM/selector/Goal/predictor分模块参数、原始梯度/裁剪后梯度、Adam位移；selector每Goal/阶段真实draw数、logit/概率变化；辅助和PPO计数分开。
5. 同状态同实际z的μ、raw范数、clamp触发、实际执行action差；按原29D映射算q_PD各关节RMS/p95/max，保存五子步表。不是新physics，不称真实q改变。
6. 实际z保持为该记录的真实插值；另算selector候选分布与mean选择的目标z变化。不得用新option重跑状态来计算old likelihood。

### 3.2 唯一选择／提交规则（适用于签后新工程PPO）

按α从大到小，选择第一个同时满足以下条件的候选：

- 相关参数/梯度/Adam及统计全有限，原raw联合logprob断言不变；
- **整块原joint KL≤.02**；相对本块行为策略，而不是只相对上一个minibatch；
- 固定该minibatch、该GAE、该λ上的**原合成目标** `L=PPO_task+lambda*retention` 不大于提议前L。任务surrogate与全块surrogate均完整另报，不宣称有限minibatch下降保证物理改善。

这条目标不增条件也是本次待签的明确优化选择，不能悄悄新增到旧v2。无通过候选→恢复θ0/S0及其step计数；有通过候选→提交θα与**一次**SA，不再重跑Adam，不把moment乘α，不累增step五次。auxiliary仍原一次完整step＋原joint-KL事务，不用缩放制造额外未来监督。value仍独立更新，不纳入Actor获接受计数。

事务测试覆盖整套共同Adam（含先前已有Goal/预测moment）与未参与该次梯度的参数、step、RNG、λ/计数。候选评估不得改变RNG。故障原件单独保存，atomic latest只指完成事务。D0原始拷贝最终恢复且哈希一致，不发布为训练模型。

### 3.3 D0出口

两锚点均有候选通过以上规则，且该候选相对θ0：

`S_mu = RMS(all valid 29D mean differences)/.01 >= .005`

并且 `p95(abs(actual Δq_PD)) >= 1e-5 rad`（分别附每父组与保留域）。这两个数是**待签的执行可测性工程线**，不是任务成功／物理安全线。要求它们超过相同权重重复前向的数值底噪；repeat本身应按原batch-one合同逐值一致。固定θ0重复前向不一致→正确性失败，不能提高容差。clamp把差异全部吃掉也不算有效动作。

- 两锚点通过→D1；
- KL有可接受步但均低于可测线→`STEP_TOO_SMALL`，停止，进入§7替代设计的审阅；
- 无α通过→`NO_ACCEPTABLE_PROPOSAL`，停止，不往1/32再试；
- 只有一臂通过→不挑好的一臂冒充匹配资格，提交两臂差异；
- 参数/Adam/复现失败→工程正确性停止，原始资产保留，原§9内唯一等价修复机会适用。

## §4 D1/D2：只预登记最少的新on-policy资格

**新工程采样硬帽8192总条目（E/I各4×1024）**。原8192原账不动，工程历史与新账合计≤16384；正式E/I仍0。任何失败、未提交块采样计满实际/保留上界，不补。四块顺序P2/P2/P3/P3，真实源/父组日程、seed660301及原公共初始化相同。

D0锚点权重／Adam不作为训练起点。两臂从共同原始BFM tail＋fresh零FiLM/selector、fresh value/Adam重新构建隔离scratch，四块中每完成更新就由当前策略采下一块；**旧scratch只供D0反事实，不进入新PPO**。标准两epoch只复用本块on-policy数据，不在actor改变后再把旧块当新rollout。辅助buffer按原阶段/8192上限和真实未来mask。

- **D1：各两P2块，共4096新行。** 通过D0后实施相同回溯规则，检验第一真实接受tail/FiLM更新、合法动作、完整IO增长和train保持sentinel。若任一P2块仍没有接受Actor步，完整记录后停止相关资格，不靠多块等运气。
- **D2：各两P3块，共4096新行。** 继承各自D1真实状态/Adam；这补齐原I缺失的P3类型，但不把原I缺格追改PASS。两臂均在新规则下各有2P2/2P3。每臂至少6完整回合，必要来源/停止/释放记实际覆盖，凑不齐则UNDERPOWERED，不越8192。
- 每个新P3块至少一个accepted Actor事务；真实draw上的selector梯度非零、其输出参数实际改变、同状态相同输入的类别概率max变化≥1e-6。该数是待签的可测性线，不要求argmax立刻克服prior，更不要求一定选walking。如果没有足够真实draw/Goal覆盖，判覆盖不足而不是selector算法失败。
- 记录每个真实policy版本SHA；PPO accepted、auxiliary accepted、value更新与真实行为版本分栏，不能沿用总accepted计数冒充Actor版本。

### 4.1 真实物理／恢复门

完整每tick硬界、权限、状态恢复与原raw概率门不变。第一次接受P2与第一次接受P3后，两臂各跑原锁定none/zero/真实release恢复三个train-only sentinel（12完整回合上限）。P2→P3另按原G06执行每臂6个train-only新categorical随机资格完整回合（none/zero/near/mid/stop/continue，按原train稳定顺序固定首个匹配项，兄弟整组；不换父组找好例子），最多再12完整回合。合计≤24完整物理评价，不重复原mean/sample各12与reward24分支。G04联合概率资格在新P3首次正式scratch采样前复核。原合同§5.4保护：none/zero slip≤.10m/s、yaw excursion≤15°、root路程≤.10m；真实恢复slip≤.45、yaw≤15°、同完整状态原生合格参考root路程＋.10m；真实q excess≤.02且不失支撑。三分滑移语义/实际法向力/关节连续性原样报告。当前策略自己释放状态如需相对参考，必须从其真实完整状态计算，不抄另一轨迹历史；这类参考验证计物理与预算，不能进PPO。

至少每臂验证一份**已经接受tail/FiLM/selector且Adam非空**的mid-command完整checkpoint；新进程读回后继续实际12条，再完成一个完整更新，和不中断分支逐值比较：模型/全部Adam/moments/step、全部RNG、λ/计数、history、last-z/action、option/age/slot、Task/callback/solver有效状态、replay、评分与实际物理。验证支路是同一个逻辑后继，不算两个独立训练样本；重建/验证物理全计tick。未完成的随机增量截断另计，不能伪装完整1024工程块。

执行层/Adam首次被接受后的core/delta大小、热更新、真实IO与临时峰值必须重测；不能沿用先前head-only .743GiB作存储PASS。原lossless/atomic/不可覆盖停止包不变。物理硬界／保持失败不改门，停止相关臂；共享实现错误两臂停。

## §5 放行／失败三列表（预锁）

|门|明确通过证据|不通过与唯一后续|
|OR0 归档与资源|Yi签本稿、固定SHA/参数/起点；实时单GPU串行、free≥100GiB＋峰值；资金/输出/tick到账|未签或护栏未过只允许文档，不跑D0|
|OR1 同提议|两个归档锚点原首步KL逐值同；α1直接copy；五α及全Adam事务表完整|复现／事务失败STOP_CORRECTNESS；没有完整GPU网格不称通过|
|OR2 可接受执行步|两锚点存在满足原KL＋目标不增的候选，S_mu与q_PD工程线过|无步/步太小→停止该优化配方，交分层替代稿，不加α或改sigma|
|OR3 新P2|每臂2新块均有真实接受Actor；实际on-policy、保持与安全、首sentinel通过|失败分类为优化/质量/数据；不按loss下降继续硬跑|
|OR4 新P3|两新块、足够完整实际回合；selector梯度/参数/概率变化与真实draw证据|draw覆盖不足→UNDERPOWERED；有覆盖但selector不动→联合更新阻断候选；不调prior找补|
|OR5 完整恢复|接受执行Actor后的两臂新进程12条＋完整更新逐值一致；所有原严格断言|失败STOP_RECOVERY；不以head-only历史PASS代替|
|OR6 全程可负担|各模式实测max×1.25，含真实accepted-Adam IO/临时包、64k、48sentinel、100Dev/8消融、失败reserve的全部上限满足|只交有界资格结项；预算或存储未过，原32k仍BLOCKED，不缩评价或假称成本已测|

OR6通过仍只交本次资格结果与正式发车建议；本次授权范围不直接启动正式32k/臂。

## §6 账本、有限上限与完整成本投影

### 6.1 已发生（原账不重置）

|项|原始实际／保守账|
|---|---:|
|v2 stage已计|7807.703624s|
|其中P1|3837.064558 /4500s；余662.935442s|
|旧scratch含失败|8192 /8192|
|P1输出|约.743 /.8GiB；仅拒Actor路径|
|全部真实/保留5ms ticks|201096 /950000|
|旧R2累计|29345.631881 /57600s；余28254.368119s|
|另已批6000s|未使用|
|项目240h|仍适用；混合旧账尚无可证明的去重总累计，不虚报剩余额度|

**本次只读与文档400秒整体保守上界另列新review账**，覆盖两次CPU包读取、工具启动/IO与归档；不修改原ledger.json或把它写成GPU实测。计入之后R2参考累计29745.631881s，旧R2余额27854.368119s。若本稿不签，这400秒也是已授权文档成本，不会消失。

### 6.2 待签的局部调整（所有调整集中）

|条目|已签旧值|本稿建议|影响|
|P1 job段帽|4500s|10100s（＋5600）|历史3837.064558保留；以下新D0/D1/D2/修复总≤5600，余原662.935442不作为自动续训额度|
|P6 job段帽|1000s|1400s（＋400）|含本稿review400；不挤掉原收尾余150s|
|未来stage总帽（不含旧400稿件）|41000s|47000s（＋6000）|这是完整阶段上限，不是现在启动正式训练|
|P1工程真实行总帽|8192|16384（旧8192＋新≤8192）|新E/I各4×1024；失败计入，无追加资格轮|
|P1输出段帽|.8GiB|2.2GiB（＋1.4）|仅允许新资格的模型/Adam/增量/恢复/失败包；单项超限即停|
|stage总新输出帽|7GiB|8.4GiB（＋1.4）|正式段其余存储分配不偷偷移动，原R2 12GiB仍独立约束|
|本次新增全流程物理|无额外授权|≤160000个5ms ticks|含采集≥32768、prefix/reset/参考/24个完整资格与sentinel/恢复/失败；原stage950000帽不变|
|新增额外job资金|另已批6000s|**再申请5000s**|旧57600不改；不自动从项目240h透支|
|纯工程修复池|800s，已用677.307783|不增，余122.692217；本次最多预留100s|超出或科学规则需变→停报|

建议新工作包分账：review400（本轮已发生／保守）；D0≤900；D1≤2000；D2≤2600；唯一等价修复≤100；合计≤6000s。子帽和总帽任一先到即停。所有失败、冷启动、恢复/IO、评价、CPU子进程纳入，不重复计嵌套墙钟。预算不能借来增加α、父组或正式条目。

资金对账：原可用未来资金36062.071743＋已批6000=42062.071743；修订完整stage47000差4937.928257，故申请5000留62.071743总资金余隙（原3000 failure reserve已经在47000内，不能再加一次）。包括旧400文档的完整计划为47400。后续正式阶段仍必须OR6实测投影，不能用这62秒当充足风险余量的证明。

### 6.3 单卡Pro5000投影与未知项

真实原热块：collection43.55–62.85s、update29.81–33.91s、IO2.35–6.93s；全块82.31–100.93s、10.15–12.44真实50Hz行/s。E/I全作业492.80/398.27s；新进程恢复194.84/196.48s。它们含实际V/aux和被拒PPO，**未测接受Actor moments路径**。

|未来项|现有证据与保守估计|不能冒充的结论|
|D0两锚点×五α|10次整块policy/KL/输出测量，非10次Adam；900s硬顶|未实测；不引用α²估计为通过|
|新8个1024块|旧全块82–101s只是低估；加入最多五次候选KL/目标/输出及Adam IO，工程暂估150–240s/块，约4.3–6.8行/s|不是已测新吞吐；第一个成功块即重投影，不赌未来加速|
|采样＋更新8块|约1200–1920s热成本，冷启动／双恢复约650–850s，24完整资格/sentinel约800–1400s|高端合计约4170s，接近D1＋D2=4600；首对成功块若保守投影装不下即停，不删sentinel|
|完整64k真实正式行|约64个上限1024块，热成本约9600–15360s；另48sentinel、prefix/cold/真实恢复，P2+P3暂估13500–20500s|旧P2+P3合计14000s仅低端能装；当前不能承诺32k按旧局部帽完成；原950000 tick也必须按实际manifest重新核全部prefix/评价，不能只减64k×4|
|100Dev＋≤8消融|完整时长、source与冷启动/IO逐spec计；原P4=12000/P5=2500仍硬顶|未实际执行，不把2s probe吞吐当完整回合吞吐|
|accepted-Adam存储|约5.23M trainable参数的FP32参数＋两moment仅张量下限约62.8MB/臂，不含V/引用/数据；tail每次变化使delta压缩与旧路径不同|旧P1 .743GiB不能证明正式P2/P3的2.5GiB分配足够；第一对成功块测无损增长/临时包后OR6裁决|

给的是范围，不是保证。若OR6显示原正式段14k秒／2.5GiB或全stage/tick资金不够，**本稿到有界资格即停**，交一次完整正式预算修订，绝不边跑32k边申请。也不以未实测的批量物理／GPU占用率把估计改成乐观值。

## §7 有证据的切换条件：分层更新备选（只设计，未授权执行）

不是因cat KL小就改算法。依次区分：

1. D0无满足KL/可测动作的候选→当前Adam方向/宏观密度尺度资格失败；不证明探索或reward失败。
2. D0通过但新P2在实际新块连续被拒或几乎无执行变化→有限回溯没有解决可训练性，停止该配方。
3. P2可动，P3同块有足够真实draw和非零selector梯度，原提议cat本身很小、但Gaussian导致整事务回滚；或累计两个P3块selector概率仍低于1e-6可测线→支持**优化耦合/尺度**问题，优先分层信用与独立提交设计，不调prior/std。
4. 主要缺FOLLOW/替代option draw→判覆盖不足；不宣称联合KL禁止selector。完整报告实际父组/阶段/option，不用理论期望次数替代。
5. Actor与selector可测学习且物理安全，但任务未改善→该资格本身通过；行为信用／探索是否足够仍未证，不能继续缩小步长当药方。

备选训练设计须另审：

- 低层每20ms对实际29D raw Gaussian算ratio与native GAE；在采低层动作之前，以当前真实held option/age/z条件构建训练期V，不能泄漏未来。保持按真实时间积分的reward与gamma20/lambda20。
- 高层只在真实每0.5s option抽样边界更新categorical，用实际持续至下一draw/终止的SMDP折扣return与独立高层advantage；不能重复五次held option。100ms命令控制和真实z插值保持，Goal变化不偷开新draw。
- 低层/selector独立提议与完整事务，避免低层超KL无条件抹掉高层可接受更新；共享Goal/表示参数明确唯一Adam所有权及更新顺序。若拆V或优化器拓扑，明确是下一科学修订，不叫记录修复。
- 原100ms joint KL .02仍作共同外层物理控制分布检查，不能仅把它改成每20ms .02以实际放宽五倍。一个可审的预算分配示例为每20ms Gaussian均值帽.0032（五步.016），每真实draw categorical帽.02、约每五个macro一次（平均.004），并仍重算真实整块joint≤.02；这只是**待审备选**，不替代真实非均匀draw统计，不在OR1自动采用。
- 同预算/同物理/同权限的有限比较才判断是否更好；目前没有20ms方案的真实收益证据。

没有分层方案授权就只交修订稿并停，不自动开新网络、critic warm-up、Teacher/Q考试或扩大技能。

## §8 签署栏与交付

待Yi一次性确认：固定五α；真实Adam参数＋moment事务；minibatch原目标不增条件；D0执行可测线；P3概率可测线；新采样≤8192；5600s新资格＋400s文档；P1/P6/总帽及输出调整；新增资金5000s；正式32k仍不自动发车；分层替代仅提案。

- 批准人：Yi
- 版本：OR1 v0.1
- 当前状态：**待签；本轮未执行新优化／采样／物理**
- 通过后出口：D0/D1/D2实际资格报告、完整原始失败与checkpoint/runtime SHA、OR6真实成本，不以资格通过宣称已获得接触行为。
- 失败后出口：明确一个首阻断及一份可判别的分层替代合同；不无限再开资格轮。
