# R2 E/I v1.1 — 资格门语义三列表与自动诊断分支

**状态：DRAFT / REVIEW ONLY。** 是 [R2_EI_TRAIN_CONTRACT_V1_1.md](R2_EI_TRAIN_CONTRACT_V1_1.md) 的规范性组成部分。v1.0已实际批准，但首个sample按原5ms总变差门失败并暂停；不能因为此处发现定义错误而追改其历史判定，也不能未经Yi重新签署就重启物理/训练。

## A. 先把门量的物理意义说清楚

目标防的是未经Standing/Hold公开Goal允许的大幅偏航、多圈自转、真实滑移、失支撑、越界与无意义行走；不是因为探索噪声使机器人几度反复摆动就把TV加到几十度，然后叫它“转了几十度”。

设 θ(t) 为5ms真实pelvis yaw的时间连续unwrap序列；新硬门 M=max_t|θ(t)−θ(t0)|≤15°，窗口t0必须分别是online执行起点或**真正接触释放**后合法恢复接管起点。另存 TV=Σ|Δθ|、net=θ_end−θ0、range=maxθ−minθ、p95 |dθ/dt|。小幅摆动可以TV高而M低；大幅往返或整圈旋转的M仍高。旧任务末世界绝对yaw≤20°依然存在，是另一个指标。不得用wrapped末yaw替代M。

root路径 P=Σ||p_xy(t)−p_xy(t−1)||，捕获往返走动；净位移 N=||p_xy(end)−p_xy(start)||不能代替P。none/zero P≤.10m；true-release恢复P≤同完整状态、同Goal冻结零U合格参考P+.10m，缺同态参考标UNMEASURED，不能用其他父组/其他模型Reference偷换。

none脚本末4秒**不是**真实release；真实post-release必须有先前真实接触与解除事件。原真实q硬界、无支撑、loaded slip、active有效推进与任务参考不因yaw语义修订放松。TV高/角速度异常仍触发单次解释性诊断，但TV本身不是“持续偏航”的15°硬门。

## B. 三列一致性注册表

第三列的PASS/FAIL必须带实际身份和证据等级。若没有观察过真实符合或失败的样本，明确写“缺实测”，可以给存档推导或单元fixture作为**逻辑验算**，但不可充当真实E/I资格PASS，也不可据此签成“已通过”。原研发尚未取得新Q跨4父组合格或I的BI收益，这些只能是未来结果门。

| 防什么（门ID） | 怎么算：物理量、窗口、阈值 | 已有数据各一PASS和FAIL／缺证状态 |
|---|---|---|
| G01｜共同零U初始化与E/I物理一致 | E/I全新U同seed、输出零，target/Actor原尾层一致；同完整源首状态z/action/qPD/Q输入逐值相同；禁止继承A128。 | PASS：v1.0入口E/I全新U动作和Q输入逐值一致；FAIL历史反例：旧A128 U导致none连续偏航1081.13°，不符合本轮零U初始化要求。 |
| G02｜信息权限与父组去重 | Actor/U隐藏伙伴与未来字段的独立扰动不改输出；train/dev/sealed真实source/partner和stop兄弟整组，前缀物理逐值一致。 | PASS：v1.0入口因果隐藏字段干预和stop/continue前缀通过；FAIL构造自现存数据：将同stop父组兄弟误作两个独立父组，真实parent_id相同，应被拒绝；该错分并未实际发生。 |
| G03｜动作采样与实际replay合同 | std=.01,clip=.06，一次29D噪声；真实执行action==replay action==qPD反算原映射，同存档RNG；区别采样和mean。 | PASS：v1.0入口480个sample控制验证动作/采样一致；FAIL案例缺实际动作不等记录（历史只是实现错误/恢复错误而非采样改变）；以同现存row篡改一维replay用于可重复拒绝测试，必须注明故障注入而非物理事实。 |
| G04｜真实q硬界（不是qPD限制） | 每实际5ms qpos相对机器人机械限位计算 excess=max(max(qmin-q,q-qmax,0))；原控制硬终止0.02rad不变；qPD仅作附表。 | PASS：v1.0 sample none excess=0；FAIL：B19 mean左踝pitch excess=.027559rad，真实触线；历史qPD超界但真实q未越界不能判此门失败。 |
| G05｜足支撑真实丢失 | 每个真实5ms physics tick读取实际support；none/zero/恢复的no_support_ticks必须0，不能用时间平均或末时刻掩盖。 | PASS：v1.0 sampled none 0 tick；FAIL：B19 adapter-on release段16个no-support ticks。 |
| G06｜无接触/零需求足滑移 | 在整个online none/zero窗按原stage_loaded_slip_mps/原载荷口径逐5ms取peak≤.10m/s，另报p95/越界tick/法向力。 | PASS：v1.0 sampled none peak=.007856m/s；FAIL：旧A128 none peak=1.566231m/s。 |
| G07｜无接触大幅偏航/自转（不误罚小幅抖动） | 以online首5ms为t0，从实际pelvis矩阵取yaw时间unwrap；M=max_t\|yaw_u(t)-yaw_u(t0)\|≤15°；TV/net/range/p95角速度只作不同意义指标。 | PASS：v1.0 sampled none yaw range=3.730°，所以M≤3.730°，即使TV=39.323°；FAIL：旧A128 none unwrapped net=1081.130°，所以M≥1081.130°。 |
| G08｜无需求时root多余往返行走 | online全none/zero 5ms root xy逐步路径总长sum \|\|delta p_xy\|\|≤.10m；记录净位移但不能以净位移替代路径。 | PASS：v1.0 sampled none path=.07726m（net仅.00225m）；FAIL：旧A128 none path=4.31614m。 |
| G09｜真正的恢复标签 | 实际有contact建立然后release且force实测归零，才把release之后的真实窗称post-release；完整源/自身z/物理和Goal要能同状态比。 | PASS：R2 mean near实际release9.8→13.8s并存档后继；FAIL：v1.0 sampled none始终无force，脚本尾4秒绝不可当release恢复Teacher。 |
| G10｜恢复时大幅偏航/回转 | 仅在真实release起或合格完整状态接管起窗口，M=max\|yaw_u(t)-yaw_u(t0)\|≤15°；旧20°世界末yaw另报。 | PASS：R2 mean near真实释放后旧TV=1.869923°，数学上M≤TV<15°；FAIL历史：Return64 P2恢复3s净偏航−136.972°，M必≥136.972°（不同实验，只作量尺反例）。 |
| G11｜恢复时无谓root游走 | 同完整物理状态、同Goal和相同时长3s/4s，候选实际root path≤合格冻结零U同父组参考path+.10m；无合格参考标UNMEASURED，不跨状态偷换。 | PASS：R2 mean near已资格参考自身release path=.012293m；FAIL风险实测历史：A128 near释放阶段root实际路径1.9655m vs零U near释放路径.0123m；两策略抵达release时的物理状态不同，不能直接拿来判本门的**同状态恢复相对资格**。此门尚无合格同态真实FAIL案例，必须标UNMEASURED，不以旧Return报告的root净位移冒充路径。 |
| G12｜恢复时真实滑移及支撑/关节 | 真实release后时窗loaded slip peak≤.45m/s，原q excess≤.02rad、no-support tick=0；按相同5ms窗和原法向语义。 | PASS：R2 mean near release slip=.001195、excess=0、support缺失0；FAIL：Return64 P2真实恢复slip=.7843m/s（历史物理，非新E/I）。 |
| G13｜同状态同z的Actor功能保持 | 只在已物理资格train保持域计算ΔqPD=新Actor−原Actor，父组p90 RMS≤锁eps_rms=.015rad、p95单关节≤.030rad；主动接触推进不强制原生。 | PASS：新E/I初始tail=原BFM，ΔqPD=0；FAIL证据限制：旧B19末状态存在−.349rad单帧adapter增量，说明为何应检查，但它不证明R2整父组p95失败。整门失败例暂缺。 |
| G14｜U不在无接触/已资格恢复偏离Goal | 原公开standing合法norm z与当前真实U命令角距离按父组p95，无接触≤5°、已资格恢复≤10°，active不加归零门。 | PASS：新零U命令与standing角差0°；FAIL缺实测：A128对物理旋转有害但其U角距离p95未从存档算出，不可用yaw1081°冒充；要求签前只读补例或注入同现存z的6°/11°数值拒绝fixture并标级。 |
| G15｜保持对偶统计不被单帧支配 | 每独立资格父组p90相对eps，父组等权求normalized violation，锁定λ_next=clip(λ+.05*(violation−1),0,3)，不看Dev重定参数。 | PASS：v1.0冻结初态Actor偏差0，lambda初始化0且应保持0；FAIL历史机制反例：Return=.0008来源单B19状态梯度尺度而非跨父组标定，按本轮禁止；不是对偶代码实测错误。 |
| G16｜Q非正且后继计算合法 | −softplus twinQ输出≤0，实际contact_physical reward非正、真实a_exec/z、duration、termination/truncation、targets按训练合同一致。 | PASS：新R2工程Q负输出结构及真实update/恢复资格；FAIL：旧B19在3058/4750 replay上Q>0，绝对值不可信，不能作为新教师。 |
| G17｜Q真实排序有资格驱动Actor | 已见至少3500至多5000真实online transitions，真实none/near/mid/zero/stop+continue及release回放；≥4独立train父组同状态可执行首动作＋共同后续的非平凡pair排序≥70%，不足证据为UNDERPOWERED。 | PASS整门尚不存在，因为新R2正式训练=0；仅D1 P2 Control vs Return等单pair方向一致，不等于整门；FAIL类证据：旧D1旧Q仅7/12=58.3%、且仅2父组，若用相同尺度评估不得放行。此门结果只能在新Q warm-up后实判，签时不假填PASS。 |
| G18｜不再用固定旧Q离线长时间更新Actor | 首250真实数据只Q4；其后warm-up Q4/I表示2/Actor0；Q资格后每块250 Q4/I表示2/Actor+U1，下一Actor更新前须有新真实物理块。 | PASS工程局部：E/I真实250在线采样+更新和恢复可运行，但原工程步数不是v1.1节奏资格；FAIL历史：旧Return在无新rollout及Q更新时离线64次Adapter更新，本合同禁止。 |
| G19｜每轮真实恢复+事务存盘 | 250样本+对应完整优化后写不可变core+delta、Adam/replay/Task/solver/历史/RNG，hash/fsync/原子指针；新进程物理逐值续跑；故障包拒载。 | PASS：E/I工程250+5条与更新新进程逐值一致；FAIL：Issue56 97D训练512更新后末轮未保存，只剩step0。v1.0合成存盘测试不算正式E/I全链恢复资格。 |
| G20｜时间、磁盘、GPU和全physics计数 | R2 cumulative≤57600job秒、≤12GiB，已记5544.704秒/.814GiB；GPU0重作业串行，共享free≥100GiB+峰值余量；reset/前缀/失败/tick全计。 | PASS：R2现账在16h/12GiB内；FAIL历史边界：S1上限45k总physics ticks、已知实耗≥72,280，说明漏计reset会失败；这不是R2已超额。 |
| G21｜主动接触任务真实完成而非力小就算 | near≤4cm/4N、mid≤8cm/7N、stop hold≤8cm/6.5N、真实推进、slip peak≤.45m/s、末绝对yaw≤20°和原边界，按active窗/相同父组。 | PASS组件：旧零U near误差3.86cm/力3.14N，但因推进不足，整任务仍FAIL；FAIL整门：零U mid20.33cm/8.72N且推进不足；旧F1 L1有11个已暴露参考合格窗口但非新E/I学习成功。 |
| G22｜BI真有表示增量而非参数更多 | 冻结训练后I，在跨父组heldout未来.5/1s接触后果probe中比较BR/BI/concat/currentContact，配套公开Goal prompt disable/shuffle消融，先物理增量后表示。 | PASS真实E/I增量样本不存在（I尚未开训），不能伪称；FAIL机制实例：仅固定connected prototype/无正负标签变化时必须判probe_underpowered，不能算BI成功。此门为未来结项判据，不是签前当前模型PASS。 |

## C. 每道门都预写失败后的唯一合法动作

**仅在Yi明确重新签署v1.1后生效**。任何**硬门FAIL**先立即停止相应job并记录；单门只准执行一次安全诊断/冻结对照，诊断后默认STOP_AWAIT_DECISION，**不因诊断通过而自动将主门改为PASS**。**例外是非阻断警告**：当旧TV>15°而新yaw_excursion≤15°、且全部其他硬门均通过时，自动运行一次只读yaw/rate/root分解并记录WARNING_EXPLAINED，随后允许沿原已批准流程继续；绝不能把这种warn当成硬门FAIL或忽略其他真正安全失败。如需改科学方法、阈值、奖励、seed、硬安全、Dev/Sealed或共享资源，则停止问Yi。

诊断池总上限 **3600 job秒、1GiB新输出、18000个全流程5ms physics ticks**，均包含在R2已有 **57600秒/12GiB** 内而非另加。每个物理对照最多2条×3秒，单job限600秒，含加载/前缀/恢复/失败/IO；先用现存归档。共享free≥100GiB、GPU0单重任务串行；预算/状态资格不足则只读报告，不强制开新物理。原240小时总账继续累计，所有失败保存。

| 门失败ID | **只能自动执行的一步**（诊断／冻结对照，执行后停） |
|---|---|
| D01：共同初始化失败 | 只读重新比较E/I全U/Actor/Q张量、原A128和零U首控制；不换seed、不加载B19、不改网络。 |
| D02：权限或父组污染 | 只读字段依赖/真实source与partner parent/stop sibling配对表；停止使用污染数据，不换父组。 |
| D03：sample/replay/PD不一致 | 同已存输入RNG离线一次前向和原g(a)逐值复算；普通工程修复须新源码锁并重过原门，不能跳门。 |
| D04：实际机械越界或失支撑 | 立即停物理和相应臂，保存完整physical/solver/action/qPD/q；只读首失败边界，不自动重跑危险动作。 |
| D05：slip超保护 | 只读5ms有效载荷/材料点/接触部位和持续时长；必要时在预算/安全与完整状态许可下同状态1个冻结mean≤3s物理对照，不调法向门。 |
| D06：yaw大幅偏离/TV与偏离矛盾 | 同已存5ms真实轨迹重算unwrap/max/net/range/TV/角速度；若轨迹记录确实不完整才允许同train状态一次冻结mean/sample配对，不改阈值、reward或噪声。 |
| D07：root路程/恢复多余移动 | 同窗5ms path/net/velocity/foot steps分解，检查是否错用source或换参考；最多一个合格同状态冻结zeroU回放，对照不合格判UNMEASURED。 |
| D08：none误归release | 只读contact active/force/event时间线，降格为no-contact hold标签，不用假release作为Teacher。 |
| D09：qPD保持或U角度不合格 | 同状态同z、跨父组分位统计；用已冻结原Actor/零U原样一次对照（train-only、完整输入），不调整对偶和采样。 |
| D10：辅助对偶梯度/统计异常 | 只读每父组mask、p90、梯度、λ轨迹、实际训练字段；不从单状态重新标λ。 |
| D11：Q可信排序或数据覆盖不合格 | 停止Actor/U更新；只读Q符号、真实后继、TD、action support及共同续跑证据。额外Q-only优化属训练，不是本自动诊断权限；需在资格判定前按锁定warm-up安排完成。 |
| D12：更新节奏/信息来源违背 | 停止当前Actor优化作业；只读checkpoint版本与250条新后继证据；不靠旧replay连训修补。 |
| D13：完整checkpoint/恢复失败 | 离线manifest/hash/delta坏包注入和最后完整已提交轮次新进程读回；无新危险物理，不删原失败，不减Adam/replay。 |
| D14：预算/空间/其他GPU占用 | 先安全存盘退出owned job，仅只读资源和完整ledger/p90 IO投影；不自动扩大16h/12GiB。 |
| D15：任务不合格 | 按原主动接触评价表复核力/误差/真实推进与slip/yaw Pareto；若硬物理事件先D04；不以loss/低力补过判据。 |
| D16：I表征探针不足/行为无增量 | 停止I晋级；只读标签分布、父组泄漏、BR/BI容量/公开prompt的离线消融；未经额外资格不自动触发新物理或更新。 |

## D. v1.1 的签署检查

v1.0入口已计R2累计5544.704秒/.814GiB，新合同绝不能改成原草案3088秒起算。签署前由Codex将本表转换成机器可读gate_semantics_review.json，对每一门登记：gate_id, phenomenon, raw_field, frame, time_window, unit, formula, threshold, comparator, parent_key, pass_source, fail_source, evidence_grade, status, planned_diagnostic, max_cost, stop_action。每个证据来源要能查到commit/SHA及具体数值。

本规范的可复用约束见 [RESEARCH_GATE_SEMANTICS_STANDARD.md](RESEARCH_GATE_SEMANTICS_STANDARD.md)，适用于后续所有研究合同。

**不能签的情况**：一列测的不是另一列想防的现象；把历史别的模型的类似案例称作本新模型实测通过；缺现存真实PASS却写“已通过”；把采样none末窗冒充release；把qPD跳变当实际关节瞬时变化；把Q的单pair正确当跨4父组70%通过；把BI probe未跑当方法成功；失败分支没有时间/空间/physics上限或自动改方法。

**门可合法在未来才有结果**：G17 Q warm-up与G22 BI probe在签署时状态必须是PENDING_UNMEASURED，合同可以预注册将来何时PASS/FAIL/UNDERPOWERED，但不允许在模型真的通过前让Actor越Q门，或让I越BI晋升门。G11只有历史同态类比，没有冻结零U的候选真实失败例；G13/U角度仍需read-only提供真例或签署中明确“经验FAIL未有、只能用故障注入检查代码”。严格意义的“三列实证全部齐”现在仍未闭环，v1.1**不能假称已取得全门实证PASS**。请先审阅再决定是否签署。

**当日检验示范**：R2实际none采样TV39.323°按v1.0真实FAIL；新M至多等于yaw range3.730°，满足≤15°；旧A128同条件net1081.130°，因此新M必>15°。两份原报告字节不改，新合同只用于重新申请未来实验资格。

## E. 版本与引用

事实基准：[R2_EI_ENTRY_RESULTS @b8bc894](https://github.com/luoxinyuan/compliance_BFM/blob/b8bc894e97bd660895518995326d0dbfbb57be5d/research_notes/R2_EI_ENTRY_RESULTS.md)、[R2_EI_TRAIN_CONTRACT_V1 @b8f62f1](https://github.com/luoxinyuan/compliance_BFM/blob/b8f62f1/research_notes/R2_EI_TRAIN_CONTRACT_V1.md)、[原生零U结果 @d12a0e6](https://github.com/luoxinyuan/compliance_BFM/blob/d12a0e69f2923785c6897a5527a43c8c93913173/research_notes/R2_ZERO_U_RESULTS.md)、[mean参考](https://github.com/luoxinyuan/compliance_BFM/blob/gaoe05/research/research_notes/r2_ei_entry/mean-reference.json)、[B19](https://github.com/luoxinyuan/compliance_BFM/blob/bf4b857df47a893a2822b01bc20f90eb9d928a1f/research_notes/INTERACTION_PILOT_B_FAILURE_RESULTS.md)、[Return](https://github.com/luoxinyuan/compliance_BFM/blob/f6bd9eb/research_notes/ADAPTER_RETURN_PILOT_RESULTS.md)、[D1](https://github.com/luoxinyuan/compliance_BFM/blob/gaoe05/research/research_notes/R2_D1_RESULTS.md)。

本稿只推送到GaoE05研究桥接仓库，因为目前GitHub集成对luoxinyuan/compliance_BFM的内容写操作没有权限；由Codex在Yi签署后审计并同步到gaoe05/research，不能宣称已经写入执行分支。
