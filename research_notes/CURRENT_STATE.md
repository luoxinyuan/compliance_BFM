## 2026-11-19 无力站立同状态动作保持，小型配对对照

U 最后一批各 8 组更新；可见静站起点上的无力、持续和提前停止指标见[完整判读](CTFT_QUIET_RETENTION_PILOT.md)。

## 2026-11-19 安静原生站立共同前缀对照完成

单父组六回合，U降低持续牵引的腕力和误差，同时在零力时自行移动和偏航，三个分支均未过承重质量门。见[实际力、无力对照和首个控制差异](CTFT_QUIET_STANDING_RESULTS.md)。

## 2026-11-19 冻结 U 伙伴竖直顺从性对照完成

8 个训练祖先、三分支。见[真实力方向、腕肩增量、支撑和停止恢复](CTFT_VERTICAL_COMPLIANCE_RESULTS.md)。

## 2026-11-19 Issue 69 冻结 U 小幅命令对照完成

8 个训练祖先、24 对完整条件。首秒 0.15 球面比例，实际推进匹配 7/8，裁决 NO_MATCHED_QUALITY_GAIN。。实际引导方向须与三维腕力分开解释。见[存档审计、冻结物理结果与单一下一项决策](CTFT_D2_69_RESULTS.md)。

## 2026-11-19 CTFT-D1 前瞻诊断完成

8祖先24条冻结U参考在新进程逐tick复现。首动作3个可辨别祖先；1s walking提升推进但0/8推进匹配，无力也走。维持 UNDERPOWERED，不启动训练。见[前瞻结项与分阶段表](CTFT_D1_PROSPECTIVE_RESULTS.md)。

## 2026-11-19 CTFT-D1 诊断资格

Issue 68：训练动作档案与终点U确定性接触参考身份不同，合格接触祖先 0/至少6。D1/D2未运行。见[逐事件表、证据与新增参考建议](CTFT_D1_DIAGNOSTIC_RESULTS.md)。

## 2026-11-19 BFM-CTFT v1.1 固定终点已结项

结项阅读入口：[机制解释与下一项判别实验](BFM_CTFT_MECHANISM_AND_NEXT_TESTS.md)，含无力漂移、抖腕、早期承重失败及通用奖励的适用边界。
两臂各 131,072 条，252 个正式评价条目。静止源同态有力相对无力的身体前进增量中位数 U +54.2 cm、T +58.1 cm；持续牵引联合质量门两臂均 0/24。承重滑移是重复出现的早期失败，T 额外降低牵拉但停止与释放更差。两臂均新增原生硬停。保留反馈正例，两臂不晋级。见[正式结项](BFM_CTFT_V1A_CLOSEOUT.md)、[配对结果](BFM_CTFT_FIXED_ENDPOINT_RESULTS.md)与[恢复勘误](BFM_CTFT_ENDPOINT_RESTORE_ERRATUM.md)。

## 2026-11-18 v1.1 已签署，U通过恢复并续训

共同8192条暖身包完整别名启动。U 首个1024条及8组联合更新已提交，正式计数9216。新进程验证的12控制和一次更新逐值一致；修复 Adam 内存快照别名后，完整磁盘恢复再核通过，已进入持续训练。各臂新收122880条，U 完成才串行运行 T。旧 v1 物理失败与 v1.1 第一次恢复的工程失败分别保留。见[恢复记录](BFM_CTFT_V1A_RESUME.md)及[签署锁](bfm_ctft_v1a_signed.json)。

## 2026-11-18 U 硬停已逐值复现，修订候选通过20个源窗口

同源同扰动下，原生 B0 也触发同一左踝硬门；U 均值分支随后连续55ms无有效承重，单独关闭噪声不足。冻结 B0 均值采集通过20/20 train-only窗口，含原失败源，最长无有效支撑50ms。建议 v1.1 在两臂保持 g1 原采样，将 g0 采集切到冻结 B0 均值，并从同一8192条暖身包开展匹配对照。已有17,408条 U 正式提交与失败证据保留，T仍未启动。签署修订前不启新的正式臂。[诊断](BFM_CTFT_V1_U_HARD_STOP_DIAGNOSIS.md) · [候选合同](BFM_CTFT_V1A_MATCHED_RECOVERY_PROPOSAL.md)。

## 2026-11-18 正式 U 触发左踝机械安全门，T 未启动

16k资源复核通过。U 最后完整提交17408条，136组完整更新、72组含Actor。其后605条已物理完成但未提交；无伙伴力的第33回合，左踝 roll 实际关节角超出机械下限0.021228rad，高于0.020rad停线。双脚有支撑。已同源只读复核，原生轨迹同一时刻未出现关节超限；不同动作历史下无法归因为单一训练机制。最近提交包和独立失败包读回均已完成。T未启动，未出统一终点评价。见[实际硬停记录](BFM_CTFT_V1_U_HARD_STOP.md)。

## 2026-11-18 正式U已启动，T按成功条件串行排队

S0已完成并机械锁N=131072/臂；250GiB追加已批准，含轨迹/评价余量的保守投影242.48GiB。独立正式runtime `4f1ce5d`，从原始BFM/fresh五Adam/空replay启动U；同一GPU0只有一个重作业。当前尚处冷加载/初始存盘，不能说已经取得正式学习或物理收益；U失败不自动启动T，U固定终点完整成功后才运行同N的T。首16k按实际IO/轨迹增长再次审预算，不按reward变更规模。

持续运行账本为stage的 `runs/formal-pair-01/ledger.json`；原资格账本是启动前snapshot，不能当当前采样进度。S0实际报告/资源锁/全部errata见下。

## 2026-11-18 S0完整恢复通过；250GiB追加已批，共同N锁定

真实8192新行/64组Actor冻结＋两完整联合轮、完整PhaseReplay/Task新进程恢复、620控制跨实际reset后的下一联合更新均逐值PASS。scheduler少计12控制的工程缺陷单列erratum、旧包保留，新独立修正包回归通过。原BFM/PD及数学/安全门不变。

Yi依次批准输出120→200→250GiB；加入训练物理轨迹和评价各8GiB预留后，131072/臂保守输出242.48GiB、时间约6.78h/臂；更大N超时，已在正式训练结果/新评价前机械锁N。当前资格结算时正式U/T尚未开始；下一步按已批整体合同从原BFM/fresh五Adam/空replay串行U→T，首16k复测全链成本与存储增长。不继承任何S0工程学习权重。

[完整S0报告](BFM_CTFT_V1_S0_PLUS90M_RESULTS.md) · [独立计数勘误](BFM_CTFT_V1_SCHEDULER_ERRATUM.md) · [250GiB批准](BFM_CTFT_V1_OUTPUT_CAP_250GIB_SIGNED.md) · [N资源锁](bfm_ctft_common_N_resource_lock_v4.json) · [实际账本](bfm_ctft_s0_plus90m_closeout_v1.json)。以下历史状态和全部负结果保留。

## 收尾计费勘误：S0分项超时已停止

此前28795秒是预结算，61.475秒发布预留不足；最终保守结算 **29023.027s／28800s，超出223.027s**（含本次90秒纠正／发布预留），已完整进入stage／项目总账。超时发生于Git／Issue／Master收尾与核验，无新GPU作业、采样或训练；40h总帽未超。原预账本保留，不声称本次全部在S0帽内。详见[独立计费勘误](bfm_ctft_s0_closeout_overrun_erratum_v1.json)。

## 2026-11-18 S0追加两小时结算：严格恢复已通过，完整正式环路仍未发车

Yi批准S0六→八小时已归档。原生CPR仅12训练祖先／144767参考帧；实际6条采样资格全过（两个训练祖先，2700控制）。Packed完整中断／事务写失败／新进程下一更新逐值通过。初次batch摘要差查明仅为未进网络的time元数据别名：11/12行差1，实际模型／Adam／梯度／物理从未分叉；修为步进前冻结输入，旧失败不重判。实际已用梯度缓存清空与不清空的完整未来执行／更新、新进程存盘恢复亦逐值通过；未删有效梯度或Adam。

足底固定材料点：78完整支撑段均可测，37段过原双阈值，12条完整native源均各有超界段，不能称完整质量已通过。最终source身份／恢复分层锁为12早静止＋6近过渡＋6实际已移动，24独立祖先；不是24条contact成功。

**S0按追加后的8h分项预算停止，正式U/T=0，N仍null。** 下一轮8192新行／64 Actor冻结＋两完整round试车因剩余时间小于其1200s＋收尾预留，被预检拒绝；不缩减找补。仍缺PhaseReplay＋TrainingLive完整联合恢复、正式每轮保留器、伙伴终检清单和N资源锁；不把代码准备写成完成。[完整结果](BFM_CTFT_V1_S0_PLUS2H_RESULTS.md) · [实际账本](bfm_ctft_s0_plus2h_closeout_v1.json)。以下历史状态原样保留。

## 2026-11-17 Yi追加批准S0两小时，资格继续

[S0追加签署](BFM_CTFT_V1_S0_PLUS2H_SIGNED.md)：分项上限现8h，stage40h等总帽及科学门不变。下列预算暂停为批准前中间事实，现已恢复资格工程；U/T尚未启动、N未锁。

## 2026-11-17 S0续跑结算：尺度与严格恢复通过，生产资格未闭合，尚未长训

共同aux标准差已校准并锁定 **0.8752717972 FP32**；A1此前已批准且执行，无待批准科学项。完整接触Task＋五Adam/模型/RNG/replay的新进程短程继续与下一更新逐值通过。另完成3072真实工程采样＋24完整联合更新，单环境采样热态约16–23s/1024行，8次更新约1.25s，完整存盘47–56s。4worker跨batch数值门失败保留，不改精度/容差。

12训练祖先与28新候选祖先无交叉；选中12早静止＋12真实转换附近源状态 **24/24完整恢复与12控制后继逐值通过**。不是24个新接触成功，也未把硬停原生尾当正Teacher。材料点累计主指标、正式CPR参考过滤、生产增量replay/存盘与故障注入、正式sampler/伙伴清单仍未闭合。

**正式U/T=0，N未锁，S0按资源边界暂停**：累计21311.255s/21600s（含240s收尾预留），当前输出约59.167GiB/120GiB；总40h未耗尽，但不自行挪用分项。GPU0无本任务遗留。见[完整续报](BFM_CTFT_V1_S0_CONTINUATION_RESULTS.md)、[资源/数值负例](BFM_CTFT_V1_RESOURCE_GUARD_ERRATUM.md)、[实际账本锁](bfm_ctft_s0_continuation_closeout_v1.json)。

以下历史状态保留，历史“共同尺度未完成”已被本次实测更新。

## 2026-11-17 奖励批准状态补正：A1已批准且实现，校准已授权但尚未完成

[Issue67批准记录](https://github.com/GaoE05/robot-research-bridge/issues/67#issuecomment-6940383957)与[A1附件](BFM_CTFT_V1_APPROVED_AMENDMENT_A1.md)明确批准：aux接触项使用逐步成本 `-C`，物理积分另存 `-dt*C`。代码及既有接触probe已执行；本次只读检查有力/无力合计1050条，单位恒等式最大差0。两份最新报告已补写批准/落地/未完成三种状态：[恢复工程](BFM_CTFT_V1_RESTORE_ENGINEERING_RESULTS.md) · [S0接触推进](BFM_CTFT_V1_S0_CONTACT_PROGRESS.md)。

“共同aux尺度待train-only校准”表示数值校准尚未完成，**不是待Yi批准**；该校准已获授权，按原合同继续。工程恢复的g0数据和scale=1仅是占位资格，不是正式共同尺度；正式U/T仍未启动。本次只读核查与文档补正不新增物理、模型更新、科学修订或资格PASS；以下历史状态原样保留。

## 2026-11-17 S0已恢复推进

learner原生eager严格恢复通过后，已完成12祖先存档注释及同父组有力/无力冻结接触链记录，两支含停止与4s释放恢复均无硬边界；不是随机策略、完整Task+learner恢复或正式行为训练通过。新物理6001ticks、正式U/T为0，N未锁。见[实际推进](BFM_CTFT_V1_S0_CONTACT_PROGRESS.md)。

## 2026-11-17 严格learner恢复工程已通过

原生eager执行下，完整模型/五Adam/RNG/replay初始读回和同一下一完整更新全部逐值一致。编译路径残余失败保留；普通工程继续授权已归档。采用已验证eager继续原S0，正式U/T尚未发车、N未锁。接触全链/物理联合恢复、来源分组、共同尺度和吞吐仍待资格。见[工程实测](BFM_CTFT_V1_RESTORE_ENGINEERING_RESULTS.md)。

## 2026-11-17 恢复工程继续

Yi已明确授权自行修复严格learner恢复的实现问题并回归后继续；A2失败与计费保留，不变更科学/安全/逐值恢复门。见[继续授权](BFM_CTFT_V1_RESTORE_ENGINEERING_CONTINUATION.md)。正式U/T尚未启动。

# Current — Issue67 A2已签并执行；完整编译更新通过，严格恢复失败

2026-11-17：[A2签署](BFM_CTFT_V1_A2_SIGNED.md)与[实际结果](BFM_CTFT_V1_A2_RESULTS.md)已归档。只禁用AOT donated-buffer优化，compile=True与原数学/五Adam/全部门槛不改。原eager等价逐值通过，2组Actor冻结＋2组联合更新均通过；完整learner快照约8.14GiB、原子写入约51s。

新进程初始模型/五Adam/梯度/RNG逐值一致，下一expert/train batch、更新末全局RNG和z-buffer一致，判别器仍一致；但实际下一更新后F/critic/Actor出现469个模型张量等差异。严格恢复判FAIL；只能定位至F相关输出边界，尚无首算子或误差量级，不推测为可忽略舍入。A2单次尝试已停止，不改容差、不切eager。GPU0本任务已结束；正式U/T未启动、N未锁，S0接触/材料点/源分组与全链资格未补造PASS。A1支撑正反例和36源状态恢复证据仍保留，旧失败不重判。

## 以下保留此前状态

# Current — Issue67 A1已批准；S0局部资格通过，编译有序反向复发阻断

2026-11-17：[Yi追加批准A1](BFM_CTFT_V1_APPROVED_AMENDMENT_A1.md)已落实原aux逐步单位与新接触项单位接口修订，在线当前5ms支撑与离线0.10s滑移段分离，阈值不变。真实原生前进12次落脚均即时有效、最长无支撑5ms不误停；真实85ms失支撑反例应于55ms停。原材料点滑移全套量尺尚未通过，不填PASS。

[中间实测与失败](BFM_CTFT_V1_S0_PROGRESS_A1.md)：12真实原生祖先的36完整状态恢复及后续12控制/48tick逐值一致；原生完整退出失败保留，不把较早合法状态恢复泛化为全域。原checkpoint strict加载、Actor/F零新增列输出和隐藏信息隔离已通过，B/target-B及原normalizer冻结，aux与五Adam fresh。未继承旧E/I或U/selector/FiLM学习权重。

一次性g0完整更新接线试车遇AOT共享反向tape错误；一次工程修复后eager梯度、Adam和参数逐值等价，但compiled受donated-buffer限制再次失败。按合同§9同根因复发停报，状态BLOCKED_COMPILED_ORDERED_GRADIENT_LIFECYCLE；完整agent.update完成0，正式U/T未启动，N未锁。没有擅自关闭compile、删除GP或放宽恢复/物理门。建议另准一次仅禁用donated-buffer优化的兼容验证，未执行。真实接触采集/固定aux尺度/全链存盘与吞吐/血缘和材料点资格仍PENDING。短暂SSH中断已恢复、GPU0作业均已结束。原单位阻断与所有旧结项保留。

## 以下保留此前状态

# Current — Issue67 BFM-CTFT v1已批准；S0原辅助奖励单位阻断

2026-11-17：Yi明确批准正文＋执行锁、40job小时/120GiB/16M ticks及最新两代＋里程碑checkpoint策略。[批准归档](BFM_CTFT_V1_SIGNED.md)。已建立独立bfm_ctft_v1，无旧E/I或selector/FiLM/U/residual权重继承。原BFM文件SHA与完整训练配置/模型配置一致，F/B/critic/targets存储键齐全；尚未strict加载或更新。

[单位核查与局部修订建议](BFM_CTFT_V1_S0_REWARD_UNIT_BLOCK.md)：原aux critic使用未乘dt的raw成本，环境rew_buf的dt不传入aux路径。执行实际原函数的CPU fixture确认20ms下−.04vs−.0008的50倍口径差。执行锁A4明确要求先报告；没有静默改reward或normalizer。状态BLOCKED_REWARD_UNITS，N未锁，物理/transition/agent update均0；源池/联合更新/恢复/吞吐/U-T均NOT_RUN。需Yi决定局部单位修订后再继续S0，非新学习失败。GPU0未启作业，旧Issue66负结项与全部历史保持。

## 以下保留此前状态

# Current — E/I v2正式负结果结项；停止当前配方

2026-11-17：完整报告、100条件表、445阶段×Goal表、逐关节质量、训练覆盖、优化图和因果在线上下文probe均已归档。[完整结项](CONTACT_GOAL_EI_V2_FINAL_RESULTS.md) · [逐条件](CONTACT_GOAL_EI_V2_FINAL_TABLES.md) · [分阶段](CONTACT_GOAL_EI_V2_FINAL_STAGES.md) · [最终门表](contact_goal_v2_final_gate_index.json)。

E/I各32k真实50Hz数据、127次Actor接受；实际覆盖34训练父组/44规格、64完整回合，FOLLOW→HOLD/HOLD→FOLLOW实际8/7父组。100/100固定终点Dev：E相对Goal-only near改善3.554%、mid−0.096%、stop/continue0.580%，两臂均未满足两族10%门；E/I/Goal-only联合任务质量父组均7/16。机械/保持/恢复保护无新失败；稳定不等于有效跟随。

训练中motion各1135条/46次抽样，终点却未调用motion；selector接受categorical KL最大E5.82e−9/I3.99e−9。同317个实际I决策状态，两最终selector相对初始先验argmax均未改变，raw-logit修正最大约0.00033。高层有效变化小已证实；探索/信用/梯度或reward何者主导尚未证实。

12源祖先分组、3762起点的0.5/1s预测probe：I在线上下文AUROC .9018/.8782，当前deploy/public-Goal基线 .9609/.9093，无独立预测增量。该读数是当前因果F_I预测上下文，不是未来结果B_I编码；独立B_I(state)与8回合正向归因消融维持UNMEASURED，不填PASS。行为已失败，按§7.4第三分支停止，不加训练/不读sealed/不自动开下一阶段。

正式新增stage扣费40652.8081s（含300s未逐项计时小IO保守扣费）、956642个5ms计数、现有输出约7.1554GiB，均在原批准帽内。本次测得CPU收尾1457.7894s，新增物理/策略更新0，三次读取/表生成失败完整计账。项目136.293h为继承阶段重复扣费的保守上界，不冒称去重实际工时。GPU0没有本任务遗留作业。

三条真实qpos 1×视频已在Mac指定videos/ei-v2-dev-20261117目录；完整结项另交本地reports目录。原BC500、BFM/PD、所有旧运行、模型、erratum和正式判断保持。

## 以下保留此前状态

# Current — 100完整Dev已完成；E/I未达到预锁行为增量

2026-11-17：主评价100/100，实际2h11min、352761物理ticks，GPU0释放。相对Goal-only，E近距J改善3.55%、中距约持平、停止/继续改善0.58%；I几乎同E，未达两族10%必要条件。主学习回合none/zero及真实恢复保护无失败，不能用保护PASS替代任务收益。[首轮实际读数与边界](CONTACT_GOAL_EI_V2_DEV_FIRST_RESULTS.md)。只读100存档和16组共同前缀已PASS；终点E/I均只选standing/arm15且执行z完全相同，分阶段结项继续整理；不加训、不读sealed。

三条实际qpos 1×视频已按Yi指定目录交付Mac，SHA及完整解码/全身画面核验通过；[视频布局、来源与实际成本](CONTACT_GOAL_EI_V2_VIDEO_DELIVERY.md)。新增物理/训练0，未改变行为失败判读。

## 以下保留历史状态

# Current — E/I正式训练完成；独立完整Dev已启动

2026-11-17：E/I各32000条、各127次Actor接受；全部固定训练保护通过。独立评价runtime590aa54，formal-dev-02已启动冷加载，固定100完整回合，尚无新Dev结果。此前缺少已验证隔离接触依赖的启动失败在物理前退出，已逐文件同SHA补齐并过8项PINS；失败/成本保留。训练原runtime08b9c96与终点模型不改，不加训、不读sealed、不按Dev反选。[完整事实、入口与成本](CONTACT_GOAL_EI_V2_FORMAL_TRAINING_COMPLETE.md)。

## 以下保留历史状态

# Current — E/I各32k训练及最后保护均完成，统一Dev待执行

2026-11-17：两臂各127次Actor接受，全部固定节点保护PASS；训练runtime08b9c96，终点模型已冻结。Dev尚未执行，不宣称E/I行为收益或BI增量。正在接通独立100完整回合评价入口；原错误与恢复记录、全部科学门/预算/封存边界不变。[完成证据与评价合同](CONTACT_GOAL_EI_V2_FORMAL_TRAINING_COMPLETE.md)。

## 以下保留历史状态

# Current — 两数组逐状态干预通过，E28k严格恢复续跑已启动

2026-11-16：node28首次差异为efc_J_rowadr/efc_J_rownnz；三真实状态×五分支×3s逐数组干预全部逐值PASS（真实力峰0/4.53/10N，均有约束）。修复为完整恢复这两个存档数组，最终全部有效状态仍须逐值一致，不改容差/物理门。新runtime08b9c96，attempt06已启动；当前冷加载/完整读回，不预称28k保护或续采已通过。将核验两次12步，补28k原保护，保留640条pending与28384边界，E完成再I。旧28k停点、107次Actor接受和全部失败成本保留。[完整记录](CONTACT_GOAL_EI_V2_FORMAL_NODE28_RESTORE.md)。

## 以下保留历史状态

# Current — E28k已保存，严格恢复索引差异修复中

2026-11-16：E已到28000条/107次Actor接受，4k/8k/12k/20k保护通过。28k资格源恢复出现efc_J_rowadr/efc_J_rownnz逐值差异，尚未执行该节点物理保护；完整node28与stop28000保留，640条待更新，I未启动。冻结首差异已定位，正在三真实状态逐数组3s扰动资格；不以字段名推断、不改比较容差、不跳过28k门。用户已授权修复后连续推进。[证据与计划](CONTACT_GOAL_EI_V2_FORMAL_NODE28_RESTORE.md)。

## 以下保留历史状态

# Current — 正式E完整恢复通过，已续采

2026-11-16：runtime c724c0f，attempt05已逐值恢复1095条的模型/Adam/RNG/物理/历史；两次各12条真实后续核验逐值通过（24条仅作恢复检查、不计PPO）。已保留block2的71条待更新数据，仍按2048原边界更新；最新观测累计1700条，I随后串行。首块已有2次Actor接受及三项保护通过，不宣称行为收益。

GPU0没有其他CUDA计算作业；此前79MiB图形上下文属于GPU1–6训练的图形初始化，旧“需等空卡”判断已更正。目录统计竞态和恢复检查步数错误均保留失败/成本，不改科学门或预算。详见[恢复记录](CONTACT_GOAL_EI_V2_FORMAL_RESUME_1095.md)。原新鲜初始化仍在attempt03，续跑的initial包不是零学习基线。

## 以下保留历史快照

# Current update — formal E saved after supervision race

2026-11-16：E完成1024条首块、2次Actor接受及none/zero/released三项保护；监督目录计量撞上原子临时文件消失而误停，完整保存1095条（含block2未更新71条）。I未启动。目录计量竞态修复及CPU六项回归已过；新进程完整恢复/续跑入口待完成，不称训练仍在运行。GPU0实时有其他项目进程，不抢占。[证据与恢复要求](CONTACT_GOAL_EI_V2_FORMAL_SIZE_RACE.md)。历史数据、失败、科学门不改。

## 以下保留历史状态

# Current update — 正式attempt03已恢复物理启动

2026-11-16：attempt02在正式采样前因新目录缺少已验证的状态布局辅助库退出；原记录保留。已同哈希补齐并通过布局检查，attempt03开始实际物理前缀。源码仍 `646ba25`，I串行等待E正常完成。正式样本/更新以progress与sidecar为准；不将前缀计入PPO数据。[启动修复记录](CONTACT_GOAL_EI_V2_FORMAL_STARTUP_REPAIR.md)。

# Current — E/I v2正式E臂已启动，I串行排队

2026-11-16。用户确认空间恢复后，实时检查GPU0空闲、共享空间重新高于已批5GiB地板及完整停止包预留。正式E进程已启动，runtime `646ba2531e5ef418a93d7e990bb4bf2850616e6c`；I仅在E正常完成后串行启动。启动快照尚在模型/环境冷加载，不把进程存在称为首块训练完成。实际采样、更新、资格和成本以新阶段sidecar/ledger为准。

原发车因共享空间骤降到0而在preflight退出（正式数据0），以及队列包装语法失败均保留；新attempt02不覆盖旧sidecar。此次源码只增加运行编号，已完成语法检查；学习数学、数据、概率、物理及预算合同不变。磁盘地板仍5GiB，另预留1GiB完整停止包；空间不足会保存停止，不擅自清除证据。

[正式入口](CONTACT_GOAL_EI_V2_FORMAL_ENTRY.md)、[完整预算](CONTACT_GOAL_EI_V2_FORMAL_BUDGET_AMENDMENT.md)。两臂各32k真实50Hz、固定训练保护节点与完整终检，未读sealed。当前没有新的行为成功或BI增量结论。

## 以下保留历史快照

# Current — E/I v2正式预算修订已授权，运行准备中

2026-11-16。Yi已直接授权按实测修订完整正式训练/评价/存盘预算并发车，后续确认继续；[完整预算](CONTACT_GOAL_EI_V2_FORMAL_BUDGET_AMENDMENT.md)与[最新资源签署](CONTACT_GOAL_EI_V2_FORMAL_RESOURCE_SIGNOFF_20261116.md)已归档。原OR1工程资格与历史失败保留。正式仍是每臂P2 12k＋P3 20k，完整100 Dev、48 train sentinel、12 P3入口及最多8消融，不使用sealed。

本轮新增预算56,000 job秒、10GiB输出，combined stage上限74,000秒/18.4GiB；物理新增1.1M ticks、stage1.5M。完整成本区分实测与投影。当前正式训练0，未把预算批准或源码准备写成发车；新入口/评价/资源与项目历史总账核验后才启动。

## 以下保留历史快照

# Current — Issue66 OR1有界资格完成，正式训练未启动

2026-11-16。[最终报告](CONTACT_GOAL_EI_V2_OR1_FINAL_RESULTS.md)：D0/D1/D2及E/I新进程12条＋完整更新逐值恢复通过；新工程采样8192/8192，两臂各4块、Actor各接受12步。I最后恢复379.938秒/4145ticks/新增训练行0，GPU0释放。

输出和I恢复补齐授权已归档；原tick轮询超支及所有失败保留，不追溯改判。OR6按实测热块/IO投影尚未放行；不自动32k/Dev/sealed，不宣称接触行为或BI增量成功。下一项仅提交完整正式预算修订。

## 以下保留旧进度快照

# Current — Issue66 OR1：P3工程IO修复，按已批合同续跑

2026-11-15。D1 E/I完整通过；E的P3首1024行接受4次Actor、累计9次，入口6/6及none/zero通过，但actual-release期间监督器ESTALE中断，未宣称D2通过。用户已授权修复后继续。[工程修复与续跑02](CONTACT_GOAL_EI_V2_OR1_COUNTER_RESUME_02.md)只调整计数IO互斥、异常退出元数据和完整block1续跑。CPU并发/故障注入通过，真实恢复尚待运行。

剩余固定清单：E补release与第二P3块，I入口/两P3块/保护，两臂新进程完整恢复。已有采样5120/8192，剩3072；不重采首块、不改五α/KL/物理/科学门，不自动32k/Dev/Sealed。旧失败和成本保留，分段秒数超支按已签补充报告。

## 前次进度快照（保留当时事实）

# Current — Issue66 OR1：秒数补充已签，继续补齐D1/D2

2026-11-15。Yi最新明确批准[OR1分段秒帽补充条款](CONTACT_GOAL_EI_V2_OR1_SEGMENT_SECONDS_SIGNED_AMENDMENT.md)，原文逐字归档，执行源码340930d。**P1、D0/D1/D2、修复池仅秒数超支改报告制；stage47000秒、240h总账、原输出帽、新采样8192条、五α及全部科学门不变。** OR0按Yi决定接受现状、继续执行，具体资源私有记录。原OR1分账6000秒作为历史参考持续核账，不独立阻断合同内阶段。

已有实证：D0真实五α回溯PASS；新rollout E/I各1个P2块、各2次Actor接受，原logprob误差0，none/zero/实际释放恢复各3/3 PASS。见[此前实际续跑记录](CONTACT_GOAL_EI_V2_OR1_RESUMED_RESULTS.md)。尚不能称主动接触或BI行为收益。

D1现已完整通过：E/I各2048条、各2个P2块，Actor每块接受2/3次、累计各5次；第一接受后的none/zero/实际释放恢复各3/3 PASS。所有已有首块与合格检查复用，新增采样累计4096/8192。E/I断点严格恢复和同状态行为均值身份核查通过，未混用另一策略旧数据。

下一按原合同执行P3入口6项随机资格、各两个P3块、首次接受后3项保护与新进程12条＋完整更新恢复。为守原输出帽，新增无损expected-checkpoint引用与actual-delta格式；真实完整Actor/Adam/物理/RNG签名CPU测试逐值PASS，单bit差异完整保留；不冒称GPU恢复已过。详见[IO资格与成本](CONTACT_GOAL_EI_V2_OR1_LOSSLESS_VALIDATION_IO.md)。

OR1累计3850.106770job秒；D1为2573.457050（原2000、超573.457050），修复/IO343.165106（原100、超243.165106）依已签补充报告，非停工门。超支原因为早期两次IO故障/恢复、原超短作业时限、真实接受Actor回溯成本与必要的无损存盘资格；新投影见IO记录及实际后续sidecar。stage已计11657.810394/47000，采样4096/8192、次数/输出/科学门不变。

旧stage与R2引用不覆盖，项目去重总量缺口不虚报。

E/I完整P2、P3、接受Actor/selector后的新进程12条＋一整更新恢复及OR6仍待验证。正式32k、Dev任务、Sealed、Student/FB长跑均不自动启动；原BFM/PD、BC500、main不变。

## 历史记录（以下为当时状态，以本页顶部为当前事实）

## 2026-11-15：独立审查后局部优化修订待签（未执行）

已按[Issue66独立审查6893703131](https://github.com/GaoE05/robot-research-bridge/issues/66#issuecomment-6893703131)完成存档/合同/源码只读核查。[事实审计](CONTACT_GOAL_EI_V2_OPTIMIZER_AUDIT.md)与[OR1可签修订稿](CONTACT_GOAL_EI_V2_OPTIMIZER_AMENDMENT_OR1_DRAFT.md)、[待签参数](contact_goal_ei_v2_or1_draft_lock.json)已归档。裁决：P1优化规则资格失败，不是正式E/I行为学习失败。原v2已签正文与原工程账本不改。

实际P3 option draws E83/I41；I只有3次FOLLOW且全部arm15。历史clip_fraction全0是pre-step记录，不能证明被拒Adam提议温和；未保存proposal后的完整surrogate/逐关节变化，不补造。当前没有新GPU网格/优化器/采样/物理，α1端点只有CPU修复。

OR1建议固定α[1,.5,.25,.125,.0625]，原KL.02/sigma/reward/物理不变；原Adam一次提议，缩参数配完整新moments事务。先两个归档锚点、可测执行步后才给有限新on-policy资格；依旧不直接启动32k/臂。联合100ms信用/selector耦合仍为机制候选，20ms分层替代只设计、不运行。

待签调整：新工程≤8192条，旧8192原账保留；新工作包总≤6000s（含本轮只读review400），另申请资金5000s；P1段10100s、P6段1400s、stage总47000s（原41000）、输出8.4GiB（原7），其他原帽及科学门保留。原正式训练完整成本待accepted-Actor吞吐/存盘实测，不承诺旧段额度必能容纳。

[新review独立费用凭证](contact_goal_ei_v2_or1_review_receipt.json)：400s整体保守上界，无GPU/物理；原stage7807.703624不覆盖，合并review为8207.703624；R2参考29745.631881/57600，原另批6000未使用。原P1 3837.064558、采样8192、physics201096不变。新科学工作等待Yi签OR1，无队列。


2026-11-15。Yi已批准v2、新增6000秒及自主修复实现问题。已修复并继续到原工程采样硬帽；现在需要科学优化配置审阅，不是等普通实现批准。

[P1权威报告](CONTACT_GOAL_EI_V2_P1_ENGINEERING_RESULTS.md) · [机器读数/身份/成本](contact_goal_ei_v2_p1_engineering_results.json) · [NaN修复](CONTACT_GOAL_EI_V2_P1_GRADIENT_ERRATUM_01.md) · [同提议诊断](contact_goal_ei_v2_step_diagnostic_partial.json) · [已签合同](CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_APPROVED.md)。

已通过：源修复12真实前缀/66对；mean/sample各12/12完整资格；4测试父组/24reward分支、11强Pareto pairs来自3有信息父组、父组等权100%同向。含实际停止、release后4s及同态原生恢复。原失败全部保留，不声称全域reward正确。

真正执行策略未学习：E4块4096、I3块3072；Actor接受0/32、0/24。只有辅助E16/I12与value32/24更新。tail/FiLM/selector权重逐值同初始化、相关Adam全空。Adam提议joint KL约.16–.26超过原.02，全回滚。原提议joint .188249485/Gaussian .188249483/categorical2.78e−9已逐值复现；缩放网格未完成、不改正式步长。

恢复：两臂新进程模型、已有Adam/RNG、有效物理/历史、后续采样与一轮更新逐值同；仅验证确已存在的预测/表示/value状态，没有训练后tail Adam可验证。

G07不通过：失败1024+E4096+I3072=8192帽；I缺第二P3块且仅5完整回合（E7）。不补采、不虚报4块、不正式训练。正式E/I=0、Dev任务=0、sealed未读，无Student或FB长跑；GPU0释放，无队列。

新增stage 7807.703624job秒；P1 3837.064558/4500；修复677.307783/800；201,096/950000 ticks；R2 29345.631881/57600，独立6000未用。下一是审共同可接受优化步及低层/selector尺度，不放宽KL/std/reward/物理。新规则和实际复测规模须另行明确，未执行。BC500、共享原BFM/PD、main及旧结项不动。
