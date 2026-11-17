## 2026-11-17 Yi追加批准S0两小时，资格继续

[S0追加签署](BFM_CTFT_V1_S0_PLUS2H_SIGNED.md)：分项上限现8h，stage40h等总帽及科学门不变。下列预算暂停为批准前中间事实，现已恢复资格工程；U/T尚未启动、N未锁。

## 2026-11-17 Issue67 S0续报

原始BFM独立CTFT资格继续：共同尺度完成，真实接触完整Task/learner短程严格恢复通过；24独立新源状态资格通过。正式U/T尚未启动，生产CPR、增量存盘/故障资格和主指标仍待完成，S0分项预算门暂停；不是学习负结论。详见[实测报告](BFM_CTFT_V1_S0_CONTINUATION_RESULTS.md)。

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

# 当前执行：正式E已启动，I串行排队

2026-11-16：空间恢复后的新attempt02通过资源preflight，runtime `646ba2531e5ef418a93d7e990bb4bf2850616e6c`。启动不等于训练完成；实际首块、保护和终检结果待运行。历史空间阻断与工程失败保留，学习/概率/物理合同不改。当前状态见 [CURRENT_STATE](CURRENT_STATE.md)。

# 当前更新：E/I v2正式运行准备

2026-11-16：Yi已授权按OR1实测完成正式全程预算修订并开始长训；[预算修订](CONTACT_GOAL_EI_V2_FORMAL_BUDGET_AMENDMENT.md)。OR1科学与恢复资格复用，正式数据/初始化与工程scratch分离。当前尚未启动，具体以CURRENT_STATE及新阶段账本为准。旧已结项判读保留。

# Humanoid Compliance — 长期主线与当前执行入口

2026-11-16：OR1局部优化规则已完成有界工程资格及两臂完整恢复；[最终报告](CONTACT_GOAL_EI_V2_OR1_FINAL_RESULTS.md)。真实接触行为与BI增量未验收，OR6正式全流程成本尚未放行，当前停止。

2026-11-15最新：Yi已签[OR1分段秒帽补充](CONTACT_GOAL_EI_V2_OR1_SEGMENT_SECONDS_SIGNED_AMENDMENT.md)。仅分段秒数超支改报告制，stage47000/240h/输出/8192新采样/五α与科学门不变，OR0按决定接受现状继续。已获得E/I各首P2块2次真实Actor接受和各3项保持PASS；正在复用已有完整状态补齐原D1/D2，不再因原分段秒帽等待。原成本停点/IO/SIGKILL不重判，正式32k和Dev/Sealed仍未放行。[CURRENT_STATE](CURRENT_STATE.md)为当前执行状态。

长期目标：Contact-conditioned whole-body behavior selection, modulation and transition；由真实身体、接触历史与公开目标共同决定可持续响应。原生运动先验提供可追溯能力，真实闭环学习分配和转换，之后才验证受限观测Student及自身RL。不能将固定force→skill规则、Goal查表或物理搜索结果冒称已学会通用交互。

## 当前：已获批 Contact-Goal E/I v2

[CURRENT_STATE](CURRENT_STATE.md)为唯一当前执行状态；[v2规范正文](CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_APPROVED.md)、[签署](CONTACT_GOAL_EI_V2_SIGNOFF_20261115.md)、[不可变锁](ei_v2_locks/protocol.json)。Yi整体批准并追加授权自行修正实现问题继续；独立新6,000job秒已有批准，旧R2帽/项目240h不改。

两臂都以原BFM/PD为先验、同等Actor-tail与零初始化contact-FiLM执行适配；E普通有限结果预测，I有限交互后继编码。公开FOLLOW/HOLD目标有明确且不同的实际reward。先Goal-only原生命令和50Hz Gaussian低层PPO，后最低0.5s dwell的有限option/低层联合PPO。两臂无旧I/C0/T30高层actor继承；不是自由256D U探索、标量动作Q考试或完整基座FB重训。自身理想里程计/理想接触测力是仿真假设，不能声称已验证G1感知。

阶段顺序锁定：P0合同/身份→P1完整mean/sample/reward/工程更新恢复及成本→P2各12k低层→通过既定门→P3各20k联合→固定终点统一完整Dev→有限probe/消融→五分支结项。不用任务尚未完美作为额外门；科学/安全/权限/恢复/预算失败按原合同停，不放宽数值。

P0源重复实现已授权修复。P1完整mean/sample各12/12、有限reward及已有状态新进程恢复通过；但原8192工程采样（含失败1024）已满，仅E4/I3块。Actor0/32、0/24，执行参数逐值初始化；更新的是预测/表示/value。原joint KL门正确拒绝.16–.26的Adam提议，一提议.188249已精确复现，不能把辅助loss下降当RL学会。G07因采样/覆盖缺格暂停，正式P2/P3/Dev/Sealed均未运行；[P1报告](CONTACT_GOAL_EI_V2_P1_ENGINEERING_RESULTS.md)。下一先科学审可接受优化步，不能自行放宽KL或加样本；尚不能判断E/I或B_I行为价值。

## 已完成事实与边界

H0 P-E4/4、P-I3/4有限Goal局部物理效应保持有效；零学习Goal查表能复用E四组结果，窗口未跨真正stop，不能升级为接触驱动walking或BI独立行为收益。旧QA2/G17/Return/Envelope的失败保持，但不用其独立Q门阻塞不含该Q的新算法。BC500/BFM/PD与全部历史模型、失败、erratum保留。

[本次P1之前的完整overview存档](RESEARCH_OVERVIEW_BEFORE_EI_V2_P1.md) · [H0之前历史](RESEARCH_OVERVIEW_BEFORE_H0.md)。历史“草案未签/源阻断/需下一批准”文字只说明当时状态，不覆盖上述最新签署和实测，不再次启动旧路线。
