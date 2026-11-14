# CURRENT_STATE — R2 E/I v1.0已获Yi明确批准，入口资格准备中

2026-11-14。用户直接签署[R2-EI训练合同](R2_EI_TRAIN_CONTRACT_V1.md)，正文旧“DRAFT/待批准”已由明确授权覆盖。两臂必须完整新随机U、零输出，不能沿用零U诊断中A128隐藏层。E/I各32k真实50Hz、总R2≤57600秒/12GiB，继承3088.839565秒，不重置旧账。先实际采样/保持参考、完整恢复与存盘投影，再Q warm-up资格，之后才Actor/U更新。当前训练0，sealed不执行；原A128/BC500/负结果保持。

---

# CURRENT_STATE — 原生零U冻结对照完成，未启动训练

2026-11-14。[零U配对结果](R2_ZERO_U_RESULTS.md)：同六个暴露回合，去除A128 Guidance后严重旋转/高滑移消退；原生standing仍未完成中距引导，稳定起点不等于接触能力。零U干预及完整源/首观测逐值匹配，物理/reward/门不变。R2累计3088.8396秒/3600工程秒；正式训练0、sealed0。保留D1、A128/V1/Return64及所有历史失败。下一步仅建议从零U学习接触推进，需锁定正式科学合同；本轮完成即停。

---

# CURRENT_STATE — D1完成，R2工程通过但初始化资格失败

2026-11-14。[D1结果](R2_D1_RESULTS.md)；[R2实测和阻断](R2_ENGINEERING_AND_INITIALIZATION_RESULTS.md)。E/I真实更新及新进程恢复通过；正式训练0。A128 Guidance初始化在无接触/近/中等完整回合严重旋转、牵拉，按原设计停止，不换零U/B19隐瞒失败。六回合与工程失败完整保存，sealed未执行。R2保守2308.780秒、D1保守1529.680秒，分别记账。GPU0无本阶段后续作业。用户授权普通修复自主解决；改变科学初始化须显式新决策。

---

# CURRENT_STATE — Issue63 D1完成；R2工程资格正在执行

2026-11-14。Yi明确批准启动D1/R2；[授权登记](R2_EXECUTION_AUTHORIZATION.md)。旧B19/EnvelopeV1/Return64与BC500身份及负判定保持。

[D1八分支结果](R2_D1_RESULTS.md)：两个原V1分支physics/control/action逐值复现；Control64/Return64只执行首20ms后统一V1，全部3秒可恢复，slip峰值最高.08287m/s，没有旧完整新策略的~.8m/s/−150°崩坏。支持连续反馈累积的重要性，不证明仅第一步问题或旧Q的无限时域唯一bug。Q与共同V1有限窗pair排名一致7/12，四项Control/Return相对V1均Q更高、3sreturn略差。无sealed、无训练、全部失败保留。

D1物理runtime290739f；共14903/18000physics ticks，含准备/分析发布保守1529.680/5400job秒，含工作树~.26GiB/1GiB。D1已结束。

R2新E/I部分Actor执行器已实现，初始同A128动作/z逐值相同、隐藏伙伴输入不影响actor；原运动Actor末块+输出层4,263,965可训练参数，独立副本，原BFM/PD/BR不变。E工程实际采样250条18.327s、3个更新组.0842s、完整存盘2.824s；新进程读取后继续5条采样+一次完整更新，physics/replay/RNG/模型/Adam逐值一致。峰值显存~.852GiB。这是短工程测试，不是32k正式学习结果或长期吞吐承诺。

I工程资格进行中；所有新作业GPU0串行，工程≤1h含在R2≤16h/12GiB内。正式训练尚未启动：仍需完整初态/恢复物理资格、训练域保持/辅助项校准并锁定、IO投影。不得把工程系数当科学最优，不继承工程更新模型开训。新train/dev/sealed父组recipe分组已生成，sealed未执行/未用于调整。下一步继续已授权R2资格与配对pilot；触科学/正确性/资源边界则停报。

---

# CURRENT_STATE — Issue63 D0只读审计完成，R2设计待审阅

2026-11-14。最新用户授权仅D0档案机制审计＋R2方法设计；本轮**新物理0、优化0、GPU使用0、sealed读取0**。旧B19/Envelope V1/Return64负结果保持，B/C正式训练仍停止。

[R2_CAUSAL_TRIAGE](R2_CAUSAL_TRIAGE.md)已核：实际reward能明显惩罚坏恢复，但inactive heading不计、slip成本封顶；原固定Q在3058/4750真实回放上为正（真实非正reward的价值上界0），故绝对值校准不足，不据此宣布同状态相对动作排名或唯一critic bug。两恢复起点的新两模型27/29执行action相同，raw→exec局部导数27/29为0；相对稳定V1首q_PD最大已变约.086–.108rad，真实状态首5ms分叉，随后支撑/滑移恶化。原生CPR和两个稀薄Return参考域没有保住另外恢复域。

已交付[逐关节/数据/Q证据](r2_reward_q_execution_audit.json)、[D1最小首20ms动作＋共同V1后续合同草案](R2_D1_MINIMAL_PROTOCOL.md)、[下一代R2训练设计](R2_INTERACTION_FB_DESIGN.md)。D1建议仅2既有恢复状态/4动作、≤1.5job小时；R2建议真实rollout部分Actor适配强基线vs新增BI/FI/Goal Composition，均是待批准设计，**未执行、未解冻、未新建训练臂**。A128与BC500保留参考，旧B19不冒充新公平臂。

事实runtime f6bd9eb606975202b34e693968784308bad3745c；分析仅CPU前向/自动微分/档案复算，不调用环境/optimizer。全轮读取/分析/文档/IO保守按≤3600job秒记入项目总账，数值作业receipt另列；公开输出约0.5MiB。不是新物理/训练预算授权，也不重置旧账。本轮完成即停，后续由Yi确认。

---

# CURRENT_STATE — Return小试验负结果结项，全部停止

2026-11-14。[结果与12条物理对照](ADAPTER_RETURN_PILOT_RESULTS.md) · [执行勘误](ADAPTER_RETURN_PILOT_ERRATUM.md)。先恢复域资格后，完成同B19起点λ0/.0008各64次adapter-only更新（另2个工程步），U/Q/discriminator/targets及原BFM/PD冻结；原Envelope/reward/噪声/物理门未改。

Return训练代理末批.1466，对照13.777（−98.94%），但两个独立暴露恢复父组都明显退化。均值：V1滑移峰.0336/.0181、yaw−15.14/−15.37°、3s root位移6.42/4.82cm；Return64变为.7843/.8348、−145.73/−148.46°、42.31/39.52cm。采样分支同向。12分支均真实q越界0/失支撑0，释放力0不作为任务收益。首观测/z/BFM动作/q/qdot逐值一致；不是换初态比较。

按“独立恢复未保住即结束本配方”停止队列，已运行第二父组完整保留。active/none/已知B19末轮物理缺格，未宣称通过，也不加训找补。新off相对门在独立dev缺匹配参考，记未测；V1不是off。结论仅为该固定offline adapter配方失败，不否定所有Return/接触RL/交互F/B。原V1负判定不改，B/C不恢复，sealed未读。

训练runtime1c06170，物理8afe076；Return64 f2056e30087e…、control64 c4e1c1525b37…，全SHA见报告。累计本次保守3504.11/7200job秒，新增约0.422/2GiB，真实physics27584tick（含初始化/重建/失败），14个新3s窗，RL训练0、Teacher查询0。GPU0已释放；无后续作业/自动新阶段。

08:11定时触发却因同会话writer冲突失败，09:06人工接续；IPC与efc_JT全量恢复缺口已分别修复并留勘误，未屏蔽数组或放宽比较。旧模型、全部失败和旧Pilot账不覆盖。

## Previous snapshots (historical, superseded status only)

# CURRENT_STATE — Return小试验已获批；GPU0占用，未开物理/训练

2026-11-13。Yi明确批准[Return小试验草案](ADAPTER_RETURN_TO_PRIOR_PILOT_DRAFT.md)，[批准协议与准备状态](ADAPTER_RETURN_TO_PRIOR_PILOT_PROTOCOL.md)已另存，未改写原草案或V1失败判定。两B19克隆各64次adapter-only，λ0/.0008，U/Q/discriminator/targets共同冻结；7200job秒/2GiB，先恢复参考域资格再学习，不是原B/C续训。

CPU核查：B19哈希及8个存档状态来源可用，原4750/760条经父组隔离后训练replay1255/native205条；六个开发父组排除，native按10个真实前缀块归属。[清单证据](adapter_return_pilot/preparation-summary.json)。这只证明存档可用，尚未证明物理重建/恢复有效。预记准备成本保守600s（含已测8.45s CPU分析），余≤6600s，旧Pilot账不重置。

现场GPU0正被SoftSONIC队列使用（smoke后已进入formal），遵守单GPU重任务串行，未操作其进程、未安装抢卡/watch。新物理/更新/Teacher查询=0；两组Return模型尚不存在。需卡释放后重新核资源，沿现有授权完成恢复资格与学习，不需重复审批。正式B/C仍停，sealed未读，原负判定保持。

## Previous snapshot (historical, not current authorization)

# CURRENT_STATE — Issue62只读机制审计完成，仍停止

[一页结论](ADAPTER_RETURN_AUDIT_SUMMARY.md) · [完整依据/逐关节表](ADAPTER_RETURN_TO_PRIOR_AUDIT.md) · [待审阅小试验](ADAPTER_RETURN_TO_PRIOR_PILOT_DRAFT.md)。本轮新物理/训练/优化步均0，只读CPU分析；未改原实验runtime、科学阈值与负判定。原V1失败仍以9441ec3为准，B/C未重训。

确认CPR是在原生前缀上比较当前策略z与存档native z对应的总动作，不是访问恢复域的同z adapter归零约束。B19恢复replay37秒/1850条，实际抽414次；特定失败父组恢复只抽1次；nativeCPR760条无访问恢复。旧reward在匹配3秒窗更喜欢off（−.276 vs V1−1.011）；局部Q偏好相反，支持但不证明外推/信用问题。执行后Return在一个已存状态79.3%输出分量被原Envelope饱和；提出raw Return代理作待审阅方案，未改clip。

建议仅有条件做B-only隔离试验，不重启原完整B/C：独立恢复域先验证，若有效才做同数据/64步adapter对照；新协议预算建议2h/2GiB，尚未获准执行。A/B/C原权重、BC500、旧失败、封存、旧10.279462h Pilot账本均保留。当前任务完成即停。

## Previous state snapshot (unchanged historical verdict)

# CURRENT_STATE — Issue62 Envelope frozen qualification failed; stopped

[Results](ADAPTER_ENVELOPE_V1_RESULTS.md) · [Harness erratum](ADAPTER_ENVELOPE_V1_ERRATUM.md). Yi's inward numerical amendment is implemented: actual representableaction/g(action) shared across physics/replay, max measured correction≤4.77e-7rad. Old strict inverse failure preserved.

FrozenB19 same11.8s sample branch completed3s: actual joint excess0 and no-support0; however loadedslip peak **.919526** andp95 **.150268m/s**, above matchedoff limits **.727233/.041672**. Physical qualification therefore fails under locked §4/§9. No tuning or furtherbranches/B/C/dev/probe. A128/B19/BC500/oldresults unchanged; sealedunread.

Runtime `e4c2d3ff08fc0a29152dc7b1f0ee00201b077012`; physical trace SHA `2fb2feacb607e4c73227daeb69d4a5790174a132beb9703d070d071819e4a95b`. Gradient audit separately failed due outer no_grad after physicaltrace saved; no newoptimizerstep. Minimal harness fix is syntax-checked only, not fully qualified. Mean/active/no-contact gates remain missing, not inferred from sample.

Pilot cumulativechargedupper **10.279462/18h**, output~3.4340/16GiB; GPU0 released. Newformaltraining0; actualphysics7641ticks includesreconstruction. Complete forecast and frozenstate manifest remain at e4c2d3f; current stop ledger supersedes queuepending status. Any future redesigned recipe needs an explicit scientific decision; do not auto-restartB/C or changeEnvelope constants.

## Previous snapshot (historical)

# CURRENT_STATE — Issue62 Envelope inward numerical revision, qualification pending

Yi已批准确定性向原许可区间内部舍入；实际action/g(action)作为物理、actor/target、critic和replay共同真值。仅≤1e-6rad纯表示修正可自主处理，科学常数/物理门/reward不在授权内。c13cc69严格逆映射失败保留。

CPU/CUDA离线算子已通过，最大数值修正4.76837e-7rad；完整物理资格尚未运行，不称新训练成功。已锁四个暴露父组、20新分支，B19原on/off存档复用须核对身份。A128/B19/全部失败不变。

完整投影17.9636累计job小时、12.7414GiB（含历史/全部B+C128轮/恢复/五方100回合dev/probe/失败余量）；按6827917070条件预授权采用18h/16GiB，未重置历史。新物理前机器锁和两份投影提交。当前已计34028.589s＋本次数值实现/分析/准备上界1800s；九stage inode去重3.4223GiB，修正旧较窄统计范围而非删除资产。资格失败立即停，不自行换常数。

[协议](ADAPTER_ENVELOPE_V1_PROTOCOL.md) · [时间投影](ADAPTER_ENVELOPE_V1_TIME_PROJECTION.md) · [存储投影](ADAPTER_ENVELOPE_V1_STORAGE_PROJECTION.md)。重GPU单GPU0串行，sharedfree≥100GiB，sealed未读。

## 历史快照（不追改）

# CURRENT_STATE — Issue62 Envelope逆映射前置核查阻断

2026-11-12。已承接最新授权comment6827917070；[数值预检报告](ADAPTER_ENVELOPE_V1_PREFLIGHT_RESULTS.md)证实原FP32 action→q_PD映射无法逐值表示批准Envelope的一个投影目标：NVIDIA反例由两相邻action输出严格夹住，最小差2.98e-8rad。这不是物理质量失败；但命中§3/§9明确的逆映射正确性暂停条款，未自行放宽或近似通过。

新物理/训练均0，A128、B19与全部历史保留；B/C/dev/probe未启动。协议/机械表以BLOCKED版本记录，不冒充完整资格/预算锁。建议Yi确认理想目标与实际可表示目标分开、确定性朝许可区间内部舍入并以实际g(a_exec)作全路径逐值真值；尚未实施。

Pilot累计9.452386/15h、输出3.052665/8GiB，未用条件扩额；GPU0释放。执行脚本SHA `1ab0bad8338ca61c813cfefc5554fa37b253af97a3174df9946c659b64275cd4`，基准e60fff4。科学/物理/概率阈值、封存和共享资产不变。

## 之前快照（保留历史）

# CURRENT_STATE — Issue62 adapter-off对照完成，等待配方决策

2026-11-12。[冻结反事实报告](INTERACTION_ADAPTER_OFF_RESULTS.md)：同一B19完整11.8s状态，关闭整个执行adapter但保留学习U/Guidance与原BFM/PD，采样和均值均完成3秒、实际关节越界0；原on采样/均值约0.51秒左踝pitch撞硬界。共同前缀/首观测/z/RNG严格一致。支持adapter在该释放状态中有实质因果贡献，不是全任务因果唯一解释。

off滑移峰仍0.606/0.524m/s，未宣布质量全过或新配方成功。新增训练0，A128/B19和失败保留；B/C/dev继续停止，按Yi“结果出来再定配方”交付，不自动续训。

runtime `33d5277c9c232bd817302aee1cf5d45823510754`；pilot累计9.0827/15h，输出3.053/8GiB。GPU0已释放。下一配方建议聚焦execution adapter实际动作与无接触恢复域约束，不再默认将该越界归因探索噪声；具体方案尚未获批。

## 之前快照（保留历史）

# CURRENT_STATE — Issue62 B均值真实越界，命中用户停止边界

2026-11-12。[B定性报告](INTERACTION_PILOT_B_FAILURE_RESULTS.md)：从B19同一完整状态/RNG恢复，sample原动作/控制/5ms物理逐值复现；去噪声的真实mean分支也在12.310s左踝pitch越界0.02756rad（原sample12.315s、0.02138rad）。发生在25.984cm mid任务release10.8s之后，实际力0。原sample最后状态：BFM q_PD −0.8106rad、adapter后mean −1.1598rad、采样−1.1623rad，下限−0.87267rad。不是仅噪声，也不是恢复错配；不据此把adapter唯一因果或全回合mean失败写成事实。

Yi明确规定若均值动作本身越界、修法需改训练配方则停止。本次符合，故未改共同std/reward/硬界/划分，未跳父组或重抽RNG，未续训B/C/dev。A128=32k/256更新保留；B19=4750/38及26条未提交失败保留；C和dev缺格，尚无新三臂行为比较。

诊断runtime `9bceb9956baa0d29d9c6d6d222b9e3484ef3707c`；所有旧模型与失败完整。新增正式训练0；成本累计8.6295/15h、输出约3.021/8GiB。GPU0释放。下一建议只做adapter-off同状态物理因果对照，再决定是否修订adapter约束/保持域；尚未执行，不自动追加训练。

## 之前快照（历史保留）

# CURRENT_STATE — Issue62 新共同噪声合同已获接纳，A重跑进行中

2026-11-12，Yi明确接纳std=.01/clip=.06为A/B/C共同合同：[批准记录](https://github.com/GaoE05/robot-research-bridge/issues/62#issuecomment-6822542264)。保留单窗峰值筛查未通过，不改判历史；该筛查非§5.1成功判据，残余尾部由原dev指标报告。原真实物理/正确性硬界不变。

独立stage interaction-pilot-v3-20261112；实际runtime `62d55b0d26127440253dba2b5d0baa8bd1e457fa`。A从原共同模板/原生standing初始化、fresh optimizer/replay、零训练计数重跑；已逐值验证common网络tensor和初始化action与旧A一致。不是旧A65续训，不是BC500初始化。实际action一次Gaussian采样std=.01/clip=.06及RNG合同已核对；探索配置随checkpoint保存并在resume核对。

启动核查已提交round3＝750真实50Hz transitions、6完整update，所有已完成轮完整存盘；监督PID2374035、A PID2374051。这是运行快照，实时以jobs/rounds/ledger为准。单GPU0队列A→B→C各32000，然后独立dev三臂/共享initial/BC500与B_I首读；sealed不执行。当前无新行为收益或三臂胜负结论，单seed有限规模不自动满足§5.1正式证据要求。

[共同实施合同](INTERACTION_PILOT_V3_ACCEPTANCE.md)及[机器锁](interaction_pilot_lock_v3_approved.json)：Pilot累计15h/8GiB，继承22156.050970s，另保守预记1200s准备/发布；新子job和失败/IO/评价另累计，项目240h不重置。历史全部输出纳入8GiB计数，shared free≥100GiB。启动现场GPU0空闲、共享free约427GiB、CPU210核/RAM可用616GiB；这些不替代以后实时检查。旧A65/失败/噪声对照及共享资产保持。

## 之前状态快照（原记录保留）

# CURRENT_STATE — Issue62 全维降噪冻结验证未通过，训练未重启

2026-11-12最新授权：共同std=.01/clip=.06先验证，通过后A从原共同初始化重跑、B/C同合同；Pilot累计上限15h，仍计入项目240h，输出8GiB不变。

[新结果](INTERACTION_PILOT_NOISE_RESULTS.md)：同一A65完整状态/RNG、冻结权重3s。匹配窗口误差14.42→11.17cm、力8.49→8.11N，关节越界消失；加载滑移p90 .0855→.0349、超.45 tick5→2，但峰值 .5822→.6197m/s（右脚接触点146N）。因此未通过运行前登记的峰值不恶化规则；未事后改门，未启动A重跑/B/C。该局部负结果不等于新噪声无法学习。

实际runtime ecf8856，新增训练0；GPU0已释放。累计保守成本6.15446/15h、聚合输出约1.41/8GiB。原A65和全部失败保留，旧A不充当新合同匹配对照。下一决定涉及是否接受这一Pareto修订，不能默默放宽本次验收规则；本轮未扫其他噪声、未读sealed、未改BFM/PD/reward。

## 之前快照（历史保留）

# CURRENT_STATE — Issue62 工程修复完成，采样安全修法尚未通过

2026-11-12最新纠正：用户要求解决后继续；监督器改名竞争/异常回收、Pilot清单恢复及两个存档MuJoCo数组写回已修复。五项CPU测试通过，round65新进程两次恢复的全部后续命令/action/control与500个physics ticks逐值复现原失败，未放宽断言。见[修复与反事实结果](INTERACTION_PILOT_REPAIR_RESULTS.md)。

冻结反事实：原采样两次均在7.80s右踝越界0.023757rad；mean完成3s无越界，但质量仍不足。另测仅两踝noise×0.2：无越界，但同窗口加载滑移峰值0.582→0.852m/s，未采用。原Agent默认std/reward/PD/边界均未改，不能把这个诊断冒充已修好行为。

A仍为round65=16,250真实训练transitions/130updates；旧未提交125条保留。B/C/dev未启动，sealed未读。本次六个冻结诊断进程（含两次恢复失败）新增优化更新0；GPU0已释放。Pilot累计保守job上界21,046.1843s=5.8462h/8h，输出约1.392GiB；保留原无法精确确定退出时刻的账本上界，不以活动下界替换。

下一建议只有全动作探索尺度的有限冻结对照（全29维std.01/clip.06；未执行）。正式采样修改须明确新三臂共同合同，不能仅改A、静默跳过失败父组、放宽真实物理边界或将旧/新噪声产物当严格匹配。余量不保证完成剩余三臂与评价，不自动扩8h或重新计账。原BFM/BC500/所有历史保持。

## 之前状态快照（保留历史，以上方纠正为当前状态）

# CURRENT_STATE — Issue62 Pilot 已停止：A物理硬界＋监督器故障

2026-11-12。T0三门通过结论保持，用户已接受；每臂3,000提案被否决，原草案另存。此次仅完成获准工程准备，没有启动Pilot、Student/PPO或新机制。

[修订锁](PILOT_BUDGET_AND_SPLIT_LOCK.md)建议单seed A/B/C各32,000真实50Hz transitions，共96,000；8h/8GiB总框架，准备成本也扣除。实测存盘/吞吐保守投影最大约33,500/臂，建议32k留余量。34训练父组/44回合第一遍约21,650步；dev16父组、sealed28父组整组锁定，stop/continue不跨split。新清单尚未物理执行，哈希去重不冒充统计独立。

[准备结果](INTERACTION_PILOT_PREPARATION_RESULTS.md)：A/B完整update+IO warm吞吐9.565/9.549 transitions/s；C7.869。common网络初始权重/实际action相同；C增加表示模块容量，不能唯一归因Factorization。delta-v1完整checkpoint新进程恢复两轮逐值一致；中断保留最近提交轮。独立完整评价C/BC500及共同前缀检查通过；参考入口错误/汇总错误和全部失败保留。最终runtime `6ee2dd283ca532862ca190c3df7b543895470680`，实际运行SHA逐job见账本。

成本2560.4251s（含300s保守CPU记账），新资产约0.316GiB；项目累计48.8623h/240h。原BFM/BC500/所有历史不改。首读仅dev方向性+B_I探针；§5.1正式判据尚不可满足，不读sealed、不作显著性/收敛/晋级声明。

**用户已确认32k/臂修订锁。** 原草案JSON保持不变；批准版本为 `interaction_pilot_lock_v2_approved.json`，引用Issue62 comment6816360266。独立stage `interaction-fb-pilot-20261112`，计划A→B→C串行训练，再独立开发评价及CPU probe；当前尚无Pilot行为结果。累计预算包含准备2560.4251s，另保守预记当前启动/报告CPU600s；不重置8h/8GiB。运行PID和每轮账本以stage/jobs及ledger为准。以后启动仍须实时核GPU0与CPU/IO、共享free≥100GiB；三臂单卡串行，完成轮必须完整存盘。

[Issue62](https://github.com/GaoE05/robot-research-bridge/issues/62)、[Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)、[T0报告](R1_T0_TRIAL_RESULTS.md)、[方法](INTERACTION_FB_METHOD_SPEC.md)、[历史入口](CURRENT_STATE_BEFORE_ISSUE62.md)。不得将代码推送当新行为成功，不自动开启后续阶段。

运行快照：2026-11-12，A臂pilot-A-r1已完成至少round1（250真实50Hz transitions、2完整更新组及事务checkpoint）。runtime `cf247d5f87beab89ff8be1a6eeaca764f0f93599`；queue PID2349347、A PID2349359。B/C及统一开发评价依次排队，尚无行为结论。首尝试缺少stage-local状态布局.so，采样前失败105.2319s，精确复制既有helper后重启，见[启动记录](INTERACTION_PILOT_LAUNCH_ERRATUM.md)。实时进度必须读取独立stage/job sidecars，不能把本快照当最终轮次。

最新实时核对（2026-11-12，覆盖上方运行快照）：A-r1最后完整round65=16,250 transitions/130更新；下一轮125条采样后右踝roll真实q=0.285557rad，XML上限0.2618rad，越界0.023757rad>原0.02保护，停止，未新增更新。round65和未提交失败包均保留。监督器更早在扫描.pending-18目录时与原子重命名发生竞态，异常退出，未终止/回收子进程；子进程之后自行触发物理保护停止。B/C/dev均未启动，GPU0空闲。旧ledger stopped/FileNotFound及缺失wall不能当完整成本；独立status-reconciliation保留日志活动下界与首次确认进程已死的墙钟上界，退出时刻无法精确恢复。此为需独立记录的监督缺陷，不能仅描述为日志延迟。不得自动重启、放宽硬界或宣称Pilot方法胜负。
