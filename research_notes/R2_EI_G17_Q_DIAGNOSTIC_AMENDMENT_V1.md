# R2 v1.1 Addendum G17-QA1｜Q更新量 × 行曝光的匹配诊断

**状态：REVIEW ONLY，待Yi新签，不自动开启GPU/训练/物理。** 本文是已签v1.1 G17失败后的**局部更新日程修订提案**，不是重写原合同；旧3500条/56组G17 **FAIL** 不追改，70%／至少四独立父组门、原reward、任务、PD、force cap、噪声、Actor/U/Goal结构、train/dev/sealed分组、16小时/12GiB及项目240小时总账不变。原v1.1 D11只读诊断许可不包含新增Q optimizer更新。

## 0. 真实证据、拟检验的问题

见[原正式结果](https://github.com/luoxinyuan/compliance_BFM/blob/de434bc/research_notes/R2_EI_Q_RANKING_RESULTS.md)和[原锁定协议](https://github.com/luoxinyuan/compliance_BFM/blob/d93d69b/research_notes/R2_EI_Q_RANKING_PROTOCOL.md)。E/I各3500真实50Hz主轨、各56 Q组(batch64，两Q各一个Adam更新)、I另26 B_I/F_I组；Actor/U均0。八实际train状态／五父组／40个首20ms真实分叉＋共同mean至1秒，四父组58个有效动作对，两臂26/58=44.83%（父组等权52.14%），mid进入3/10、mid保持0/9；原70%门真实失败非UNDERPOWERED。执行输入/状态/奖励重积分/Q重算没有已证实错位，TDloss约1.5e-4不能保证动作排序。

56×64=3584次Q抽样，实际unique1838/3500=52.51%，none456/480、continue125/565；训练时buffer逐渐增长，完成六族真实覆盖不等于六族曾均衡进入loss。现在**固定3500数据不再增采样**，先问优化/target传播不足与行曝光不足各贡献多大。直接56→512的均匀抽样会同时提高更新次数、unique覆盖、重复曝光、target EMA跟随，是**联合假设**而非干净单变量；但节点仍值得保留。

## 1. E-only两分支共享完整Q56起点

只读取既有E round14完整checkpoint和3500条immutable实际训练replay；锁它们的manifest/core/agent/Adam/target/replay SHA。两分支从该**同一个Q56参数及优化器target**复制，原Actor尾层、U/Goal及其targets、原BFM/normalizer/PD全部冻结。I完全暂停；**仅E的Q1/Q2、Q-target、Q-Adam继续更新**，不进入物理执行或Actor/U。所用action一律现存实际a_exec，reward、discount、duration和nextstate来自原replay；batch64、lr和target EMA=.01均不变。

- **U＝uniform fixed replay**：在最终3500行上均匀有放回抽样，锁定独立随机流。
- **C＝coverage-first matched-quota replay**：每组64个抽样槽按U预生成日程的**实际独立父组ID×物理phase**多重集一一匹配；在每个槽位选择本组所属层次中**累计历史曝光次数最少、优先从未见过**的实际行（并列由预锁随机流解决），直至该类耗尽再循环。两支每组更新数、每组父组/phase配额、loss/target/optimizer参数相同，只改变同类内部哪些行重复出现；真实唯一行及按父组/phase曝光逐组登记。训练时元数据用于replay抽样不进入Actor/Q特权字段。缺真实phase归属或父组映射则在新Q更新前BLOCKED，不能悄悄改成另一套抽样分布。

**共同锁定总组数节点：56、128、256、512（512硬封顶）。** 从Q56到128新增72组；到256新增200组；到512新增456组，每分支各如此，绝不把两个分支合计1024错写成单臂更新。128节点重点看同预算U/C的覆盖差异；256→512时观察unique是否已近饱和及价值/target传播是否继续改善。中间节点仅诊断，主报告以预先固定512为最终终点；不因128/256分数好就后验改选checkpoint。任何非有限loss、权限/保存/源状态SHA错误或预算护栏触发立即停，不扫学习率/EMA/层数/seed。

## 2. 目标条件与不合理价值的只读前置审计

在第一步Q更新前只读旧40条真实物理分支与E56数据：从实际完整时序核对Q的首动作输入、100ms high-level previous/target z、E的冻结target策略与物理共同续跑mean在20ms/100ms处是否一致；逐20ms列reward、discount、target-Q的backups，特别记录1秒 n-step Bellman残差。

**mid保持**真实1秒mean回报-6.380468而Q≈-0.957223：在同一个确定性未来、相同后续策略、未来奖励均非正的条件下，完整价值应不高于已经付出的1秒成本；若Q的输入不足以描述未来伙伴spec或目标网络的后续策略不同，则是状态充分性/条件价值语义问题，不能只拿“无限时域≠1秒”搪塞，也不能武断宣布代码bug。

从现有stop/continue**完全相同的过去前缀**检查：Q可见输入相同但未公开未来partner要求是否可导致相反的优选动作；若旧档案没有同态不同未来的动作对则标UNMEASURED，不发明证明。识别不充分的状态应被报告为G17不可用风险，不能靠512组试图让一个确定性Q凭空辨认隐藏未来。

Q56→512时另报current Q/target Q及其EMA残留、TD与n-step误差按near/mid/stop/continue分层，确认改进的是时间传播、拟合还是纯头部数字量值。

## 3. 旧40条的评估不能冒充再过G17

所有56/128/256/512在**同样已曝露、实际reward已计算的40条分支**上只做critic前向，不重复物理，也不把这些分支作为训练标签；58非平凡动作对/四父组/原1e-4差异门与70%准则一字不改，另报四组等权与mid保持0/9的变化及Q数值/目标校准。**对旧40条排序改善仅可写retrospective fit；不构成新G17真资格，也不允许启动Actor/U或I。** 若512<70%或mid/stop仍明显高成本却Q严重高估，终止本诊断，不补节点、不改loss和标签。

**独立前瞻确认仅在512已达到上述诊断条件时触发**：在新Q训练之前锁定四个不属于这3500条replay父组的train-only真实源状态与时刻、类别覆盖、五个合法首动作的生成seed及1秒共同固定mean物理续跑；原Q更新后最多为这四个父组生成20条真实分支，新的物理结果绝不回填Q训练replay。按仍为≥四独立父组、70%资格判读，不用Dev/Sealed；资源不足标PROSPECTIVE_UNRUN。若前瞻失败，则E Q仍不晋级；即使通过，也只授权交回**进一步E/I正式训练日程与I重资格决策**，不得自动Actor训练或拿E Q权重替换I。

## 4. E-only不是“I的BI无关”

当前execution_agent.py：E的GoalEncoder在Q-only阶段固定；I的B_I/F_I在Q-only阶段每组继续训练，而其z_I=normalize(B_I(public prompt))是Q输入与target-Q下一动作的一部分。即使E/I曾执行逐值相同的行为、Q排序同为26/58，I的prompt/target分布可能非平稳。**E-only优先是隔离公共物理critic学法问题、节约计算；不意味着I已自动通过或I表示绝对不影响本诊断。** 将来需为I另锁Q更新/representation drift控制与G17资格，未授权时保持冻结。

## 5. 三列表与STOP纪律（沿用v1.1门语义规范）

| 防什么 | 怎么算 | 已有PASS／FAIL案例与缺格 |
|---|---|---|
| 避免重判旧G17 | 锁3500/56组/40分支和原70%；另开G17-QA1结果栏 | **真实FAIL**：26/58=44.83%；新Q优于此的真实PASS尚未出现 |
| 防update与coverage含糊解释 | 同一Q56、同组数和每组父组×phase配额的U/C比较；记录unique，512封顶 | **PASS实例**：旧E/I3500真实主轨/动作对齐；**新sampler的PASS/FAIL尚未发生**，运行前须先用immutable buffer做完整schedule fixture和拒绝测试 |
| 防奖励/价值时域误用 | 同首动作与固定后续比较Q、G1s、Qtarget、n-step残差及未见未来spec条件 | **真实疑点**：mid保持Q≈-.957 vs G1s≈-6.380；程序/reward核对曾PASS，但还未证明新配方的价值校准 |
| 防在曝露40分支选择checkpoint | 128/256只报诊断，512固定终点；旧40 retrospective，前瞻另四新train父组≥70才有资格 | 旧40真实FAIL；未来512和新前瞻PASS/FAIL **PENDING** |
| 防成本/资源及自动复训 | 新增≤3600job秒、≤1GiB、≤18000全流程5ms ticks；原R2总16h/12GiB及项目240h不变 | 当前真实账15241.192/57600s、2.372/12GiB在帽内；本Addendum新物理训练尚未发生 |

## 6. 预算、预写失败分支、交付

新Addendum批准后仅授权本E-only冻结数据Q组实验、只读目标检查及**条件前瞻train-only**，新增阶段上限**3600 job秒／1GiB输出／18000实际＋保留5ms物理ticks**，包括真实GPU0单卡冷启动历史150–167秒、日志/优化/完整Q模型Adam/target checkpoint/IO/失败/前缀重放与资料发布。**这是包含在原R2累计57600秒/12GiB内的子额度，不是加预算。** 运行前按实际Q batch更新速度、冻结replay读盘体积、前瞻prefix长度重新保守投影；无法容纳则BLOCKED，不以过度压缩/删旧checkpoint或跳物理复核凑预算。共享free≥100GiB、任何其他GPU0重任务占用则停报，不自动抢卡。

遇到非法Q/action、来源哈希/RNG不匹配、optimizer/target/更新组数错、训练/评估数据泄漏、真实物理硬界、存盘失败或预算触顶：停对应job，保留首失败，**只允许一次同输入只读调试与旧档案复算**；失败诊断后STOP_AWAIT_DECISION，不自动重新设计sampler/lr/EMA/reward。512未过旧40门，则不做前瞻、不启Actor/I；512过旧40但前瞻失败/不足，记录Q对已曝露数据拟合但无独立资格。即使前瞻通过，也不自动启动原32k训练。

交付：G17-QA1正式协议和源/父组/phase/固定schedule哈希、旧Q56与两个分支各128/256/512快照、每组unique/per-parent exposure/target EMA、冻结40条数值/排序/成本、可选前瞻4父组物理、Q值对当前reward的calibration/n-step审计、全部错误与消耗、研究下一决策。提交后同步Issue #63和Master #18，保留v1.1已签合同与旧失败。

**签署状态：本修订尚未获Yi新批准；创建/审阅本文件不允许Codex先启动Q optimizer或新物理。**