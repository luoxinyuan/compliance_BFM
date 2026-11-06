# CURRENT_STATE — 新会话先读

最后核对：2026-11-06 20:35 CST。**Issue47已完成4个PPU后端回归、28个完整接管/因果对照、两seed各16×768真实PPO。两组按原开发规则均选择batch8；联合延长门槛未满足，训练已经结束，不再追加。冻结终检`final-02`正在执行，尚未科学结项或推广新模型。** 早期C0→I恢复有效但晚期不可靠；C0能在部分已起步状态继续行走，不是完全不会走。

[当前阶段结果](C0_ENGAGEMENT_RESULTS.md) · [预登记协议](c0_engagement_protocol.json) · [Issue47](https://github.com/GaoE05/robot-research-bridge/issues/47) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18) · [BFM原生复用最小路线](BFM_REWARD_REUSE_ROADMAP.md)。RESEARCH_OVERVIEW是长期纲领；ChatGPT Library2026-10-28是历史快照。

## 科研目标与可靠基线

Contact-conditioned whole-body behavior selection/modulation/transition：依据真实接触与身体状态，合理组织局部手臂、躯干/root及支撑响应；不预设先手后脚，不以root走远替代任务合适性。冻结BFM/PD。有限直行Teacher→ideal-force Student、原生局部伸臂、有限双向转腰均已成立；没有验证真机精确外力感知。

- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- 实际行走基线I `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`。
- T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`；S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`。
- D2 `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`。
- C0局部有效，72cm持续需求下也会真实迈步，但启动及持续推进迟到；不再仅概括为长距失败。C1恢复walking却失去局部优势。Issue45 F/R均未稳定共存，冻结I参数不等于闭环保旧。全部旧权重/失败保留；不按任务拼模型。

## Issue46实际结果

1. 原reward同条件复算认可C0局部和I长距有效解，未改reward/网络/历史/绝对z/BFM/PD。10N主物理、K80/D8、原观测尺度；5N回归。两个seed同I actor，独立fresh critic/Adam，std=.008，各32×768；无新增监督初始化/Student。
2. 两组固定开发规则均选择iteration0。末轮分1.619/1.612，I1.500；预登记联合延长条件不成立，未追加训练。best两个文件不同critic SHA，但均值actor/归一化与I相同，最终评估去重，不能算新学习成功。
3. **部分正结果保留**：同一4601末轮在新6/24/72cm误差4.190/4.617/5.478cm，I4.452/5.290/7.632cm；72cm真实力5.877→4.511N。4602长距改善但新6cm退化4.790cm。不是两个seed稳定共存。
4. 停止取舍：新共同前缀停止4601为4.956cm/4.038N，I4.387/3.602；旧早停止6.062/4.967 vs4.707/3.889。退化发生在仍接触的移动/保持阶段，不能用释放后自然收步解释。共同前缀7.37s前实际physics/命令一致，分叉后可反馈调整，但不是最优行为分配。
5. 动作质量限制：新6cm4601足部路径.33/.75cm，远少于I20.80/.80cm，却伴随root降低6.77cm；世界腕相对肩X仅+.90cm，C0+4.32cm且误差仅2.293cm。减少脚步不等于自然伸臂已经成功。继续保留XYZ、相对运动、滑移和连续性。
6. 242完整审查回合无安全终止；seed4602中距有共10个无支撑标记tick（单段10–15ms），未隐去。最大q_PD20ms跳变68.29°不等于实际q5ms3.60°。无接触稳定，释放后不考核旧anchor跟随。
7. 探索噪声能改变同状态BFM动作，未证明std太小；但单步方向覆盖不等于持续有效反馈。raw logprob复现≤.0001221，联合KL≤.01996，critic EV提高不能代替actor成功。每seed208完整采样回合，减速决策仅136/213；覆盖及优化共同限制尚无唯一根因。

## 当前授权：Issue47 C0局部→全身参与（接管/PPO已完成，冻结终检进行中）

[详细记录](C0_ENGAGEMENT_RESULTS.md) · [Issue47](https://github.com/GaoE05/robot-research-bridge/issues/47) · [协议](c0_engagement_protocol.json)。[方法定位与自主学习思考](CONTACT_LEARNING_RATIONALE.md)。用户观看C0后更认可其先局部、后身体参与的组织，当前优先研究delayed engagement；旧原生reward-prompting建议改为后备工具，未执行。

讨论结论：继续既定C0/I接管有价值，用有限同状态干预区分条件运动能力、启动/维持反馈与恢复域；证据支持后才进行原授权PPO。最终主线是接触/身体状态条件的自主行为选择，不是人工goal答案拼接。人工goal用于资格、初始化和探索；BFM风格先验及PPO均不保证任意状态/latent序列自然或全局收敛。这部分是方法澄清，不扩大本轮实验变量、预算或多部位/新力范围训练。

进一步讨论已写入CONTACT_LEARNING_RATIONALE第8节：原生候选可提供有起点的探索，真实源状态/共同前缀需求变化应常态覆盖，后续明确研究“蒸馏初始化→部署观测Student真实闭环RL→可靠专家聚合”。均是待验证方法候选，不宣称std=.008够不着已被证实、不保证原生候选便宜且合格或Student RL必胜；严格最差任务/频繁回滚也需防止冻结学习。Issue47原预算仍不变；在用户授权继续后，用新PPU完成原任务回归、因果接管并启动条件PPO，没有启动新科研方向。

C0精确SHA `ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c`，`runs/prior-recovery-20261105/correct-C0/C0.pt`。I起点、750条六类示范、2000监督更新；原C0权重冻结，本轮PPO后继4701/4702另存；没有在线先手后脚规则。仍是1088D特权Teacher，不是部署Student。

已完成保存72cm轨迹100ms/50Hz分析：C0首次持续10N约8.18s，左足约10.22s卸载、10.35s落地，前进约5.90cm但root几乎未推进；约12.43s才建立持续root前向速度。最初至7.5sC0误差3.36cm/力2.93N，优于I4.77cm/4.25N；后续明显追赶不足。C0 7.5s目标z已偏离standing约52.6°，不能说高层一直没发命令。尚未隔离高层、BFM反馈和外力贡献，也没证明手臂已耗尽自然可达余量。

### 2026-11-06真实进展与运行身份

- 物理runtime `d24622b259f984a42a3f628e983a15e8ea16d547`；4个基线＋28个完整回合，未触发安全边界。C0/I双向接管逐项保留MuJoCo integration/solver、BFM观测与动作历史、实际上一z、力/接触与控制记忆，同后端分支前缀精确一致。
- 72cm全接触误差/力：C0 **30.827cm/8.625N**，I **7.642cm/5.885N**；C0→I@6s **7.117cm/5.598N**、@7.5s **7.126cm/5.016N**，@9s却 **27.655cm/8.625N**。不能将早期恢复标签推广到后期。近6cm原C0 **2.293cm/1.895N**，切I@6s退化至 **3.530cm/2.991N**。
- I→C0@7.5s可接续已组织行走（7.541cm/5.572N），@6s却30cm附近；能力依赖真实源状态，不是C0完全不能走。固定6s实际z、BFM仍实时反馈时，带力root仅前移约3.81cm且没有真实抬脚/卸载；正常C0约49.98cm且有步态。进一步真实撤力仅约0.44cm。高层变化有因果贡献，未量化“主动百分比”，固定z不是完全被动机器人。
- 两seed4701/4702均从原C0 actor/归一化/std逐值相同起点开始，独立fresh critic/Adam；没有在线I/M10/Cyclic、没有额外BC恢复标签或Student。正式PPO runtime **`eeeb8979258b774a87dbedf8b17d18e3d15fc47a`**，两独立worktree；同一独占PPU，两job各3个worker，不增加第二张卡。
- 首轮尝试runtime `4acd257790f517cff94fbc23fd0f2d01b9bdd5e1`各采768决策，在第一次更新前因PPU批量/单条FP32 arithmetic导致原logprob断言失败；**共1536采样决策计入预算，0正式更新**。逐条重算原logp完全一致，改为learner与采样同一batch-one actor路径后，原3e-4断言通过且梯度/Adam/KL回滚正常。未放宽阈值或更改raw概率空间/std；不是已证实的历史PPO错误。
- 第8批：4701等权near/mid/far/stop开发相对代价 **0.9796**，far **0.9497**、near **0.9853**；4702 **0.9943**，far **1.0036**、near **0.9745**。两者局部保留，但长距收益尚未重复，连接零需求仍有退化。16批已完成：总体开发代价4701 **.9967**、4702 **1.1089**；后者near退化至 **1.5100**。两组均选batch8，不满足联合far≥5%收益门槛，未延长。不用混合任务采样force曲线宣布改进。
- 源姿态留出`new_ready205`在训练前修正缺失ready_progress及不实际进入arm-ready的前缀：真实执行到6s、进度.45，接触时刻等整体后移1.8s。其余训练/测试条件不变，详情见协议amendment。

运行根目录：`/root/compliance-ppu-20261106/issue47/`；正式训练`ppo4701-r1`、`ppo4702-r1`，worktree为同名`repo-ppo-4701-r1/4702-r1`。失败初尝`ppo4701/4702`原样保留。SDK每次shell须加载`/usr/local/PPU_SDK/envsetup.sh`；不改运行中源码/环境。原NVIDIA实验目录和共享checkout未动，BFM推断与梯度都在PPU，不CPU替代。

原上限14000累计job秒/49152新RL决策/9GiB。首段两组共24576正式决策＋1536失败采样；仍需计入评估/渲染及未提交部分。扩展只能依原16批联合开发门槛，且扣除失败样本后等量延长，不能无视总cap自动各跑32。新增监督/Student更新0。实际正式24576＋失败1536＝**26112 RL决策**，35＋37＝**72次正式Adam更新**；两训练job墙钟1812＋1761s。缓存更新smoke为独立诊断，不当正式候选。精确整阶段结算待终检/最终视频完成，不沿用旧PID或空闲快照。

**冻结终检接力**：runtime `c6f4a31c3ad32c62d2ce8be810a85fe0d195fa46`，`/root/compliance-ppu-20261106/issue47/repo-eval`，输出`final-02`；同一PPU，3worker，0新增RL训练。固定比较C0/I/4601及两seed开发选定/末轮模型，原15开发＋6新封存＋7明确暴露回归/5N条件；在新评估前锁定，不按终检改选或补训。监督进程有2400s硬时限；预计21:00 CST附近完成。若有timeout，仅检查/补齐未完成冻结比较，不重跑PPO。实际owned supervisor PID12948、learner12953是20:35快照，接力先查活进程，不按旧PID操作。启动壳SDK参数错误及新worktree缺私有motion symlink的18s失败完整保留，未产生正式更新；修复仅隔离launcher/已有资产链接，无物理改变。

选定4701 SHA `f23a6818dab277b775b41d5926641c359546cd6d5f6b2f2003a33514c446e2f6`；4702 `37f2c7e54375df301c9cc8160b2363ff8b8eb03edf251b5b1b93cdac583ef4ce`。初始/候选8/候选16/末轮全部保留，[实际训练曲线与数据](c0_engagement_artifacts/ppu-training16.json)及[图](c0_engagement_artifacts/ppu-learning-curves.png)。当前仍不算已解决局部/长距共存。

新接管视频已同步Mac：`~/Projects/compliance-bfm-diagnostics/videos/c0-delayed-engagement/01-causal-recovery-window.mp4`＋INDEX。四格为C0、C0→I@7.5s、C0→I@9s、I→C0@7.5s，正常1×真实qpos，覆盖完整接触和释放恢复；它证明恢复窗口差别，不是新统一模型。SHA/14.96s/374帧/完整解码和画面已核对。

PPU基础兼容检查是此前独立准备：[PPU_COMPATIBILITY_RESULTS](PPU_COMPATIBILITY_RESULTS.md)。原18状态BFM/q_PD微差及8s基础回合通过；这不等于正式任务跨平台逐位相同。原NVIDIA严格断言的微小prefill数值差保留，正式接管使用同后端完整前缀精确校验。[阶段报告](C0_ENGAGEMENT_RESULTS.md)记录实际物理、概率路径修复及进度。
## Issue46历史身份与成本（不表示当前GPU空闲）

训练runtime `f6ca5c5f6c4de205b3457bef4a024345e2105009`；最终评估/渲染 `057f5c6bd302185097770c3c1ed4d10dfb2ce109`。长期个人分支`gaoe05/research`；最终文档commit见Issue46链接，勿混作训练runtime。

- 4601 last32 `8fa4759d3766568dee8fbba0367e772a97977694d19a503868708efa6d8c9228`。
- 4602 last32 `e9a1972166b25e3b915b8d75e97d3377e540583e215d91f269e8dba4550310c9`。
- [完整模型清单与服务器路径](coexist_artifacts/final-frozen-models.json)；[资源账本](coexist_artifacts/resource-closeout.json)。
- Run：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105`。初始/每8批候选/best/last、原始rollout、所有失败均保存。
- 总49,152高层RL决策、194实际Adam更新；0新增Student/监督更新。7711.53 GPU-job秒＋165.02分析秒＋180秒保守CPU余量，约4.24GiB，低于授权18,000秒/12GiB/98,304决策。训练两独立worktree未修改。
- Mac：`~/Projects/compliance-bfm-diagnostics/videos/feedback-coexistence/`，仅3个1×真实qpos MP4＋INDEX；6cm、72cm四模型对照，以及同4601停止/继续。SHA/时长/完整解码/视图已核对；未挑seed重跑。
- owned进程均结束，GPU2/5/6释放；共享checkout仍`6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`干净。未改BFM/PD/原权重/main、SoftSONIC/他人任务，未做真机。

新会话先读本文件、RESEARCH_OVERVIEW、C0_ENGAGEMENT_RESULTS及Issue47，先核验final-02运行/完成/成本；不重启已结束的4701/4702。终检完成后作全任务/分阶段分析与剩余1–2条关键视频，再正式结项，不能拿开发改善代替终检。核验runtime/模型SHA与资源，不复用旧空闲快照或PID。历史Issue44/45、M41仍有效，不能用这轮局部正例抹去停止与跨状态失败。
