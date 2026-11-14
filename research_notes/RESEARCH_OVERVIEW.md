# Humanoid Compliance：长期研究纲领

本文件说明研究目标与判断原则；执行状态、模型身份及下一决策见 [CURRENT_STATE.md](CURRENT_STATE.md)。完整历史与里程碑决策在 [Project Master #18](https://github.com/GaoE05/robot-research-bridge/issues/18)。它不是实验启动指令。

## 当前方法决策（F1收官，Issue61）

Steering-first只作为下一研究主线：分离目标Goal、当前接触Guidance与冻结底座Execution，先验证真实Guidance→合法动作/latent合同。原BFM、P-BFM原生ERI与I-BFM交互表征应按权重、动作、观测、物理资格分别审查；接口可调用不等于真实接触能力成立。

F1证明有限状态有可执行反馈，也证明真实未来并非尾部质量问题的通用解；不据此宣布必须解冻或已有可部署通用策略。当前只授权S0，Contact Adaptation是否必要留给匹配物理证据，S1及训练不可自动启动。历史LPP/action alignment/Teacher→Student仍为有需要时工具，不强套形式。

## 最新接力：Issue63 R2 E/I v1.1

当前执行方法为全新零U、原Actor尾部副本的在线适配，配对比较E执行适配与I交互后继表示；不继承B19/A128旧Guidance或Adapter。四块真实Q-only及独立新进程采样/更新恢复已通过，Actor/U尚未更新，不能据此宣称任务能力或BI收益。接下来按原合同取得3500–5000条真实后继与Q排序资格，然后才能进行受保持约束的Actor/U学习。详见[入口结果](R2_EI_Q_ENTRY_RESULTS.md)、[v1.1批准附件](R2_EI_V11_SIGNOFF.md)。Dev用于最终方向性比较，sealed本Pilot不执行；旧判定和失败保持。

## 科学目标

**Contact-Conditioned Whole-Body Motion Transitions / General Whole-Body Compliance**：机器人接受人的局部物理引导，根据真实接触、本体与动作历史，以及自身可维护记忆，组织任务合适的手臂、躯干/root、支撑和步态响应，降低不必要的交互负担，保持稳定与运动质量。

例如同样向前拉右手：余量充足时可以主要动手臂；接近自然可达边界时可能需要身体协调或移动。实际响应应由身体状态、接触需求与已有运动能力共同决定，不是写死“先伸手、再弯腰、再走”。不锁臂、不人为削弱基线来制造多行为必要性，也不以迈步数量衡量成功。不能把某条+X轨迹、手工姿态插值或一个checkpoint当成最终目标。

## 为什么使用冻结 BFM-Zero

BFM提供预训练的全身运动先验与可执行的行为latent接口。外部策略选择、组合或调整合法latent，经冻结BFM actor和原PD落实为真实控制。冻结底座有助于复用已有动作、约束研究范围并明确学习发生在哪一层，但不保证任意专家动作都可表达，也不保证原生motion与当前接触任务适配。

必须区分：模型接口允许生成z、原生动作实际可执行、专家在真实访问状态有效、Student闭环有效。reward名称或演示不能代替资格证据。底座权重、观测预处理、控制节拍和action/q_PD映射均需有对应身份；历史上一个狭窄latent范围失败，不等于完整BFM无能力。

## FIBER-inspired LPP 主线

```text
任务反馈Expert（训练期可用当前特权状态）
  → 同状态真实 action/q_PD 目标
  → 冻结BFM下的 action-aligned latent projection
  → 可部署Contact Policy / Projector
  → 合法latent → 冻结BFM → 原PD → 真实MuJoCo闭环
  → 在Student实际访问状态重查询Expert → 数据聚合与更新
```

专家可以是原生BFM行为、任务反馈算法或学习型特权策略；不要求先有万能Teacher或两个命名技能。专家已输出同权重合法z时，验证动作一致性后可直接保存标签，不必重复近乎恒等的优化。异构关节动作则需投影并验证表达与物理执行。Projected-z监督与经冻结actor的Action Alignment均是有用方法，不要求后者每轮胜出。

Student只能读部署可得的force/proprio/action history、合法接触位置编码及自身记忆，不能读隐藏anchor、未来轨迹、实验编号或专家真值相位。历史力必须保留其link、作用点、坐标系与有效性。仿真知道的接触位置和力不自动等于真机能够准确测到。

逐状态特权z优化是训练工具，不是可部署策略；单点动作MSE低、直接模仿alpha或播放固定时钟轨迹，不足以称为完整LPP。FIBER启发的方法要素与本项目实现见 [FIBER/LPP架构审查记录入口 #21](https://github.com/GaoE05/robot-research-bridge/issues/21)；本项目尚非完整Flow Planner或SF模块复现，不把SF距离当物理质量真值。

## 与 Xinyuan 路线的真实区别

团队已有直接contact-to-latent RL/PPO实验：通过交互奖励优化输出BFM latent或residual的策略。**该路线也可以冻结BFM**；差别不是“我们冻结、对方重训”。本研究主线将反馈专家获取、同状态动作投影、部署信息受限的Student学习及闭环访问数据聚合分开研究。特权Expert内部使用高层RL是行为来源获取，不代表已开展正式PPO-vs-LPP比较，也不应把团队直接PPO原型称为已完成LPP。

## 可能形成的独立 insights 与证据边界

|研究问题|需要的证据，而非预先宣称的创新|
|---|---|
|任务适配的行为组合|调用walking/下沉时保留接触手要求，抑制无关抬手；证明收益不是姿态美化或牺牲任务|
|专家恢复域|从Expert自己轨迹有效，能否扩展到Student访问偏离状态；区分查询合法、动作可表达与真正可恢复|
|局部接触的部分可观测性|接触意图、释放与当前姿态保持是否能由部署历史/自身记忆辨识；同状态因果对照，不泄漏时钟标签|
|随机训练与部署一致性|学习反馈在均值执行中是否有效，而非主要依赖动作噪声；匹配探索、训练预算与任务分布|

冻结BFM、LPP、动作对齐、数据聚合、PPO专家获取本身均是已有方法。上述问题目前是研究假设或有限域观察，不自动构成论文创新。需通过合理基线、因果消融、独立测试与适当重复建立结论。明确区分能力来源、Expert收益、投影损失、Student差距和泛化范围。

## 最终目标与验收

仿真目标是共享策略在真实同回合接触变化、不同自然身体状态与接触部位下产生有价值的协调，而非分别播放几个姿态。逐步扩大技能族，以可信能力为前提，不强求当前阶段具备行走、跑步或任意多点交互。

报告真实XYZ力、限幅比例、三维误差、腕/肩/腰/root参与、脚步/支撑/滑移、q_PD与实际关节连续性、无接触行为和完整停止恢复。相同初态/伙伴规律不代表应伪造相同实际力。Tether是保持接触引导；未经单向接触适配不得称为真实按压。不能用加大/降低force cap、提前断接、anchor追随机器人、摔倒早停或裁剪不利时段制造进步。

开发与终检分开，测试暴露后注明开发身份；保留负结果、各任务族和退化尾部。视频是保存真实qpos的1×重放，不能以reference/q_PD动画冒充执行。未来Unitree G1 demo是目标，**不是当前已验证能力或真机授权**；进入真机前须另行审查传感可得性、模型/PD部署一致性、接触力估计、保护/急停、停止与恢复、有限域安全和资源权限。仿真成功不等于sim-to-real成功。

## 协作、版本与长期维护

- 用户确定研究目标、里程碑授权、资源和重大方向决策；不能把历史临时阈值永久化。
- ChatGPT负责科研讨论、独立审查与阶段决策整理；以可读源码/真实报告判断，不把旧Library或计划当运行事实。
- Codex负责源码检查、普通工程修复、授权内实验/训练、反证与失败分析、版本及真实物理交付。阶段内自主推进，重大架构/科研方向、额外资源、共享系统或真机风险才提前交接。
- `gaoe05/research`是长期个人集成与审查分支；实验用独立个人分支/worktree，运行源码固定。里程碑获准整合后严格检查历史，普通push，不改main、不强推、不自动合并团队变化。准确commit保留历史；阶段分支无需永久保留，但删除必须另行授权、核查引用，本次不删除。
- 每个里程碑结束：更新短小CURRENT_STATE、Project Master摘要/链接、阶段报告与代码；记录runtime commit、checkpoint SHA、失败和实际资源；整合获准代码。普通实验只追加阶段索引，不重写全史。
- Git只存审核过的源码、配置及小型摘要；权重、原始轨迹、大数据、凭据和未获准资产不公开。已有服务器/其他用户任务保持原样。
- ChatGPT Library `/Humanoid_Compliance/Current_Project/` 中2026-10-28的PROJECT_STATE/SESSION_HANDOFF为历史快照。Library以后主要放笔记与长期insight；本文件与CURRENT_STATE、最新Issue是接力入口，不要求Codex自动编辑Library。

窄域专家获取澄清：特权Expert可直接以同权重原生motion latent作为名义运动先验，独立初始化actor/critic/优化器，不要求在线依赖已蒸馏Student。参考相位与接触激活是明确的自身控制记忆，不是未来伙伴轨迹；学习收益仍须由真实闭环反馈证明。

最新Expert方法澄清：原生motion可作为运动能力/初始化；force→walking参与、相位推进和停止应由学习策略决定。旧CyclicFeedback只作历史/初始化正控制，新Expert不在线依赖它。Structured与direct合法latent是本阶段对照，不将任一表示预先宣称正确或作为永久门槛。

接触几何扩展路线：单/双点平面引导→多部位分别/组合→高度与移动→未见接触配置和身体状态泛化。逐点力与身份必须保留；合力可能抵消但交互负担和力矩不为零。单点位置跟随不唯一规定身体朝向，双点平移/旋转需区分腰部调整、pelvis朝向与真实支撑转移。受力总预算相同不等于力矩或做功相同，不能以任务名称代替真实运动资格。

## 已授权方向与长期阶段路线

M38/M38-B：保旧旋转Expert（有界对照已完成，双向闭环仍有边界）；M39：连续平移＋旋转切换；M40：跨腕/躯干共享Student；M41：同接触需求跨身体状态响应；M42：多部位/高度/未见组合；M43：真机传感与Sim-to-Real。M39–M43是研究规划，不是自动启动授权；转腰只是当前能力来源，不将总体目标缩为腰关节控制。

M38-B方法澄清：真实执行包含上一条z与5个BFM子步；静态动作对齐应与完整过渡和自身访问历史区分。单点/子步动作误差小不保证闭环稳定，上一z输入敏感性也不自动证明它是唯一根因。小幅转腰是有价值的行为，不强制迈步才算成果；反向yaw则不能被平均误差改善掩盖。

M38-C补充：接触转角必须与同时刻、同框架的真实无力分支比较，固定高层命令回放仍包含BFM本体反馈。真实访问恢复后继标签不等于大量独立失败状态；自身z短序列展开不等于可微物理。D0/D2取得新双向响应而D1封存反向，说明数据、历史监督与保旧有不同取舍，不能用单次离线损失或某个方法名称替代物理泛化证据。下一步保持M39有限过渡能力验证方向，未自动授权大规模连续训练。

最新主线：transition-centered Teacher获取→受限观测Student→代表性运动/接触组合泛化。源运动定义真实初态分布，目标由持续的任务物理质量定义；不以某帧B距离或命名技能替代成功。LPP/action alignment按实际来源需要使用，合法原生z无需重复近恒等投影。

任务分配审查补充：长距离腕部跟随与学会自然arm-to-locomotion分配应分开评价。对同一实际初态同时记录腕相对肩/躯干、机械余量、真实力与足部参与；近/中/远及速度覆盖不能只靠最终root位移判断。原生goal伸臂正控制是能力证据，不是接触反馈学习；Teacher当前几何特权也不应被称为force-only。保留M41启动历史结论，但保旧不应把合理小请求强制变成walking。

BFM原生行为复用积压项：reward prompting、投影前状态reward线性组合、F/SF候选后果排序，见[最小研究记录](BFM_REWARD_REUSE_ROADMAP.md)。B状态合同不含任意外部伙伴/接触reward；不能把reward_inference宣称为零样本通用contact policy。当前阶段不新建planner、不重训FB；先以真实物理检验来源。


### C0视频复审的方法提醒（Issue47）

低平均误差与任务合适的行为组织需分别审查。C0的先局部响应、后段真实迈步即使追赶迟到，仍提供可复用的行为线索；不预设更早walking总是更好，也不凭这个顺序宣称学会了可达余量判断。优先用完整历史同前缀接管界定有效恢复与高层反馈贡献，资源不足时保存准备状态，不CPU替代BFM。当前执行事实仍以CURRENT_STATE/Issue47为准。


### 自主反馈学习与示范的分工

最终重点是让真实任务反馈决定行为选择、调制与转换。人工goal/示范可提供已验证能力、初始化和探索来源，但不能以持续拟合人工答案代替这一问题。合法BFM latent和运动风格先验都不是任意闭环轨迹自然的硬保证；有限预算RL也无全局收敛保证。失败需区分可执行能力、观测、探索、奖励与优化，不能直接反推人类不会作相应响应。

[方法定位、I/C0继承链及Issue47干预如何支持下一决策](CONTACT_LEARNING_RATIONALE.md)保存了2026-11-05与用户的讨论。多部位力组合是后续方向，未因本讨论获得自动发车权限；当时等待GPU；用户随后提供PPU并授权继续，实时进展见CURRENT_STATE。


### 后续方法候选：能力引导探索与Student自身闭环优化

优先复用同权重原生goal/reward/motion候选作初始化、示范和探索来源，让共享策略学习真实源状态下的需求变化、持续响应与撤销；候选不等于已验证技能，不限定部署为固定动作菜单。把蒸馏后受限观测Student的真实闭环RL与可靠专家访问聚合列为后续方法候选，reward/critic可训练期特权，actor权限不扩大。它不能消除不可辨识信息，也不保证胜过BC；需要有限对照。当前已有奖励是逐步累计，不将这一构想误写为首次闭环或首次全过程优化。最差任务监控与回滚用于发现严重退化，不能机械要求所有旧指标不变。

具体依据、反例与当前授权边界见[方法记录第8节](CONTACT_LEARNING_RATIONALE.md)。这是长期研究构想，未修改Issue47实验或启动Student PPO。


### Issue47：启动、维持与恢复域

完整真实前缀接管表明，C0可维持部分已经组织的行走，却在原站立牵引中启动迟到；早期I反馈可恢复，晚期不可靠。固定高层z且保留真实外力/BFM反馈不足以复现正常C0步态，因此高层反馈确有贡献，但不能用差值分配主动百分比。C0起点PPO获得一组局部/长距有限共同收益，未跨seed与5N稳定，且前向追赶改善伴随侧向负担。下一候选是已有有效反馈方向引导训练期探索，不是部署时手工切换；尚未执行。源能力存在、被探索到、闭环稳定调用与部署观测泛化依然是不同问题。详见CURRENT_STATE与C0_ENGAGEMENT_RESULTS。


### Issue48：原生能力主线正式启动
用户新授权以原生goal/reward/motion提供可追溯起点与探索方向，独立共享Direct actor进行真实transition接触反馈RL；不再沿I/C0高层继承链修补。候选用于先验和探索，不限制为技能菜单。C0机制已结项，仅为冻结参考，不是原生主线前置。后续路线为受限观测Student初始化与自身闭环RL，当前不自动启动Student。详见CURRENT_STATE及native_protocol。


### Issue48结项：原生先验与实际反馈学习的边界
独立fresh高层网络可以从可追溯的原生goal/motion起点进行真实transition PPO，不必继承旧I/C0/T30 actor。三组各24576决策中，均值head坐标组在零需求/局部/停止学到部分物理收益，但源相位泛化与从局部进入及时walking仍失败。原生运动可执行不等于某个单时刻协方差就能有效探索完整运动过渡；也不保证源数据的低拟合误差构成高质量接触专家。实际结果见NATIVE_CONTACT_RESULTS。预定相对最差规则对近零基线可过度敏感，不能以best=initial否认真实部分收益，也不能根据封存事后换best。下一候选是有时间连续性的原生动态行为探索；尚未发车，Student自身RL仍为后续有限对照，不自动扩大general训练。


## Issue50：原生能力与短程物理反馈的角色

原生goal/reward/motion是可追溯的运动先验，不必先成为万能接触专家。原始同checkpoint replay仅支持其机器人状态reward合同；当前外部伙伴force/anchor不能冒称直接零样本推断。普通与softmax加权推断、未投影embedding与归一化z必须区分，名称和合法z不等于行为资格。

短程物理搜索在完整真实状态上比较原生提议，只执行首指令并根据新反馈重规划；预测不使用伙伴未来轨迹或停止时刻。它是昂贵的特权反馈获取工具，不是实时可部署Student；预测改善、真实闭环收益、可摊销学习分别验证。I/C0作为冻结能力参照，原生新主线不继承其高层actor。

后继仍为可靠反馈获取→受限观测Student初始化→自身真实闭环RL与有效专家查询；当前阶段不自动训练Student，不要求先覆盖全部motion或满足历史毫米/固定恢复时间门槛。


### Issue50有限证据：原生来源与可摊销反馈的分工
同checkpoint三种robot-state reward推断已真正执行，但6个候选未形成可靠低力接触walking；合法z与实际反馈质量必须分开。原生goal/motion加完整状态短程物理搜索，在相同.15牵引和严格共同前缀停止分支获得真实收益，未继承旧I/C0/T30 actor。改善来自反馈组织而非新reward-z被选或新PPO训练；预测分支不能当实际执行/on-policy数据。严重超前域仍无可靠恢复，约23秒计算才给100ms命令，偏航/指令峰值仍存在。可将其作为有限域反馈与后续摊销来源，不当成实时Student/general成功；下一候选为受限观测BC初始化与自身闭环RL，尚未启动。详见NATIVE_ROLLING_RESULTS。


### 候选、反馈搜索、学习与未来信息的分工

原生候选用于提供可追溯的运动起点，不要求它先完美解决接触任务，也不把reward prompting成功设为共享反馈学习的前置。完整状态短程物理搜索可作为特权Teacher而不训练网络；它与PPO参数学习、蒸馏及Student自身闭环RL是不同收益来源。慢牵引可能支持局部响应，但减力reward不保证唯一的先臂后脚顺序；需要运动经济性与持续任务质量的实测取舍。

用户允许后续尝试未来可见Expert作为能力上界。当前因果Teacher、未来可见actor和特权critic需分别登记，保留部署actor权限；相同可部署历史下依赖未知未来而不同的动作不能默认完全蒸馏。该方向未自动启动或获得新增计算预算，见[最新讨论第9节](CONTACT_LEARNING_RATIONALE.md#9-issue50讨论澄清候选物理搜索ppo与未来信息)。


### Issue51：有限覆盖、可蒸馏性与自身闭环学习接续

在已验证的有限原生滚动物理反馈上，先补少量完整新条件并分离真实首命令标签，比较fresh特权BC与受限ideal-force BC的真实闭环。按共同前缀族及整回合划分train/dev/final；特权BC失败先查实际标签、优化、隐藏控制记忆与自身分布，受限BC差距另查可辨识性，不凭离线latent距离直接断言。有限正结果可支持下一阶段冻结BC vs同初始化Student自身RL，但当前未启动PPO，不恢复旧高层继承链或要求万能Teacher。已知搜索偏航、超前恢复与计算边界继续保留，见CURRENT_STATE及Issue51预登记。


### Issue51结论：搜索反馈、特权摊销与受限闭环需分层

四条新增实际回合支持原生goal/motion＋因果物理搜索的有限有效域；fresh特权BC仅部分摊销，fresh受限ideal-force BC尚未获得可靠闭环。后者在接触前已分叉，平均latent MSE与完全重复输入检查都不足以证明闭环可学或信息不足。关键动作尾部拟合、实际执行历史和搜索自身记忆需分别定位。保留旧有限域迁移成功，本轮不启动PPO；下一步优先验证真实受限访问域的可靠反馈与小预算自身学习，勿以增加候选或旧高层继承链修补代替。详见[NATIVE_TRANSFER_RESULTS](NATIVE_TRANSFER_RESULTS.md)。


### Issue52：BC仅初始化，主要优化发生于特权Expert真实反馈PPO

用户新授权以原生搜索反馈蒸馏的P4000热启动高层actor/normalizer，fresh critic/Adam，直接按真实接触任务PPO优化。受限BC不作为前置门槛，当前不训练Student；不继承旧I/C0高层网络。奖励排序、采样命中有用变化、优势信用、均值动作保留及物理结果分别诊断。候选/搜索片段不得冒充on-policy采样；无默认硬模仿保持。训练目标仍是状态相关全身行为组织，非固定理想z。实时权限/预算见CURRENT_STATE、Issue52及p4000_protocol。


### 路线A小型交互验证（Issue53）

当前授权优先检验原生物理搜索Teacher在受限Student真实访问状态中的有效性及成本；有效后有限交互聚合与Student自身闭环RL，不能把特权Expert PPO当作Student学习收益。FB只作当前同权重F/B与已有reward-z的影子提名，正式动作仍经原7候选真实物理排名；不将F的机器人状态价值误称完整接触return。实际状态与停止结论见CURRENT_STATE/Issue53。


### Issue53：交互恢复证据与完整状态合同

有限Teacher接管可将同一受限Student送回可执行域；旧数据续BC也能改善新条件，所以需用匹配对照分清训练优化与访问分布补充。本轮交互臂因跨reset闭包遗漏未形成可信方法比较，Student自身RL未执行，不因配方形式虚报成功。完整状态还包括任务生成器/接触callback的spec与原点绑定，数值fingerprint相同不足以代替不间断真实控制对照。修复与独立错误证据见Issue53报告。

当前适合下一阶段的S-init是通过有限新条件的旧数据BC500；冻结BC vs部署权限actor自身RL值得受控验证，Teacher查询限定有效恢复域，不把万能Teacher/完美混合BC当全局门槛。FB只作shadow，当前机器人状态F代理与真实外部接触任务错位的有限结果不构成整个FB方法否定。长期目标仍是合理的接触条件全身组织，而非把本轮直行/存储修复当最终课题。


### Issue53小续跑：标签的局部有效性与全任务增量分开

34条已验证标签来自两个Student起点及相关Teacher恢复后继；按原形态交互聚合只改善局部小请求，未显示整个有限任务族的增量。Student自身RL从干净BC起点在两个seed中期都有开发收益，但一组末轮退化、另一组用户提前停止，未形成完整双seed封存结论。有效接管、标签监督增益、Student自身RL收益与稳定泛化必须分别归因；当前不自动追加标签/重复seed/延长训练。后续先单seed稳定真实收益再复核，并在授权预算内主动核查和利用CPU/GPU/RAM/I/O并行余量，累计job成本不因并行少计。详见[续跑结项](INTERACTIVE_INCREMENT_RESULTS.md)。


### Issue53续轮：真实开发收益、训练稳定与独立样本资格

逐批独立固定评价可以保留Student PPO的真实开发收益，但中间回退、状态/支撑闭环放大和滑移代价不能被best分数隐藏。5403有限验证中开发best与末轮同为batch7，仍不能宣称持续稳定或未见泛化；BC500继续冻结。评价不得改变训练任务与采样RNG，固定任务的物理快照一致不等于所有随机任务生成器已被恢复。

F校准的数据资格以独立Student回合/共同前缀族为单位；20个时刻的180配对或712候选行不能替代足够独立任务组。当前仅4组，按预登记暂停拟合，不把训练相关性当增量证据。完整证据见[RETENTION_RESULTS](RETENTION_RESULTS.md)，本轮结项，无自动后续训练。


## 双轨共同合同与运行能力（2026-11-09工程澄清）

冻结BFM的外部高层路线继续保留；另一条基座FB接触扩展路线目前只完成独立权重warm-start和原拓扑完整update的工程测速，尚未获准正式接触扩展训练。二者以[DUAL_TRACK_CONTRACT_V1](DUAL_TRACK_CONTRACT_V1.md)草案共享任务/物理/信息权限/父组与质量判据；分布数值、折扣真实时间匹配、FB contact schema与资源需Yi批准。现有库无需为本轮新fork，可靠接触环境仍num_envs=1，批量伙伴/Tether未实现。吞吐与显存不等于接触能力或学习效果；本轮不改变历史基线、不自动启动新实验。


## Issue56：双轨正式研究合同与当前执行边界

用户已批准冻结BFM高层与独立可训练FB副本两条研究线，原共享BFM/PD不变。共同腕部任务、真实原生源、物理分项、父组隔离和单卡账本先统一；审批资源与计划不等于学习成功。该线的97D伙伴/接触输入已在旧函数保留与短程真实物理回归中验证，全部P1–P9已经用户明确批准；正式单卡串行队列已启动（先KL匹配对照，再1M FB），初始评价与工程验证不等于学习成功。详见[获批科学配置](DUAL_TRACK_SCIENTIFIC_DECISIONS.md)。共同gamma按物理时间对齐，base原FB/CPR目标与contact辅助奖励仍须区分，不把纯状态SF价值当完整接触return。原草案、历史BC500与负例完整保留。最新执行事实只看CURRENT_STATE/Issue56，不将本段或已批准1M写成训练完成。


### Issue56：双轨有限结论与证据纪律

冻结线KL .02/.08完整对照未产生符合预声明保护的新基线；FB接触扩展在真实采集/更新后因质量早停密集而按用户授权结束，不能据短跑否定整个方法。族中位数变化与逐条件配对变化应同时报告，任务梯度、关键访问覆盖与闭环放大不能仅凭聚合分数互相替代。固定点梯度和训练域近邻不等于真实历史更新或特定开发状态因果证据。未来完整存盘必须覆盖用户提前终止，而不能只有预定大节点；本轮缺失的FB末轮权重不得用初始化或重训伪造。下一轮计划不自动发车，详见[本轮结项](DUAL_TRACK_RESULTS.md)。


### Issue57：闭环保旧与训练恢复的不同层级
微小同状态动作变化可在接触前经真实历史和BFM本体反馈传播；静态动作误差小不保证保旧。
完整前缀接管须检查恢复域及质量代价，跟随改善但滑移加大的分支不能自动当可靠标签。
checkpoint保存、逐值读回、继续采样一致、继续更新一致是不同层级；当前最后一层未关闭。
这些是有限域证据，不宣称新理论。见D1_D2_RESULTS。


### Issue57续：算子定位、原生能力与均值诊断分开

相同batch/RNG的完整FB更新追踪将首层梯度差定位到discriminator反向FP32累加顺序：
真实操作数的不同括号精确复现两条最终梯度SHA。定位不是修复，严格恢复门仍未通过，
不据微小数值差异宣称旧物理退化已得到解释。
无接触盘点支持小腰转、抬臂保持、下沉与恢复；前进motion可执行但速度/偏航/滑移质量
仍不等于低速接触来源资格，有限后退候选未找到不等于底座不可能。
原B公共均值影响reward聚合，但本组z没有完全坍缩；离线中心化增大角度分化不等于
物理收益，未用于实际控制或修改推断。三者分别见[本轮报告](FB_NATIVE_BIAS_RESULTS.md)。


## 2026-11-10最新执行决策：Issue58

冻结线以真实on-policy RL获取standing/右腕近中距接触反馈，BC500仅作同起点初始化；A纯RL、B接触前函数保持，接触后不强行复制旧动作。物理搜索退出主训练数据生产，仅用于独立能力测量或少量抽验。FB线本阶段只修复更新恢复数值与踝交接/无接触保留，不启动正式1M或5M。旧MPC/交互蒸馏属于已完成历史证据，不是当前自动主线；后续Student须有新的明确授权。


### Issue58：函数保持、有限RL收益与晋级分别评价

接触前函数约束显著抑制了输出和真实前缀漂移，B在中距/停止出现真实开发收益；质量尾部仍使两臂按原规则选初始化。不能把未晋级解释为PPO零能力，也不能把函数保持当全任务行为学习已解决。完整时序反馈、任务间取舍、探索和信用仍须分开归因。

冻结策略采样的4进程热吞吐1.686×具有完整轨迹逐值对照；冷启动更慢，不能当完整训练FPS或直接乘其他优化倍数。批量BFM影子有微小FP32差，实际仍canonical row。FB更新一致、物理轨迹一致、完整portable状态一致是不同资格层；本轮最后一层仍未关闭，保留失败而非降容差。详见ISSUE58_EXECUTION_RESULTS。


### Issue59：风险发生比例与严重度不可互相替代

固定滑移事件成本在一个配对seed上减少部分超限父组，但末轮相对旧B仍有更重的严重度尾部与旧停止回归损失。初始化收益、相对旧方法增量、预声明科学候选与正式晋级须分开。原加载滑移指标同时含接触瞬态与持续运动，解释机制须看真实法向力/接触部位/积分时序，不通过抬阈值去掉坏数据。有限同状态有效方向并不保证PPO可稳定获得全族收益。见[Issue59结项](FRL_V3_RESULTS.md)；本阶段已停止，不自动启动下一轮。


### Issue60 F1：有限能力、定价、信息权限分别测量

F1以冻结合法z连续物理能力及C-S/C-P/O-F信息权限分层为诊断路线，不自动延长高层PPO。搜索选择的信息权限取候选生成、状态初始化、伙伴预测、评分与选择中最强者；原场景未来回放不得冒称因果受限规划。L0经逐数组C指针和156条3秒干预证据，修复了NULL指针对应Python临时数组的比较错误；27状态有效恢复资格与测速完成，未改变科学质量门。该工程通过不代表接触能力或学习成功。C-S planner仍缺格；详见F1_PREFLIGHT与F1_L0_ARRAY_AUDIT。用户随后授权L1/L2连续执行；L1/L2已完成并停于正式gate：有限状态存在合格连续控制，因果重规划也有部分任务收益，但尾部仍有明显滑移/偏航代价；不能把有限搜索失败当全空间不可达。候选评分的有限窗口最优不等于物理质量同时合格，也不等于共享策略已学会。信息合同见F1_L1_L2_METHODS，实际结果见F1_L2_GATE_REPORT；候选/时域不同使未来依赖仍未识别。L3仍必须另行用户确认，不自动扩大训练。


### Issue60 closed → Issue61 Steering-first (2026-11-12)

F1 matched future-oracle comparison improves stop and part of continuation but leaves tail quality unchanged; it neither proves pure sensing insufficiency nor makes Contact Adaptation mandatory. F1 is formally closed; [final evidence](F1_RESULTS.md).

Next decomposition is explicit Goal / causal Guidance / executable backend. Original BFM has no native ERI; Guidance→legal latent must earn physical qualification. P-BFM offers native velocity/heading semantics but different depth/PD/physics; I-BFM LOGO actor has a transitive current-B/privileged-object dependency. These interfaces cannot be called interchangeable or deployable force-only by inspection. S0 original archived-action parity passed; new root control remains a hypothesis. [Audit](S0_PORTABILITY_AND_ARCHITECTURE.md), [method decision](STEERING_METHOD_DECISION.md).

S1 remains a proposed bounded matched physical test, not started. Only after actual Steering/control evidence should learned Guidance or targeted Contact Adaptation be separately authorized. Native sources are traceable priors, including offline reference edits; no claim of prior-free behavior discovery. Historical headings above are stage records, superseded for current execution by CURRENT_STATE.

## 2026-11-12 S1 evidence

[Original progress-interface test](S1_RESULTS.md) finds causal stop-hold benefit but not reliable shared velocity control; near/sustained quality regresses. This neither refutes frozen BFM nor proves Contact Adaptation necessary. Native P-BFM ERI qualification is the next proposal, not started. [Budget erratum](S1_ERRATUM.md) retained.

## P-BFM ERI bounded evidence,2026-11-12

[One native preset/export qualification](PBFM_ERI_RESULTS.md) confirms real command input/action sensitivity but not reliable zero/low-speed stopping on the local port. Native ERI existence is not a blanket execution guarantee. Distinguish preset intent, command coupling and backend portability before attributing failure to contact or starting Contact Adaptation. No training or contact extension followed this failed gate.


## Issue62: Interaction-Steerable FB research hypothesis (2026-11-12)

R0 specifies separate frozen robot representation and new interaction-successor representation, with deployable local Guidance and public long-term Goal. T0 qualified a small six-network real-physics update/recovery loop; it did not establish useful factorization or behavior. A frozen executor+Guidance, B execution adaptation, C added interaction FB must be matched before scientific comparison. Accept B if C adds no independently supported value. Wrist bilateral tether is the first mechanism; rigid torso force couple is design-only, not cross-mechanism evidence. See [R0/T0 results](R1_T0_TRIAL_RESULTS.md). Pilot remains subject to separate user approval.

### Issue62 Pilot lock revision (2026-11-12)

T0 remains engineering-qualified, not behavior-qualified. A/B execution paths and independent evaluator are now measured; an exact-recoverable incremental checkpoint replaces quadratic full-snapshot output growth in the proposed Pilot only. The rejected3k/arm draft remains archived. [Revised lock](PILOT_BUDGET_AND_SPLIT_LOCK.md) proposes32k/arm and whole-parent train/dev/sealed manifests under8h/8GiB, awaiting Yi approval. First read is dev directionality plus an initial interaction-B probe; formal§5.1, factorization causality, convergence and generalization remain unproven. [Preparation evidence](INTERACTION_PILOT_PREPARATION_RESULTS.md) records common initialization/capacity differences and all engineering failures; noPilot launched.


### Issue62 Envelope数值预检边界

Nominal-anchored执行适配的物理设计与其有限精度实现须分别验证。原FP32动作接口不一定能精确表示单独在关节空间舍入的投影目标；本次最小反例导致按授权暂停，不能称物理方法已失败，也不能用近似逆冒充逐值通过。尚未开启新B/C或改变科学判断，见[预检](ADAPTER_ENVELOPE_V1_PREFLIGHT_RESULTS.md)。


### Issue62 execution representation clarification
Nominal-anchored adapter Envelope is a constrained execution adaptation package, not a force-to-skill rule. Its deterministic inward FP32 representation uses actual executed action and original g(action) consistently for physics/replay/critic/target. This does not revise physical thresholds or prove behavioural qualification. A remains Guidance-only; C versus B shares this Envelope. Numerical-only fixes ≤1e-6rad that preserve criterion relationships may proceed with explicit milestone disclosure under Yi approval.


Issue62 Envelope V1 frozen qualification: inward numeric representation succeeded; B19 sampled release-state actual joint excess became0, but loadedslip peak/p95 violated prelocked off-relative criteria. Constraining targets does not by itself establish coordinated recovery or qualify adaptation learning. B/C retraining and five-waydev remain unexecuted; no frozen baseline promotion. Full gradient-path audit is separately missing due a documented harness no_grad defect. See ADAPTER_ENVELOPE_V1_RESULTS/ERRATUM.


### Issue62 read-only retention mechanism audit
Current native CPR preserves a joint Guidance/action function on native prefixes; it does not enforce same-current-z adapter return on visited recovery states. Existing matched release traces and exact replay-draw counts support a narrow domain-specific retention hypothesis, with Q extrapolation and recovery-domain shift still competing explanations. A pre-Envelope raw-correction Return pilot is review-only; V1 qualification remains failed. No new physics or training was performed. See ADAPTER_RETURN_AUDIT_SUMMARY.md and ADAPTER_RETURN_TO_PRIOR_PILOT_DRAFT.md.

## Issue62 Return机制小试验（2026-11-13授权）

Yi已批准一次有限B-only adapter Return-to-Prior隔离试验：相同B19起点，冻结U/Q/discriminator及targets，仅比较64步原adapter目标与固定.0008同状态同z raw Return。先验证独立恢复域，不能以稀薄inactive帧推定可恢复；物理判断仍按原门，不修改失败结论。批准不等于已启动：GPU0当前被其他项目占用，CPU准备通过，新物理/优化仍0。参见[批准协议](ADAPTER_RETURN_TO_PRIOR_PILOT_PROTOCOL.md)。不自动恢复原B/C，也不把这个机制小试验称为新的通用训练配方。

## Return pilot结果边界（2026-11-14）

同z raw Return在两个恢复参考父组上明显降训练代理，但两组各64步adapter-only更新未保住另外两个原本稳定的恢复域；不能把原生参数冻结、较低loss或逐关节Envelope等同全身行为保持。保留按物理结果判定的原则，结束本固定配方；不自动加强Return/clip或重启B/C。下一机制重点是固定Q驱动的函数更新、覆盖与真实闭环后果之间的错配，尚未证明唯一根因。[完整负结果](ADAPTER_RETURN_PILOT_RESULTS.md)。


## Issue63：真实后继、执行适配与交互表征分别验证

D0档案审计确认固定Q绝对值未校准、投影后的执行变化与raw代理改善脱节、有限恢复参考未保住独立已暴露恢复域。有限窗口不同策略成绩不是无限时域Q真值；同状态首动作与共同后续策略仍需独立物理合同。不能把原生权重冻结或小修正视为行为保持，也不能用此负例否定所有离线学习。

下一设计优先真实rollout下的原Actor部分网络适配，保留B_R/运动先验、同状态同z有效域保持与在线价值校准；在相同执行基线上比较新增B_I/F_I/Goal Composition的独立价值。旧B19 Adapter-only配方已结束，R2设计不是新训练授权；当前只读交付见[R2_CAUSAL_TRIAGE](R2_CAUSAL_TRIAGE.md)与[R2_INTERACTION_FB_DESIGN](R2_INTERACTION_FB_DESIGN.md)。

## Issue63 R2执行授权（2026-11-14）

D1首动作匹配反事实已完成：单次Control/Return偏移由共同V1可恢复，严重失效与连续新反馈相容。R2现已获批启动真实rollout部分Actor适配/在线Q校准的E基线与BI/FI的I对照；不恢复旧29D Adapter-only修补。阶段以CURRENT_STATE与独立账本为准，工程检查不冒充物理学习收益。


2026-11-14 Issue63补充：D1共同续跑支持完整反馈累积的作用；R2部分Actor联合适配工程/恢复通过，但继承A128 U的未训练初始函数严重退化，正式学习未启。详见[R2工程与初始化](R2_ENGINEERING_AND_INITIALIZATION_RESULTS.md)，不能将此记成新架构学习负结论。


2026-11-14零U冻结干预补充：已暴露五父组中，去除继承A128 U可消除严重旋转/滑移，支持该Guidance造成当前域退化；原生standing在持续引导中仍有明显跟随缺口。稳定运动底座、共享反馈获得与训练泛化需分开判读。见[R2_ZERO_U_RESULTS](R2_ZERO_U_RESULTS.md)，未启动正式学习。


## R2 E/I v1.0新授权与采样边界（2026-11-14）

Yi明确批准新鲜零U、部分Actor在线适配、fresh双Q与I交互FB的匹配合同，最多32k真实transition/臂、R2累计16h/12GiB。原零U诊断只清A128末层；本次所有U隐藏层从单一预锁seed重新构造，不能混淆身份。

10父组冻结mean保持资格通过，但首个实际采样none回合触发5ms累计绝对yaw保护门；该量39.32°对应净偏转−3.53°与范围3.73°，不能称持续转圈或新模型学习失败。Q/Actor尚未更新，门保持不变，暂停后续pipeline。累计摆动、净偏航、滑移与任务进展分别评价，避免以一种量替代另一种物理含义。见[R2 E/I入口结果](R2_EI_ENTRY_RESULTS.md)。


2026-11-14：Yi明确授权签署R2 E/I v1.1，yaw保护改为真实pelvis连续展开后相对窗口起点最大偏离，TV独立保留。恢复需真实接触解除、同完整状态原生参考；none末窗不是释放。签前审查明确区分历史证据、synthetic拒绝fixture、未来Q/BI资格，不把签署当成全部实测通过。见R2_EI_V11_SIGNOFF.md。
