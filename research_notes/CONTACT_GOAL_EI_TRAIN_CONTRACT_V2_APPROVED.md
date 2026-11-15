# Contact-Goal E/I v2.0 — 接触条件执行适配与有限交互后继的匹配训练合同（修订稿）

**DRAFT / REVIEW ONLY / 待 Yi 审阅。2026-11-15。没有签署，没有启动资格物理或训练。**

本稿承接 Issue #66 审查评论 [6889896067](https://github.com/GaoE05/robot-research-bridge/issues/66#issuecomment-6889896067)，替代下一阶段设计建议，不替代任何已签历史合同或改判 H0。旧 [CONTACT_GOAL_EI_NEXT_CONTRACT_DRAFT.md @4c75ea7](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research_notes/CONTACT_GOAL_EI_NEXT_CONTRACT_DRAFT.md) 保留原样。

## 签前摘要：这份稿需要批准什么

1. **明确的新 Goal reward**：FOLLOW 沿用原跟随任务成本；HOLD 是公开指定的固定腕位置目标，替换与 HOLD 冲突的 anchor tracking/relative-velocity/heading 项，保留真实力、滑移、连续性等公共成本。它不是“原 reward 完全不变”。权重与测量窗见 §2。
2. **一个明确的感知假设**：理想力＋机器人自身里程计/运动学用于公开 HOLD 目标的相对位置；不是伙伴定位、active 真值或未来信息。自身里程计是本稿明示的仿真假设，不能称当前 G1 已具备；若不接受该假设，须改稿，不能签后悄悄换成 torso-relative HOLD。见 §2.1、§3。
3. **离散合法命令＋直接接触条件 Actor**：首轮不训练自由 256D U。高层四选一，低层原 Actor 最后一个 ResidualBlock/输出层与零影响接触 FiLM 同等适配。两臂执行能力相同，接触不只经过 z 间接进入 Actor。见 §3。
4. **先低层、再联合的两频率 PPO**：20ms 原始 29D Gaussian 与 100ms categorical 命令分别记概率，联合期按真正条件联合概率更新。旧 clipped-noise .01/.06 不是本合同 PPO 分布。见 §4。
5. **完整事件评价**：实际跨过停止、保持、释放以及 4 秒真实恢复，且含 arm-ready 源；不再把“任务名 stop”或 none 的尾窗当停止/释放。见 §5。
6. **固定终点与预算**：每臂最多 32,000 条真实 50Hz transitions，阶段 P2 为 12,000、P3 为 20,000；单 seed，终点模型，无按 Dev 反选。完整方案保守分账 **41,400 job 秒**，超过当前 R2 余额 **4,937.928257 秒**。本稿提前申请**新阶段 6,000 job 秒**，而非自批扩大旧 R2；未批准该新增额度不得开正式训练。见 §8。
7. **严格同伙伴的初始化规则**：公开Goal/策略分叉后不能分别把anchor对齐各自腕部。本稿明示用共同原生参考前缀的onset腕位置锁一个anchor，再恢复共同handoff状态给各方法/Goal执行；这是拟议的配对初始化科学约定，不称旧实现已经如此，也不在本次改代码。见§5.1。

以上是一个整体待审合同。**只批准文档不等于批准物理与训练；没有签署，不排队开跑。** 封存、Student、额外 seed、正式 FB 1M、额外接触部位、反应型伙伴均不在范围。

## §0 事实、证据边界与唯一问题

事实基准 `gaoe05/research @4c75ea74bd877e76853e3c49debf4673b40d2e1e`：

- H0 P-E 4/4、P-I 3/4 局部效应原判保持；E 四组均 HOLD→standing、FOLLOW→arm15，零学习 Goal-only 查表可复现其已执行行为，状态依赖能力未证。
- 名为 stop 的 H0 父组实际窗 5.9–7.9s，stop_time=8.3s，**没有经历实际停止**。未测真实释放恢复；K40 mid 的 FOLLOW 误差仍 9.64cm。
- H0 20ms q_PD 峰跳约 12.14–12.17°、真实关节速度峰约 3.7rad/s。它们不是相同量，硬 q 安全通过不代表平顺性或真机资格。
- H0 24 条实际 fit 后果、4 个独立评价祖先只支撑有限小样本结论。I 相对 E 的行为增量未证，不能借 B rank、训练 MSE 或菜单差异补分。
- 当前旧实现 `research/interaction_fb/agent.py:Agent.action` 的原 BFM `_actor(normalized robot(obs),z)` 没有直接接触字段。接触通过 `deploy(obs)` 进入 U/旧 adapter；`schema.robot` 明确排除接触扩展。**本稿的新 FiLM 是待实现的科学接口，不能写成现有 Actor 已有此能力。**
- 原力、跟随、滑移等实际优化成本应以 `research_notes/dual_track_task.py:native_terms/Task.step` 的 20ms 重算为准，不能把 `clean_env.py` 或 `geometry_env.py` 中被覆盖的旧记录当实际训练 reward。

问题：在有真实接触、公开 Goal 和完整时序后果的有限腕部域中，**同等执行适配能否超越零参数 Goal 查表；I 的有限交互后继是否比普通 Goal/outcome 基线提供额外物理价值？** 不宣称全行为学习、通用 FB 或自主人类意图识别。

## §1 两臂、初始化与不变项

| 项目 | E：接触条件执行基线 | I：有限 Interaction-successor |
|---|---|---|
|原始权重|同一原 BFM，SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`|相同|
|冻结|B_R/F_R、原 Actor embedding 与前五个 Block、原 robot normalizer、PD|相同|
|执行适配|最后一个 Actor ResidualBlock＋输出层＋§3 接触 FiLM|相同可训练层、初始化与容量|
|高层|同四命令 categorical selector，无自由 256D U|相同|
|Goal|普通 24→32→16 Goal encoder|同初始函数的 B_I Goal encoder|
|普通/交互预测|0.5/1/2s 原始标准化 24D 后果预测|F_I 预测 B_I 后果＋同尺度 decoder|
|训练 Actor 的信号|共同真实 on-policy PPO、同保持项|相同；不加入 SF/Q 动作梯度|
|额外表示信号|原始后果 MSE|编码后果 MSE＋重建＋固定 variance/covariance 正则|
|执行预算|12k 低层＋20k 联合，32k 真实 50Hz 条目|相同|

全新种子固定 **660301**；E/I 的共同模块逐张量相同，独立 RNG 以同 seed 起步，分支后各自真实演化，不强行伪造匹配状态。Adam、value critic、训练 buffers 全部 fresh。原 Actor tail 从原 BFM 复制。FiLM 末层输出 γ=β=0，隐藏层正常初始化；初始 `a_new=a_original` 必须逐值验证。不得加载旧 A128/B19/Return、I/C0/T30、BC500 或 H0 head 的学习参数。BC500 仅冻结参考。

fresh state-value critic固定为256/256 Tanh→1，E/I同初始化；输入为相同的当前C-P（本体、当前伙伴误差/速度、真实active及公开Goal），不读取未来、完整任务轨迹或I独享表征。它可有训练特权，但不经隐藏特征进入Actor。原生reference仅用于已资格保持，不能覆盖执行后伪装on-policy。

E/I 的公共模块参数量须相同；包含各自表示支路后的总可训练参数相差≤1%，不得用不参与 loss 的 dummy 参数凑数。具体固定层形状、总量、梯度图在 P0 manifest 中展开；若本稿给定结构无法满足≤1%，**停下改稿**，不能按 Dev 调网络。I 多出来的表示计算单独记成本；相同决策预算不叫相同计算成本。

表示执行路径也预锁：两臂F/Predictor均以当前合法h、当前公开Goal值/mask/mode、候选option one-hot与**当前第一插值z**为输入，128/128 SiLU；E输出3×24原始结果，I输出3×16 embedding。低层读取当前已选option的预测上下文：E将各horizon预测经普通Goal encoder后取16D均值，I对三个F_I输出取16D均值；与公开Goal的16D编码一并进入公共FiLM条件。高层在真实采样边界对四个option各算该因果上下文后送同结构selector（128/128→4），不调用物理搜索。未采样的未来动作、伙伴未来、实际未来y均不作为当前Predictor输入。

I的decoder固定16→32→24，E直接预测24D；预测监督是**当前策略后继条件下的有限结果**，不把当前一次action与未来2s开放动作等同。公共Goal encoder参数位置/形状相同；I多的decoder和较窄F输出造成的小容量差异如实计，仍需总量≤1%。未来真值仅更新有明确权限的训练支路，在线上下文始终是当前因果输入产生的预测。

原 29D action→q_PD、50Hz BFM、200Hz MuJoCo、10N cap、D8、腕部接触、真实 q 超限 .02rad 护栏均不变。K80 与批准的新 K40 分层，不能更改 K 来挽救某一臂。力、reward、观测均不随 cap/K 静默重归一化。

## §2 公开 Goal、实际 reward 与停止/释放语义

### §2.1 Goal 是公开指令，不是隐藏任务类别

- `FOLLOW`：响应当前真实引导，持续减轻接触负担并有效跟随。它不含最终距离、伙伴真值速度、未来停止/释放时刻。
- `HOLD`：从**公开指令实际发出时刻**起，保持当时公开指定的腕位置 `p_g`。它在源时机器人自体里程计坐标中固定，不随 anchor 或每步腕位置重置。不是让机器人违背平衡、锁腿或复制 standing。
- 在线 Goal 字段为 mode one-hot、当前 Goal 的因果目标相对位置、有效性（有效性只指 Goal 指令，不是 contact active）。HOLD 的相对目标通过**自身运动学＋自身里程计**转换到当前 pelvis frame；FOLLOW 的方向只由当前测得的力获得，力小则方向置零，不能偷偷使用 partner velocity。
- 自身体位信号必须单列来源：仿真可按理想自身里程计实现，但必须与 anchor、伙伴状态分开。此假设只用于自身定位，未作实机资格，不得伪称原 1639D 已包含它。若后续要删里程计，是另一观测消融，不能在本轮中途改语义。
- 同一伙伴实际轨迹配多个 Goal；相同 Goal 配多个伙伴轨迹。公开 Goal 的变更按预锁独立指令流给出，不由“已到停止时刻”的隐藏 phase 触发。Goal 变化是公开命令，不是推断了不可见意图。
- HOLD 在实际接触窗有位置任务；真正 release 后撤销该任务成本，评自然恢复，**不得继续把旧 anchor 或旧 HOLD 位置误差算主动跟随成功率**。release 不自动向 Actor 发阶段 bit 或新 Goal token，Actor 只能看自己的真实传感历史。

### §2.2 共同成本与有意新增的 Goal 分支

令 `c_old` 为锁定版本 `dual_track_task.native_terms` 产生的每 20ms 成本字典；保持原每项物理单位、cap/截断、连续性与加载滑移定义。`C_keep` 为其中除 `tracking/relative_velocity/heading` 外所有项：force、uncapped、slip、tilt、continuity、soft_prior、idle，原系数不动。真实终止吸收尾项沿用旧定义。

设物理实际接触指示 A 仅在环境计算 reward/mask 中可见，不入控制器；测量窗为每一真实 20ms 内实际 5ms 行。定义：

`C_follow = c_old.tracking + c_old.relative_velocity + c_old.heading`

`C_hold = 4 * mean_5ms(||p_wrist-p_g||² / 0.04²) + 0.2 * mean_5ms(||v_wrist||² / 0.10²)`

`r_20ms = -dt * [C_keep + A * (1_FOLLOW*C_follow + 1_HOLD*C_hold)] - original_absorbing_failure_penalty`

这里 `dt` 是实际经过时间，不固定假设每次都完整20ms。HOLD 新项无额外 reward clipping；FOLLOW 必须与原成本逐值一致。HOLD 允许在同一真实拉力下有任务取舍，不能再由 anchor tracking 惩罚它“没有向伙伴让步”。force/uncapped 等公共成本继续阻止为 HOLD 制造无限牵拉。4、0.2、4cm、0.10m/s 是**本稿待批准的一组固定科学选择**，不声称已由 H0 验证，不扫权重。

保留 5N 的历史回归读数，不追加本轮 5N 全矩阵或训练。主任务仍10N。none/zero 的 Goal/指令也要显式记录，不能用 task ID 输入回答。

### §2.3 Reward 入场门（只对实际依赖负责）

P1 用预锁 train-only 同状态/同伙伴配对，至少4独立父组、每种 Goal 至少2个，包含 active、真实停止、真实 release 后后果。固定 native continuation/四候选及有限同状态首动作后共同续跑；不是生产 Teacher 标签。一个物理上更好且质量不差的配对必须满足至少1cm误差或0.5N负担改善，或 HOLD 位置误差至少1cm改善。至少8个足够差异的有效pair、按父组等权的reward同向比例≥80%，且**不存在**“持续近cap、不推进、质量不优却优于干净达成”的结构性反例。无足够 pair 则 UNDERPOWERED，不能说 reward 已正确。

本门失败只停新优化并报告目标/可辨识性证据；**不自动改权重或重跑找好例子**。它不是旧 Q 70% 门，无需训练 Q。成本计 P1，不另开审计项目。

## §3 实际 contact→Actor 路径与合法命令

### §3.1 明确数据流

`机器人本体/真实已执行 action、z 历史 + 当前/过去 ideal-force + 已知安装 + 公开 Goal`

`→ contact encoder C(h,g,同权限预测上下文) → 最后一个原 Actor Block 的 FiLM → 可训练 tail/output → 原实际29D映射 → 原PD`

并行高层：`同权限 h,g → categorical option → 有身份合法z生成器 → 原五子步球面插值 → Actor`。

FiLM 使用 `v'=(1+γ(C))*v+β(C)`，末层零初始化，不给 Actor 额外29D旁路，不接旧 adapter/Envelope。接触 encoder 是 128/128、SiLU，共用容量；只调制最后一个原 Block 输入，具体该 Block 宽度以原 checkpoint 为准。既有 robot normalizer 原样冻结；新增力/安装/Goal字段固定物理尺度，不让运行统计学到任务时钟。Actor 每20ms读取最新实际传感历史；不能把100ms起点的力复制五次充当实时反馈。

`contact_schema.json` 必须逐字段列出 producer、原始索引、frame、单位、scale、更新时间、可缺失状态、允许角色及来源。力历史固定沿用已有历史形式/长度；不展开0/1/2/4/8/16网格。安装 mask 只表示已知安装，不随真实active改变。自身里程计假设与传感未知逐项标明。

两项不同证据必须都交：

1. **权限/连通资格**：变动隐藏 anchor/future/active/clock 等而保持合法输入固定，Actor/selector/Goal encoder 不变；固定 robot 与同插值 z，改变当前真实可见力时，FiLM 路径有可导连通。初始零影响时动作可以不变，但末层梯度必须存在。
2. **训练后的实际使用**：在真实访问状态，同状态同z的 contact-history disable/shuffle，比较真实29D/q_PD/物理效应。输出变了不代表更好；必须与实际结果比较。若只有 selector 变而低层不使用接触，报告“高层接触调用”，不称 direct contact-conditioned Actor 学成。

### §3.2 四命令及相位

原生 standing、goal_arm5、goal_arm15、已归档 motion 三种原生来源中的四命令身份固定，原资产与 H0一致，不下载/扩库/接 I/C0 输出。每100ms选择一条当前原生目标；motion 的 phase 是策略自己的持续命令记忆，按固定rate .5真实推进。第一次进入时在原8-grid选与**自己的真实last-z**最近相位；暂停/退出后不借其他轨迹phase。其他命令不修改机器人身体历史。

每个新100ms目标从实际last-z按原5个50Hz子步球面插值；command完整描述、目标z、每子步实际插值z分别保存。episode source由真正原生前缀生成，solver/观测/action/last-z/force/selector自身记忆完整保留。

本轮菜单是降低首轮探索维度的研究限制；低层tail可在同合法动作合同内适配，所以不是单纯四动作播放器。**不得把结果称任意256D控制或永久四技能解。** 下一自由latent合同另签，本轮不夹带。

## §4 两频率采样、学习与更新合同

### §4.1 原始随机变量与执行映射

低层每20ms恰采一次 `u_k ~ N(mu_theta(o_k,z_k,g_k), 0.01² I_29)`；存未截断 u。实际执行 `a_exec,k = original_action_map(clamp(u_k,-1,1))`，再用原q_PD映射。这里 original_action_map 的全部旧动作缩放/限幅需展开入 manifest，不能省略二次clip。不要把未经执行的 u 叫实际action；两者及 q_PD 都存。

**没有额外 relative-noise clip=.06，没有 Envelope，没有对单关节改边界。** 固定 many-to-one 执行映射下，可按实际采样的 u 做增广动作空间 score-function PPO；不能拿 clamp 后 a 算普通未截断 Gaussian logprob。std=.01是新合同的 raw Gaussian尺度，沿用数值不代表沿用了旧off-policy概率。

P2：高层固定 Goal-only（HOLD→standing、FOLLOW→arm15），无高层随机采样/更新，只有低层Gaussian。

P3：保持100ms命令控制，但**每个option至少持续0.5s**：每5个高层槽的首槽采 `j_t~Categorical(pi_eta(h_t,g_t,m_t))`，m只含真实自身option/age记忆；其余四槽确定性延续，option logprob为0。所有四命令均可选，无隐藏phase/action mask；每20ms按已采j与实际z/身体历史采一个低层 u。selector输出层初始全零、hidden fresh，并加**两臂共同固定公开Goal先验logit**：按standing/arm5/arm15/motion顺序，HOLD先验(.97,.01,.01,.01)，FOLLOW先验(.10,.10,.70,.10)，`pi=softmax(log(prior)+learned_logits)`。这些是待签的探索初始化，不是force阈值规则或永久半径：全部option始终正概率，learned_logits无幅度约束，可覆盖先验；不把高斯raw均值与option概率混成同一std。P2期间selector不训练，P3切换必须重新做采样安全和概率资格。Goal变更不偷开额外采样机会，作用于下一个预定边界的分布。这里是10Hz命令控制、至多2Hz option随机采样、50Hz低层动作，**不能把保持中的同一option记作五次独立采样**。它不是“沿用Goal查表分布而没有改变探索”。

一个完整100ms样本的真实联合概率：

`L_t = b_t*log pi_eta(j_t|h_t,g_t,m_t) + Σ(k=0..4) log N(u_tk; mu_theta(o_tk,z_tk,g_tk), .01²I)`，其中b_t仅在真实option采样边界为1，其余为0。

P2省去高层项；实际提前终止只计已执行子步。重新算 new logprob 时用**保存的真实历史、真实option、实际z、已执行此前u/action**，不能由新策略重跑另一条z轨迹。跨Goal变更只在100ms边界，自己的采样相位/记忆逐项恢复。

### §4.2 PPO与时间合同

每1024条新50Hz transition收一块（最后块按固定终点可短），按最多205个100ms决策段打包；两个epoch、minibatch64个决策段、clip ratio .2、joint KL上限 .02、Adam actor/selector lr=1e-5、value lr=3e-4、max_grad_norm=1。两臂同合同。selector与低层用同一联合ratio/advantage更新，不分别做5次PPO把相关子步当独立高层样本。accepted/rejected步数、KL、clipfraction、raw/z/q_PD实际变化逐批存。

以上优化节奏是**本稿新预注册选择**，不是从旧off-policy代码默认“自动继承”。联合KL须同时报告categorical项、五子步Gaussian项和实际joint估计，拒步回滚必须恢复共同Actor/selector/Adam完整事务，不能只回滚参数。

缺省也锁明：Adam betas=(.9,.999)、eps=1e-8、weight_decay=0；policy/value max_grad_norm均1；categorical entropy系数0，Gaussian std固定而非学习；value用未clip的MSE、同两epoch/minibatch；不设另一个未登记value-only warm-up。advantage在本新rollout块内按实际决策段mean/std标准化（eps=1e-8），绝不与旧replay混合。每次step后用**整块保存的真实条件历史**检查KL；以非负的categorical解析KL＋实际访问五子步Gaussian解析KL之和为拒步量，同时存sample joint old-new logprob估计。后者不是精确全轨迹KL，不能用其负噪声抵消前者。概率断言继续使用已有严格容差，不为通过放宽。

`gamma_20=.99^(1/5)`、`gamma_100=.99`，`lambda_GAE,20=.95^(1/5)`，100ms等效.95。100ms实际优化奖励为 `R_t=Σ gamma_20^k r_tk`，实际duration bootstrap；完整五步等效折扣.99。真正terminated不bootstrap；行政块边界/time-limit截断用合法最终状态bootstrap，跨reset不串接；release/Goal变化/stop均不当终止。吸收尾项不重复算。新fresh V仅作GAE基线，不复活旧动作Q或引入Q排序门。

评价奖励另报未折扣物理时间积分与按秒负担；不同回合时长不得只比累计return。

### §4.3 分阶段解冻与固定训练规模

- **P2低层期：12,000 transitions/臂。** selector冻结在Goal-only；同等contact FiLM＋Actor tail训练。至少覆盖8独立train父组，含none/zero/near/mid/stop-continue与真实释放恢复，不以随机抽样概率充实际覆盖。
- **P3联合期：20,000 transitions/臂。** P2最终模型继续，按§4.1开始真实categorical采样，低层和selector联合训练。总32k/臂是本稿完整拟议规模，不是旧I/E恢复训练。两臂相同终点，无“看起来好再给I加步”。
- P2→P3只看预锁train资格：概率/恢复/预算、保持物理与数据覆盖合格；**不要求P2已解决任务或V EV达到数值**，避免把有待高层探索的问题再变永久门。若Actor出现硬安全/保持失败则按门表停。新categorical option采样资格失败也不能自动调先验/logit/std。
- 训练中固定检查点：0、4k、8k、12k、20k、28k、32k；只用train sentinel做安全检查，不用Dev改系数。主比较是32k终点；科学失败提前停的臂不以后续另一臂终点冒充等步数比较，另给最近共同节点。
- train sentinel固定为一个none、一个zero、一个真正释放后已资格完整状态（各取资格表稳定顺序首个，三者不同父组）。每臂检查首个P2更新、4k、8k、12k、首个P3更新、20k、28k、32k，共8节点×3＝24回合/臂、两臂最多48。none/zero跑完整online窗，release状态跑完整4s；成本/ticks/存盘计P2/P3，不藏进“评价另算”。真正采集中的硬界仍逐tick检查，不等sentinel节点。恢复同态冻结参考从已有合格完整状态复用，无同态参考不得填PASS。
- actor/selector只使用本策略真实on-policy样本；失败与前缀/ticks记成本，但原生前缀不冒充RL样本。原生保持样本仅作标记清楚的辅助项，不放入on-policy概率损失。

### §4.4 保持与表示（不暗开另一配方）

两臂同状态同z保护已资格native/none/zero/recovery，不能在active推进域复制standing。沿用事前公式的 q_PD eps 标定与固定对偶：`lambda←clip(lambda+.05*(父组等权p90 normalized_violation−1),0,3)`，初始0，每新1024条块后更新一次；用全部锁定资格父组，不以密帧权重替代父组。若上限3仍连续两个固定检查点违反则停臂，不能加λ。已签源公式的边界和旧物理保护见门表，不从单失败梯度重标系数。

标定公式明确为 `eps_rms=clamp(.25*parent_median(p90 native20ms qPD_step_RMS),.015,.050)rad`，`eps_p95=clamp(.50*parent_median(p95 native_single_joint_step),.030,.100)rad`。同状态同z的Δq_PD：逐帧`v=max(RMS(Δq_PD)/eps_rms,max_joint_abs(Δq_PD)/eps_p95)`，父组内取p90，再父组等权求均值进入上述对偶。保持loss为合格mask上的父组等权mean(v²)，乘本轮λ；只有已资格none/zero/precontact/真实恢复/原生保持记录进入，真实active即使力暂小也不能自动当保持域。mask仅训练期见，不能进入Actor。该loss直接约束低层与影响低层的共同条件编码，**同z保持不能证明selector不会选错z**；selector由joint KL与真实none/zero/recovery物理门保护，不另加未锁定U系数。

λ在每块完整PPO/辅助更新结束后由更新后的锁定资格集计算一次，供**下一块**使用，本块全部minibatch固定λ；第一块λ=0。完整目标为`L_policy=-PPO_clipped_surrogate + lambda*L_keep`，另四组.1辅助loss与独立value MSE；不把未来真值的辅助梯度传入Actor tail。额外输入采用原h固定尺度（力仍除5N，不因10N cap更换）；目标24D沿用H0固定SCALES，mode/mask为0/1。没有未声明的reward、entropy、GAN或SF项。

表示模型只用当前臂实际后继。E预测0.5/1/2s标准化y原值；I预测B_I(y_future)，另有B_I重建和.1 variance/covariance正则。y以H0 24D可影响结果为起点，时间窗坐标/force均值与RMS严格区分。每个新1024条块，E/I均4组batch64的辅助更新，辅助梯度权重固定.1；PPO和辅助梯度分项存。I无SF Actor动作梯度、无标量Q、无CPR/GAN表征loss、不中心化原B_R。

**共享参数的优化器所有权**：policy、Goal encoder和所有表示头使用一套共同Adam（lr=1e-5），value独立Adam（lr=3e-4），禁止同一B_I/Goal参数被两个不同Adam重复管理。每次PPO或辅助step都对同一rollout参考检查真实joint KL并做完整事务回滚；辅助更新通过改变在线Goal编码影响策略，不能绕过KL。纯辅助step不向原Actor tail/selector反传未来真值梯度，只有明确共享Goal表示的参数接收辅助梯度，其他policy参数grad=None；其后再按保存观测复算实际policy变化。E/I更新组数、接受/回滚均存，不因I多一个loss额外给更新。

在线Goal编码的24D目标表明确为：FOLLOW的腕delta为当前测力方向×.10m，HOLD为自体里程计下公开p_g减当前腕位置；wrist-relative-root delta同该值，supported fraction目标1，其余未指定槽填0。**另将逐槽Goal-specified mask和mode作为公共条件输入**，未指定的root/足步/force值0是缺失占位，不是要求机器人不能移动；它们不生成额外reward。E普通Goal encoder与I B_I编码接收同一24D目标值，公共条件分支同样读mask/mode；未来y可全字段预测，但不能把未来真值当Goal。FiLM与selector初始不依赖这些随机编码而改变原动作，须G01验证。

这准确命名为**有限时域 Interaction-successor E/I pilot**，不是已完成全策略FB Bellman训练。未来结果只当训练监督，不进在线actor。尾窗无完整未来者按horizon mask，不用跨reset或复制末帧伪造2s标签。

I辅助数学固定为三horizon等权的 `MSE(D(F_I),y) + MSE(F_I,stopgrad(B_I(y))) + MSE(D(B_I(y)),y) + .1*(L_var+L_cov)`；`L_var=mean(max(0,1-std(B_I(y)))²)`，`L_cov=mean(offdiag(Cov(B_I(y)))²)`，不归一化到旧256D球面、不与z_R相加。E为三horizon等权`MSE(P_E,y)`。每项先按有效horizon/样本平均，再同一.1辅助权重；不把I多项loss冒称同一数学基线。辅助buffer仅含当前臂/当前P2或P3阶段最近8192条实际记录，按父组等权后组内均匀，跨阶段清空；PPO只用当前完整on-policy块，不能混用该辅助buffer。未来不足者用显式mask，不能为了凑batch借其他回合后继。

## §5 父组、真实事件覆盖与评价驱动

### §5.1 身份与样本单位

继续锁定34 train/44回合、16 dev/20回合、28 sealed/36回合。旧H0四组是已暴露train侧读数，原Dev若历史已执行需逐组标EXPOSED_DEV，不重新称新封存。**本pilot不读Sealed**。

Goal/K/source衍生都继承真实source+partner祖先，不能把不同hash、兄弟Goal、K40/K80或相邻帧当独立父组。train/dev/sealed需对祖先图检查，不只字符串去重。至少一个经F1原生资格的arm-ready源；本稿固定选择arm-ready，不把未资格walking强塞来凑移动覆盖。moving若现有资格不足，明确缺格，本轮不自动改source范围。

原父组伙伴距离/时序不改；K80/K40按父组均衡赋值，同stop/continue父组用同K，源配置在看结果前锁定。arm-ready是本轮明示的source衍生域，仍归其原父组，train/dev祖先不能交叉借用。若新增source后无法保有原16独立dev父组，资格为DATA_INSUFFICIENT，不从sealed补。

**伙伴初始anchor的共同真值**：`anchor_mode=onset_actual`若分别在不同策略/Goal的onset读取各自腕位置，会改变对方目标过程，不能称严格同伙伴。本稿改为每父组从同一完整原生handoff状态分支，原生参考连续执行到onset得到一次参考腕位置，作为该父组固定anchor初值；保存全参考与物理状态后恢复原handoff，所有方法/Goal/stop兄弟均使用相同anchor初值、路径与绝对时序。不得在它们自身onset重新对齐、随机器人追随anchor或提前断接。真实force仍每5ms根据各自实际腕状态闭环重算。这是**待签的明确初始化约定修订**，不变K/D/cap/路径形状，不追改旧onset-actual实验。原生参考前算只用于环境建任务，不进入Actor/selector/Predictor/Goal issuer输入，成本/ticks全计。

先生成 `episode_manifest`、`goal_stream_manifest`、`source_ancestry_manifest` 以及训练顺序种子 **660302**，在首次物理前commit/hash锁定。它们是本文固定范围的展开，不允许选择表现好的父组。每臂32k至少完成原44回合各一次；P0按真实在线时长先算能否覆盖，若下界超过32k，**签前/开训前报告规模矛盾**，不缩短4s恢复或跳过长回合。每阶段实际回合/命令/20ms/5ms/Goal/源/K/阶段计数分别交付。

Goal流展开规则固定：在各split内，各near/mid/stop族按原稳定顺序循环分配`FOLLOW / HOLD / FOLLOW→HOLD / HOLD→FOLLOW`，三族循环起点分别偏移0/1/2，none/zero用HOLD。双向Goal变化均在真实handoff后1.5s的100ms边界公开发生，**不取未来stop/release时刻**；同stop/continue兄弟用完全相同指令流。Goal流标签/未来切换时刻不进Actor；自身option年龄只是真实控制记忆。

K80/K40在**每个请求族内**按父组稳定顺序交替，stop兄弟同K，dev由此各8父组。source固定为每个near/mid/stop族的首个K80及首个K40父组使用arm-ready，其余standing（dev为6 arm-ready＋10 standing）；同父组各Goal/stop兄弟的source相同。这样不将“每4个取1个arm”和“每2个交替K”叠加成arm只出现在一个K。该设计仍非完整Goal×K×source因子网格，缺格如实报告。需检验每个新Goal active测量窗≥0.5s和§5.2计数；不足是manifest资格失败，不事后挑新父组。

### §5.2 必须实际经历的窗（用事件，不用名字）

|窗|真实起止与最低记录|不得替代|
|---|---|---|
|entry|handoff→actual onset，onset后至少1s|只存源的一帧qpos|
|sustained/loaded|实际牵引持续段；mid晚期/高负担至少1s，按实际力分层|用早期2s near称持续推进|
|stop/continue|同前缀到真实伙伴开始减速之前一致；完整跨减速结束，hold≥1.5s；continue同源同流继续|“family=stop”但窗口终点早于stop|
|Goal变化|同伙伴过程中FOLLOW→HOLD、HOLD→FOLLOW都实际执行；至少各4独立train父组，dev至少各2|把伙伴stop当公开Goal变化|
|release/recovery|先有真实施力/接触建立，再实际解除，后续完整4s；逐tick确认active/force|none脚本尾窗；仍有接触时称已释放|
|none/zero|无实际施力与连接零需求分别整段记录|force=0被当未来active答案|

传感零是测量，不是active标签。脚本release是外生事件，不计自主成功。恢复窗只评支撑/滑移/yaw/root路程/残余速度、足步及关节连续性；不把旧anchor几何误差并入接触RMS，也不把force=0混入active平均降低负担。

### §5.3 固定物理评价规模

固定终点后一次独立dev，模型/随机流/物理状态与训练进程隔离：

1. 16 dev父组/20回合×**E、I、0参数Goal-only查表、冻结原生零U**＝80完整回合。Goal流按父组预锁：包含固定FOLLOW、固定HOLD、双向中途Goal变化；不是看结果后给模型容易的Goal。K40/K80各8个父组；固定6个arm-ready父组，其余standing。near/mid/stop与两K均须有读数，缺单元写缺格不合并掩盖。
2. 上述中的4个父组加同伙伴、不同公开Goal的匹配分支，四方法共16完整回合，作为Goal干预对照。选择固定为：near首个K80 arm-ready、mid首个K40 arm-ready、mid首个K80 standing、stop首个K40 standing；同时覆盖两K×两source并含真实stop。不存在该单元则DATA_INSUFFICIENT，不按结果另挑。
3. BC500仅作FOLLOW冻结历史参照，固定4父组完整回合；它非Goal-conditioned，不伪称HOLD匹配方法。

**共100个完整回合上限，不能改成100个2s窗。** 固定回合均包括实际发生的停止/保持/释放事件；没有stop事件的continue不称stop读数。无touch回合按完整none，不冒称release。按已锁时长、完整source前缀、加载/IO估算成本，超预算先停申请，不砍掉恢复或失败条件。

§5.2高负担与真实释放分析还使用训练末轮的全量记录；不能只用100回合总均值。训练阶段检查只有train sentinel，不跨Dev选batch。

### §5.4 共同物理判读（Goal-aware，不互相偷换）

- 硬边界：真实q excess≤.02rad、真实支撑不丢失、输入/状态/数据正确。触线停相关臂，保留原失败。
- FOLLOW科学参考：near≤4cm/4N；mid≤8cm/7N；stop保持≤8cm/6.5N；真实有效推进、loaded slip≤.45m/s、末绝对yaw≤20°，原任务推进定义与加载语义不改。不能因HOLD任务定义不同而移动FOLLOW门。
- HOLD新增任务参考：active窗腕相对公开固定目标RMS≤4cm、p95≤8cm；实际力RMS≤7N；不得借cap持续顶住、失支撑或滑移维持位置。它不使用FOLLOW的anchor误差成功门，anchor误差仍如实附报。
- none/zero保持：loaded slip≤.10m/s；窗口yaw_excursion≤15°；root XY路程≤.10m；真实q/支撑门同上。
- 真release恢复：loaded slip≤.45m/s、yaw_excursion≤15°、不新增硬界/失支撑；root路程≤**同完整释放状态、同Goal冻结原生合格参考**＋.10m。缺合格同态参考为UNMEASURED；不同策略抵达release的不同状态不能伪称同态相对比较。不同状态仍可按绝对物理坐标报告。
- 平顺性单列：每关节20ms q_PD jump p95/max，真实q/dq/加速度、力矩/饱和、root/腕速度加速度。按同父组同窗基线比较；若q_PD jump p95或真实dq p95相对查表/零U中适用基线增加>25%，且绝对增加分别>.03rad/.5rad/s，记连续性退化，不以硬界PASS盖掉。峰值仍完整报，不说实际q瞬间跳了同角度。
- 同态contact shuffle/disable与Goal-only查表的实际差分检验任务改进，不能靠任意动作差就声称反馈有用。至少4独立dev父组形成可解释对照；若只有同一个Goal总选同命令且低层无有益修正，记“未超越菜单”。

loaded slip同时报真加载、轻载擦地、其他接触三分口径，记录法向力与部位，不改法向阈值降超限；p50/p90/p95/max与超限持续时间并列。yaw_net/range/excursion/TV分列；最终包角不替代中途大旋转。

## §6 阶段放行表：三列逐门，不由执行者临时裁量

**状态词**：PASS / FAIL / UNDERPOWERED / BLOCKED / UNMEASURED。当前所有新门均UNMEASURED；H0/历史例子只是校验量尺。签署允许未来按表执行，不是现在虚构PASS。

|防什么／门ID及放行对象|怎么算／固定证据与规则|现有证据、失败例与自动分流|
|---|---|---|
|G00 授权与资金，文档→P0|本文、表及参数锁全部签署；新6,000s与原余额明确核账；无Dev/Sealed执行授权串用|本稿REVIEW ONLY。未签或新增预算未批＝BLOCKED，不开P0物理/训练；文档审查可继续|
|G01 身份/祖先/覆盖，P0→P1|原BFM/PD/四命令SHA一致；两臂公共权重/初始动作逐值同；34/16/28祖先无交叉；按44完整回合算32k下界；所有Goal/K/source manifest锁定|H0资产及恢复有历史PASS；不等于本新接口PASS。冲突→DATA_INSUFFICIENT；不从sealed/另seed补数|
|G01-P 同伙伴而非各自追随，P1→更新/Dev|同父组不同Goal/模型的anchor初值、路径、K/D/cap及时序逐值同；只允许真实受力随各自身体改变；原生reference前算严格恢复并核不进入在线输入|本新共同初始化规则UNMEASURED。若各自onset重对齐或未来ref混入O_S，停止共享路径，不用“物理更自然”解释掉配对失效|
|G02 Goal与reward，P1→任何更新|§2实际reward三项单测：FOLLOW逐值旧同、HOLD只变已列项、release后任务项真为0；≥4train父组、有效pair同向≥80%、无结构性坏解更优|H0没有HOLD reward资格。历史旧reward偏跟随是本门动机；FAIL/证据不足停，不自动改系数|
|G03 信息/Actor接触，P0/P1→更新|schema字段来源逐项；隐藏量干预输出不变；真实接触→FiLM梯度连通；已执行前向/动作与记录一致|旧原Actor没有contact输入；schema仅许U/adapter，不能填新FiLM已通。泄漏共享缺陷→两臂停，隔离产物|
|G04 采样/概率，P1→P2及P2→P3|raw采样/执行/记录逐值；batch-one/batched原断言；初始ratio1；五条件概率和joint/clip/KL正确；GAE终止/截断fixture；拒步全Adam事务回滚|旧.01/.06不等于Gaussian资格。任何漏项、断言失败STOP_CORRECTNESS；不调容差|
|G05 完整恢复，P1→P2|同进程/新进程从块中/终止前/Goal变化/option中恢复模型、Adam、RNG、真实solver/history/phase/last-z、后续物理与评分按原严格合同一致|H0固定控制与head有真实PASS，联合PPO/FiLM没有。纯工程一次等价修复按§9；否则停|
|G06 原生源与初始实际采样，P1→P2|10个train资格父组（none/zero/near/mid/stop各2）全程mean＋Gaussian；standing与arm-ready及两K；§5.4保护；新categorical selector在P3前另核同集预锁子集6回合|H0只deterministic2s；不能替代本门。原生参考自己不合格不标Teacher、不换源掩盖，停报告|
|G07 预算/存盘资格，P1→P2与P3|两臂各4完整1024块实测采集/更新/恢复/IO增长；包括完整100回合评价、probe、reserve的p90×1.25投影≤剩余各项帽；输出峰值含临时包|现有H0吞吐只作量级。联合学习未测；装不下BLOCKED_BUDGET，先请新阶段预算，不缩水发车|
|G08 P2→P3低层放行|固定12k终点；≥8train父组必要阶段覆盖；train保持全过；接触路径有真实非零有效梯度/更新；P3新categorical采样G04/G06/G07过|不加任务完美/Q70%/EV门。任务尚未改善但保护与正确性过可进入P3；硬界/保持失败停相关臂|
|G09 P3→完整Dev|32k终点或已停止完整checkpoint；全34父组/44回合实际覆盖与真实stop/release；模型冻结；训练RNG与独立评价隔离；保留失败模型|覆盖不足UNDERPOWERED，不追加步数。可做失败结项评价但不得声称充分学习或走成功支路|
|G10 真实stop/释放与保持，所有物理阶段|逐tick事件/force证据；stop前共同前缀严格同；hold≥1.5s、release后4s；禁止none尾窗充数；同态恢复参考资格|H0 stop5.9–7.9<8.3是缺测例，非PASS。少窗数据不补“推定恢复”，标缺格；测量错误停共享评价|
|G11 真实机械/支撑/质量，训练→后续更新|§5.4真实q/支撑硬门；none/zero/recovery保护；loaded-slip/yaw三分/unwrap；连续性恶化单列|B19/Return旧恢复负例保留。硬门/持续保护失败停止学习，只读第一失败边界，不改clip/阈值|
|G12 超越Goal菜单，Dev→“学到反馈”声明|§7.1对照零参数查表/原生零U，≥4独立父组跨near/mid/真实stop-continue；达预锁物理效应且无严重转移|H0 E4/4与查表等同，故本门UNMEASURED。若无增量＝menu-only/执行适配无证，不由loss补过|
|G13 I行为增量，Dev→I晋级候选|§7.2成功率/物理成本/族不劣/尾部全部数字同时满足|H0 I3/4且三组同物理，I增量未证。FAIL/UNDERPOWERED不追加BI/更强I执行器|
|G14 I表示/因果消融，候选→归因BI|组级probe和固定权重Goal/表示disable/shuffle按§7.3；不作为E或不用该支路的算法启动门|rank3.14/训练MSE不是资格；无物理消融/样本不足＝归因未证，按第五分支|
|G15 工程自治，故障→原阶段续跑|§9同根因最多一次、≤800修复秒、原数据/模型身份不变及严格数学/恢复回归；保留全部失败|只有等价修复可自动续；复发、改科学量、碰Sealed/硬界/预算即STOP，无自由裁量加训|

**阶段状态机唯一顺序**：签署与资金→P0实现/锁定→P1资格→P2低层→G08→P3联合→冻结终点→完整Dev→必要有限消融/probe→五分支结项。P0/P1不额外创造H0.1、标量Q考试、Teacher或训练候选搜索。每门结果写机器可读evidence index：门ID、原始字段/公式/单位/窗口/分母、父组与运行SHA、PASS/FAIL/不足及唯一后续动作。**表中未授权的动作默认不执行。**

## §7 预锁结果、I独立增量与五分支裁决

### §7.1 两臂各自是否获得行为

开发固定终点、父组等权（stop/continue兄弟先父组内平均），K分层。

定义Goal-aware主物理成本，不用训练return替代：

`J_FOLLOW = .5*(error_RMS/e_ref) + .5*(force_RMS/f_ref)`，near参考4cm/4N、mid8cm/7N、stop hold8cm/6.5N；先在各真实阶段算，再族等权。

`J_HOLD = .5*(public_target_error_RMS/.04m) + .5*(force_RMS/7N)`。

force/XYZ误差/推进单项不能藏在J里。恢复、slip/yaw和硬界单列，不能被更低J抵消。

J的聚合固定：各父组先按真实moving/decelerating/holding与当前公开Goal求非空交集；交集内按真实时长计算RMS，再对该父组非空物理阶段等权、各Goal等权，兄弟回合父组内平均，最后同族父组等权。precontact和release后不进入active误差/force成本。没有某阶段记NOT_APPLICABLE，不补0稀释；真实该有的窗未记录是缺测，不是NOT_APPLICABLE。bootstrap固定1000次父组重采样、seed660304，不重抽帧/兄弟。

“学到有价值反馈”的方向性标准：相对0参数Goal-only，在near/mid/stop-continue中至少两族J降低≥10%，第三族J不恶化>10%；总体合格父组率至少提升2/16（12.5个百分点），或在至少4个独立父组达到误差≥1cm或力≥.5N改善且另一项不劣>5%；不得以不推进换低力，HOLD按公开位置任务单列。none/zero/recovery不新增硬/保护失败，slip/yaw尾部不新增严重退化。两个K分别报，不允许只选较软K归功。

同一checkpoint必须完成上述；不拼E/I/Goal不同batch。单seed只能称本父组集方向性实证，无统计显著性/收敛/泛化到未测机制主张。

### §7.2 I相对E的数值晋级判据（全部预锁）

I必须同时满足：

1. **I本身过§7.1行为标准**，先有实际行为，再谈表示。
2. 16独立dev父组中，Goal-aware任务＋质量联合合格率比E至少高 **2/16＝12.5个百分点**；兄弟/Goal分支不加独立分母。
3. near/mid/stop-continue至少两族 `J_I ≤ .90 J_E`，余下一族 `J_I ≤1.05 J_E`。每族force/误差任一不得以>5%的系统恶化换综合值；同时报绝对量。
4. K40和K80各自族等权J不恶化>5%；HOLD固定目标RMS不比E恶化>5%；none/zero和真实恢复不新增保护失败。
5. I不能新增真实硬界/失支撑；loaded-slip>.45父组率不高于E，p90 loaded-slip不高于`max(E×1.05,E+.01m/s)`；p90 yaw excursion不高于`max(E×1.05,E+1°)`。这些是比较容差，**绝对旧门照旧**，不能把它们当放宽物理门。
6. 连续性按§5.4不退化。数据/Actor容量/有效更新/Goal权限一致；任何额外数据或执行能力不匹配只能称I正结果，不能归因BI。

边界情形预写：E/I均16/16时，成功率没有提升空间，本轮**不以“成本明显下降”临时删第2条**；记有成本Pareto收益但I晋级证据不足。样本/安全失败造成有效分母不足16，不重算成较小的2/n门，记UNDERPOWERED。以上没有显著性承诺；分组bootstrap只提供区间。

父组合格为该父组所有预锁真实任务兄弟/Goal分支均满足各自Goal和质量门；不以一个漂亮分支代表父组。none/zero按保持物理门判，near/mid/stop按真实Goal窗口判。新增4个Goal反事实不扩大独立分母；K分层不把16变32。

### §7.3 表示和固定模型消融

在训练完成后，按独立祖先分组未来0.5/1s后果probe，比较 B_R、B_I、B_R+B_I、当前force/proprio简单基线与E普通outcome。行为先判，probe不能补失败行为。标签由真实后果给出，不是phase/未来task类别。

- probe要求正/负各至少4独立父组，至少8独立祖先；按祖先leave-one-group-out，同prefix不拆分。达不到即UNDERPOWERED。
- 二值结果预锁为：在测量力≥1N的合法起始状态，未来0.5/1s真实力RMS≤4N、loaded-slip≤.45m/s、yaw excursion≤20°且无真实q/支撑失败。它是有限“卸载且稳定”标签，不是完整Goal成功或phase分类。使用L2 logistic线性probe，C=1、无调参、训练折内标准化；B_R/B_I/拼接/简单基线/E都用同样组划分和优化预算。
- I相对`max(B_R,简单基线,E同容量表示)`的held-out AUROC增量≥.05，父组bootstrap95%区间下界>0，方可称这批数据有独立预测增量。若置信区间不足，保留点估计，不宣称显著。
- I固定权重、同状态、同Goal、同伙伴做Goal encoder shuffle与interaction通道disable；最多4个预锁dev父组×2干预完整回合＝8回合。只用已开放Dev，不读Sealed；随机shuffle只换当前公开Goal/表征，不读取partner未来。
- 因果支持：在4父组中至少3组，干预使Goal-aware J增加≥10%或失去原合格状态，且不是输入越界/通道尺度爆炸造成。否则记BI使用/价值未证。干预引发硬物理时立即停止该分支，记unsafe消融，不用其差结果自动证明BI价值。

### §7.4 五分支（承接原GPT E/I表，不许中途改出口）

|分支|固定证据条件|唯一结项/下一建议（均不自动执行）|
|---|---|---|
|1 E成功，I没有额外行为收益|E过§7.1，I未过§7.2或近似相同|保留E为强接触执行基线，I不晋级；不加BI更新、调权重补胜|
|2 E成功，I更好且表示/消融支持|两臂过§7.1；I过§7.2和§7.3|I成为下一Factorized候选；下一新Goal/接触机制或Student另签，不称已通用FB|
|3 E/I都失败|两臂均未过§7.1，或安全/数据等使完整行为不成立|停本配方；按Goal/reward、执行、on-policy信用/覆盖、保持定位。不给旧Q补考，不自动扩大模型/训练|
|4 E失败，I成功|I过§7.1而E不通过；I有实际收益|保留I正证；只有容量/初始化/实际样本/更新匹配且§7.2/7.3也过才归因BI，否则因果未证，不直接宣称Factorization胜出|
|5 行为接近或表示/消融/样本不足|§7.2未达到；或行为有差但§7.3不足；或独立分母不足|报告Pareto与证据不足，暂停BI独立价值主张；不按loss/rank补分，不新增probe样本找补|

安全/正确性失败优先于五分支，保留单独失败状态；不是为了填五格强行执行危险回合。

## §8 实际余额、完整预算与先行申请

### §8.1 已发生与拟议，绝不重置

H0结项实际R2 `21,137.928257 / 57,600 job秒`，剩 **36,462.071743秒**。历史输出末扫 `3.543830 /12GiB`，余额约8.456170GiB；这是旧扫描快照，签后启动再实时核。所有失败、CPU sidecar、冷启动、前缀/reset、独立评价、IO/恢复、报告都记，不重复计嵌套子进程。

**本轮只是修订文档，物理ticks、RL、Actor/表示更新均0。** 本次只读、文档同步、git/发布采用**400秒整体保守上界**单独入账（不是实测训练耗时），对应下表文档项，不再重复计。归档后R2保守累计为21,537.928257秒，剩36,062.071743；未发生的方案余额为41,000秒，故资金缺口仍4,937.928257秒。其余表项只是未来上限，不是已发生结果。

|段／固定出口|完整上限job秒|输出预算GiB|含什么、不得削什么|
|---|---:|---:|---|
|稿件/身份/归档|400|.20|本次只读源码、协议、隔离文档worktree、版本/发布；不含训练；全文件末扫约.164GiB，不只计两份md|
|P0 共同新接口/概率/恢复工程|4,000|.80|FiLM、四命令selector、Goal输入、reward与两频率概率、事务checkpoint/数值回归|
|P1 采样/保持/reward/吞吐资格|4,500|.80|mean/sample完整回合、同状态reward、E/I各4真实块的隔离工程更新/恢复及存盘增长；独立scratch身份，不混正式32k|
|P2 低层12k/臂|6,000（E2,900/I3,100）|1.00|两臂从共同原始初始化正式起步；新数据/更新/完整IO；失败仍计|
|P3 联合20k/臂|8,000（E3,800/I4,200）|1.50|含新categorical option资格、采集、joint PPO、两臂同终点；不加warm-up外样本|
|P4 完整统一Dev|12,000|1.30|§5.3全部100完整回合、实际stop/hold/release后4s、独立进程/完整记录|
|P5 probe与限定物理消融|2,500|.50|分组probe＋至多8完整消融回合，失败和unsafe保留|
|P6 结项/曲线/关键视频/发布|1,000|.25|少量完整1×qpos视频，不另开物理；CURRENT_STATE/Issue/Master/全部负例|
|失败/工程reserve|3,000|.65|其中≤800秒专属§9纯工程自治；其余2,200为失败/额外冷恢复预留，不得挪作多训|
|**完整方案总帽**|**41,400**|**7.00**|阶段帽和总帽任一先到即停|

正式训练≤64,000真实50Hz transitions，P1工程吞吐另≤8,192条（两臂各4×1024），合计≤72,192，不能把资格更新隐藏在正式计数外。P1只在隔离scratch副本按同实际数学完整更新，资格结束丢弃其可训练状态并回到共同原始初始化；这不是重选更好初始化，scratch数据也不进入正式on-policy buffer。它们的成本与失败全部保留。

source/prefix、reward资格/冻结评价另计实际physics。**全流程新增5ms ticks上限950,000**，含失败、reset、source、恢复与物理消融，不与50Hz条目混算；本稿不继承H0的45k帽。按完整manifest求出的tick下界若已超950k，停下改稿；不能只按64k×4漏掉源前缀和100个完整回合。该新帽与秒/输出帽同时约束，不可凭总秒有余就扩。

**资金缺口明确：41,400−36,462.071743＝4,937.928257秒。** 因新增双频率训练尚未实测，不宜把完整成本压成8h/8k承诺后再扩。建议Yi在签这份稿时**同时批准独立新阶段6,000 job秒（1小时40分钟）**，项目240h总账继续累计，旧R2 57,600帽原样保留；形成36,462.071743旧剩余＋6,000新额度的可审计资金来源。计划用41,400，留1,062.071743未分配余额，**不自动转成训练**。新增额度当前未批，不是已取得的授权。

若不批新增预算，本稿完整训练方案不启动；可回稿另排范围，但不能直接签“先做小段，后续再说”，也不能先删真正停止/恢复评价以塞进余额。P0/P1若实测需要超过完整41,400，必须在正式P2 actor更新前提出新的完整规模/预算，不能赌后面会加速。

### §8.2 吞吐证据与保守投影（未测不填数）

- 原R2工程250真实50Hz采集热块约18.327/19.401s，仅对应旧off-policy收集，不含本稿joint PPO/Goal/完整事件评价。
- H0实际evaluation全流程11,581个5ms ticks/303.445s≈38.17ticks/s，按每20ms=4个5ms tick折算9.54个50Hz时长单位/s；实际评价16×2s外仍有完整prefix。它是冻结推断吞吐，不是训练FPS。
- H0首个冷资格包2541ticks/155.751s≈16.31ticks/s，冷启动占比大。不能拿热块数字覆盖重启/IO成本。
- 本稿P2/P3真实训练transitions/s、1024块PPO秒数、joint模式峰显存、checkpoint delta增长、完整事件回合p90 **当前均UNMEASURED**。

P1各臂4个scratch完整块固定为2个低层模式、2个联合模式；另至少6完整source/stop/release回合。按模式分别取实际块成本**max×1.25**，同时附小样本经验p90，不把2次测量当可靠总体p90。以**每臂全部正式条目＋完整episode源前缀＋更新＋存盘＋48个train sentinel＋100回合Dev＋8消融＋固定reserve**重投影。不能只用训练阶段吞吐估评价。逐段/总帽/7GiB/950k ticks以及原R2＋已批准新阶段合计全部满足才过G07。P2→P3再用已经发生的真实块成本保守重投影；只能停报，不能据此改共同规模/配方。

### §8.3 存盘与共享资源

单RTX PRO5000 GPU0重作业串行；CPU只读准备可并行，成本累计。启动实时核GPU/PID/CPU/RAM/IO，共享free≥100GiB＋本轮峰值临时空间。预算有余不代表可抢占他人资源。

每1024新条目与完整更新后不可变可恢复逻辑checkpoint，含可训练与冻结引用/targets（若实际有）、Actor与selector/value/表示各Adam、normalizer、完整buffers、采样与Goal/phase自身记忆、全部RNG、Task/solver/接触历史/真实last-z/action、计数和hash/atomic latest。lossless delta/shared immutable base可用，必须验证随机历史提交新进程恢复；不得为7GiB删失败、省Adam或只留weights。停止包不可覆盖。

## §9 工程自治条款（沿原规则，不扩大科学权力）

规范性依据：[QA2合同§9 @4c75ea7](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research_notes/R2_EI_G17_QA2_VALUE_TARGET_CONTRACT_V1.md#9-纯工程故障一次自主修复与同预算续跑待qa2签署后生效)。**允许范围、同根因一次、现场保留、等价回归、复发停报不变**；该旧文的科学G1/G2/A0与旧QA2总额度不移植成新E/I门。只将阶段名映射到本稿、修复池仍≤800秒，包含§8 reserve，非新额度。

允许：IPC/启动/参数透传、路径/序列化、CPU-GPU装载、checkpoint完整事务/深拷贝、文件锁、日志封装、调度包装等**可证明数学等价**最小修复。每根因簇全阶段一次，以稳定`failure_family_id`去重，换run/异常文本不重置。

必须：冻结首trace/PID/runtime/触发输入/完整checkpoint/replay/RNG/计数，独立最小diff；原始资产/数据/SHA与split不变；固定输入/RNG下合法z、actualaction/q_PD、reward、duration/termination/target、模型前向/适用梯度/Adam严格回归；新进程真实物理/历史恢复通过原合同；仅从最后完整提交、同锁定随机流和日程续跑。未提交半轮废弃但成本计入；不把残余Adam捞成完成更新。

不允许借工程修改：网络/梯度图、Goal/reward、sampler/顺序、lr/EMA/batch/更新总量、概率、折扣/终止/截断、输入权限、伙伴/PD/action数学、source/seed/split、门槛/分母、Dev/Sealed或共享系统。纯数值≤1e-6rad内部舍入只按既有专项通则、证明不改变真实物理与判据关系；不能据此加新Envelope/改边界。

立即停报：同根因复发；800秒池耗尽；数据/模型身份错、数学或恢复等价无法证明；真实安全/权限/科学资格失败；需要方法/规模/预算变更或逼近资源护栏。所有修复、弃轮、失败、成本和新旧SHA单列Engineering Errata并同步Issue/Master。**修好工程只恢复已获本门许可的原阶段，不自动越门，不启动Sealed/Student。**

## §10 当前交付与签署栏

本次只交本修订稿、审查对应表与签前决定清单，不写执行代码，不新建训练stage，不跑新物理/学习，不改变已结项H0/QA2/B19/Return结论。旧稿保留。

签署需要Yi整体确认：

- [ ] §2 FOLLOW/HOLD语义、新reward系数与自身里程计仿真假设；
- [ ] §3直接contact-FiLM、有限命令接口及E/I匹配容量；
- [ ] §4分阶段采样/概率、两频率PPO、保持和有限后继数学；
- [ ] §5真实完整stop/release评价、父组/Goal/K/source划分；
- [ ] §5.1共同原生onset参考锁anchor的明示配对初始化修订；
- [ ] §6逐门三列表/固定阶段证据/失败状态机；
- [ ] §7 I增量阈值、五分支，包括天花板和样本不足也不事后改门；
- [ ] §8完整41,400秒/7GiB方案及**新增独立6,000 job秒申请**，项目总账余额足够；
- [ ] §9工程自治边界。

**未签署；审阅前无任何新训练授权。** 签署后才展开机器可读参数/schema/manifest，所有数值必须忠实对应本稿；任何需要新增科学选择的空白必须回审，不能用“工程默认”补齐。当前不会因为合同稿已推送而自动运行P0/P1/P2/P3。
