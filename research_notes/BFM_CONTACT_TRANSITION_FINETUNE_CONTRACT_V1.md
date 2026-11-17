# BFM-CTFT v1.0｜接触条件下 BFM-Zero 继续训练与 Motion-Transition 采样

**日期：2026-11-17。状态：DRAFT / REVIEW REQUIRED。** 用户已认可“先做直接微调BFM-Zero”的方向并要求起草、发布新Issue；本文提出新的科学配方与资源上限，**发布不等于已经签署具体数值预算或启动服务器任务**。用户明确批准本文或将本合同交给Codex并指令执行后，Codex才能依本文连续运行。禁止借旧Issue #66或旧QA额度先开跑。

## §0 独立性、事实与问题

### 0.1 本轮是什么

**原始预训练BFM-Zero + 接触观测输入扩展 + 原生FB/CPR/辅助奖励训练循环 + 真实motion-transition状态采样。** 这是增量继续训练，不是从零预训练，也不是旧E/I v2加补丁。首轮研究单腕guidance mode下，机器人能否从局部伸手进入稳定迈步、在引导停止/解除后撤销多余运动。

只复用：（1）原始BFM训练权重和对应config；（2）原生运动数据；（3）经验证的G1模型、29D动作坐标/PD、力作用与5ms物理、完整状态恢复、数据身份/日志/存盘框架。旧F1/H0等存档可以作为**已暴露的工程反例与源状态参考**，不能进入新盲测或冒充新训练后果。

**不继承**旧A128/B19/Return/E/I v2的任何学习权重、optimizer/replay；不引入四命令selector、Guidance U、FiLM、输出动作residual、Envelope、结果预测头、B_I/F_I/z_I、旧Q70%考试、PPO joint-ratio/OR1、IK/LPP Teacher或搜索专家。原生FB所需Forward、critic、discriminator是底座自身算法，不属于复活旧任务Q路线。

### 0.2 当前事实与预算起点

[旧E/I正式结项](https://github.com/luoxinyuan/compliance_BFM/blob/gaoe05/research/research_notes/CONTACT_GOAL_EI_V2_FINAL_RESULTS.md)：每臂32k/127次Actor接受；100回合Dev；E/I与Goal-only均7/16父组合格；selector未改变已核查状态中的初始argmax。旧配方停止，sealed未读，不能重判为新方法失败。

项目现账保守上界约**136.293 job小时 / 240小时**（包含历史保守重复计费，不冒称去重实际GPU工时）。本轮单独建stage；不得挪旧阶段未用额度或重置总账。用户资源为**单卡Pro5000、报告显存80GB**，实际GPU UUID/型号/可用字节/RAM/CPU/磁盘须现场记录，不假定其它卡、CPU核数或空闲资源。

### 0.3 本轮唯一主假设

> 在相同原始BFM与相同接触奖励下，直接联合适配Forward和Actor，并增加转变前后真实状态的rollout覆盖，能否获得超出冻结底座的、由持续受力引起的全身转变？与普通状态采样相比，transition-focused reset是否有额外价值？

本轮不宣称学习通用人类意图、任意接触点、双肩力偶、真实坐椅子或可零样本寻址的完整Interaction FB。成功后才讨论新增交互latent；失败不能以“必须先scale到大模型”无限续训。

## §1 版本、模块与加载纪律

### 1.1 原生实现依据

- [官方训练说明](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/README.md)
- [FB主训练及初始化](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/humanoidverse/agents/fb/agent.py)
- [FB-CPR采样/运动先验](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/humanoidverse/agents/fb_cpr/agent.py)
- [辅助奖励、critic及Actor统一更新](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/humanoidverse/agents/fb_cpr_aux/agent.py)
- 本轮设计时读到官方blob：fb agent `7a8b14d9db3a3931fd973b18dfb73e773965d91f`；fb_cpr `8a7c21ecf0ad3047beeb14ddcae4d05aac93ae28`；fb_cpr_aux `8387bbec56ace2e8e8c69bd07bfdb5ae23227375`；README `f17264c522799d9605057f99c9e01bafb1a72f6e`。执行前固定完整commit及依赖，不能随main漂移。

源checkpoint须核对既有原BFM资产SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`及原训练config是否一致；若该资产只是部署导出而缺失可训练F/B，必须报告并定位官方匹配训练checkpoint，**禁止混用另一版本网络而称同底座**。

### 1.2 模块表

|模块|本轮状态|理由|
|---|---|---|
|原B_R与其target、原robot normalizer|冻结|保持运动latent坐标与原统计，不同时引入表示坐标漂移|
|原Actor（含原embedding和全部policy blocks）|从原权重继续训练|允许执行器真正学习接触动力学，不再只改末层或选择菜单|
|原Forward F_R及target|继续训练|策略和受力动力学改变后，重新学习后继表示；B固定不等于F可固定|
|原CPR discriminator及其critic/targets|按原生目标继续训练，输入仍为机器人运动字段与z|保留运动先验；不把force/partner字段送进判别器造成“见过接触就判fake”的捷径|
|辅助critic/target|对明确的新总辅助reward在线训练|不得加载旧B19/QA的任务Q；原aux head若目标尺度改变，需明确fresh初始化/target复制|
|接触输入扩展|新增零影响初始化输入列|直接观测反馈，不输出动作增量、不使用FiLM|
|新optimizer与replay|fresh，两训练臂完全同初值|不得继承旧Adam/行为分布|

**加载陷阱必须测：** 官方`setup_training()`包含`apply(weight_init)`。正确顺序是构建网络→加载原权重→设置新增输入列为零→按约定复制targets→fresh optimizer→冻结B_R/原normalizer。任何后续初始化重新覆盖预训练权重都为BLOCKED。逐模块列出加载键、缺键、param SHA、requires_grad和optimizer所有权。

### 1.3 接触输入与权限

部署Actor只读原本体/动作历史、0.2s的三维实际外力历史、已知单腕接触位置、自身腕/躯干运动学以及公开mode位g。力在同一身体坐标系表示，位置/速度由机器人自身获得；本轮ideal-force与自体里程计为仿真假设，不宣称真机传感器已经满足。不得输入隐藏anchor、partner速度、force的未截断真值、phase、未来停止时钟或源motion ID。

F/critics训练期可按原asymmetric学习原则读取**当前**伙伴相对位置/速度、raw/applied force；未来partner轨迹不可用于actor/action或target动作。B_R保持原机器人字段。新输入通过原embed_s/embed_z/F/critic的首个线性输入扩维实现：旧列复制原权重、新列零初始化；原state normalization不重新估计，新增字段采用固定物理尺度。正常无接触且新增列为零时，原Actor动作/q_PD需在既有FP32执行口径下等价。

## §2 训练算法：原生继续训练，不是旧PPO重命名

保留原FB后继目标、CPR判别与风格critic、物理辅助critic和latent条件Actor的更新关系。B_R固定后，FB损失仅更新F_R；B正交项只作监控，不创建空optimizer假称学习。数据按真实(s,a,s_next,z,g,dt,terminated,truncated)入库，不替换为预测后继。

Actor目标结构保持：`-(F_R·z + lambda_CPR*Q_CPR + lambda_aux*Q_aux)`，系数/scale_reg的现有来源逐字段锁定；新增接触项按§3进入明确的辅助reward。**使用原生算法仍然需要critic，但不再设独立Q排序考试作为整条主线门。** critic/FB误差、动作梯度、各项对Actor的梯度份额是训练诊断；最终用真实物理和保持评价裁决。

首次训练具体预注册：
- logical batch=1024（须与源CPR seq_length整除；若不满足，开训前按最小整倍数规则导出并锁，两臂相同）；Actor lr=3e-5，F/CPR critic/aux critic/discriminator lr=1e-4，weight decay=0，gradient global norm clip=1；B无优化。
- FB target tau=.01，critic/aux target tau=.005；discount、actor_std、stddev_clip沿用**匹配原checkpoint训练配置**，把数值打印入锁，不继承旧v2的sigma=.01和joint-KL=.02。明确每个discount对应真实20ms，而非原频率不同仍机械复制；频率不一致先BLOCKED。
- 每臂先收8192条真实新数据，Actor冻结，用64组更新启动F/critics/discriminator；随后每新增1024条真实数据做8组完整联合更新（每组F、CPR/aux、Actor及target各一次），不沿固定旧数据连续优化Actor几百步。允许这些新数据的off-policy回放；每次policy version/曝光/动作支持记录。
- 原生B目标编码与mixed-z机制保留用于无接触运动保持域；**guidance域固定使用预锁的upright/standing名义目标z_R，不依据施力方向、停止时钟或未来成功运动给不同z。** 首轮是“固定公开引导模式下学会响应”，不是任意z的接触zero-shot能力；guidance replay不重新标记成未执行的高层交互意图。无接触保留域依原生mixed-z比例，CPR reference与其z身份匹配。
- g=1时名义z提供运动/姿态偏好，并不构成“必须绝对不迈步”的硬约束；原FB项仍在，接触辅助项允许行为取舍。g=0不启用接触reward、执行原生运动任务。检测原FB项和新增项是否冲突，但不临时改变权重；若无法学出转变，列作结果和下一版本设计依据。
- 无额外scalar-Q考试、BC/IK专家、PPO回溯、selector或z搜索；不因旧Q失败禁止当前原生critic在线联合学习。只允许单个固定配方，reward/lr/EMA/更新率不得通过Dev扫描。

## §3 新接触任务与reward（公开guidance mode，明确改变旧定义）

### 3.1 伙伴、动作及时间

仅仿真、单腕、平地。沿用原G1/PD、50Hz动作/200Hz物理、K80/D8/10N原域；K40/D8/10N作为现成可核验的第二阻抗域。所有方法同分布，raw/applied公式逐值校验。**不增大10N上限来制造可控信号。** 模拟伙伴为活弹簧而非真人反适应，结论不能写成已解决general human interaction。

伙伴沿前方及小角度（±15°）引导，训练速度集合 .08/.12/.16 m/s；启动平滑、停止分为平滑/短暂保持，实际接触解除后保留4s恢复。引导路径主测试必须超过静态腕工作空间，候选.60–.90m，最终由已知机器人运动学资格锁定；不能用只移动几厘米的case证明学会走路。脚本从完整reset后再独立采样，强制同一源状态同时存在有力/无力/停止续行反事实；绝对episode time和phase不送Actor。

### 3.2 新force项的数学含义

g=1允许跟随；当前filtered外力由**因果**EMA产生，tau=.05s、真实dt=.02s。取水平力 f_h，deadband=.5N：

`v_ref = norm_clip(0.025 [m/(N*s)] * max(0,||f_h||-.5) * f_h/(||f_h||+1e-8), .25 m/s)`。

该v_ref**只用于定义训练reward与解释指标，不作为新高层velocity命令输入或手写控制器直接驱动Actor**。Actor通过实际力和自身状态学习。采样噪声、坐标系、滤波状态均存档；首次无接触history必须来自真实前缀，不伪造全零的未来历史。

每个真实控制步的新增接触reward：

`r_contact = -dt * [2*||(v_wrist_xy-v_ref)/.25||^2 + .25*(||f_applied||/10)^2 + .1*(v_wrist_z/.15)^2 + .5*w_idle*||v_root_xy/.20||^2]`，

`w_idle = max(0, 1-||f_filtered||/1N)^2`。

总辅助reward是**原物理辅助reward（原尺度/聚合不变） + g*r_contact**。force额外项按真实dt积分，不能把50Hz与5ms重复累计。新增项独立日志，与原aux总值/normalizer均归档；原aux reward normalizer若不能处理新分布，须在正式前使用两臂共同的固定train-only无学习校准集合建立一次固定尺度并锁定，**不按单状态梯度或Dev结果调权**。calibration本身包含在准备预算且不得更新Actor。

解释：它不只是“力越小越好”，而是用有界力—运动响应提供饱和区仍有的方向反馈；当实际引导消失时要求撤销无需求root运动。**这是明确带导纳先验的任务reward，不是无先验人类意图学习，也不是未来处方性FB已经实现。** 此轮g=1无力时希望停止；g=0原生walking仍可保持走路，不混成“任何释放都必须静止”。不新增anchor误差追踪Teacher，不对latent指定未来步态答案。

数值在训练前锁定；force10N时v_ref≈.2375m/s，partner .12m/s的准静态平衡力约5.3N，K80对应约6.6cm、K40约13.3cm延迟，这解释本轮为何不能盲搬旧统一8cm任务门。真实动力学不保证达到这些近似值。

### 3.3 运动保持

原native安全reward/CPR示范混合必须包含站立、平地走/停/小转向，而非只有standing。每个训练臂75% episode为g=1 guidance域（其中20% episode无施力），25%为g=0原生运动保持域；两臂比例相同。保持域不加入“有力必须站立”监督，不把有效接触步态视作分布错误。无接触恢复测试必须从该策略**真实受力后release状态**进行，不用另一个好状态代替。

## §4 sampler：核心变量是初始状态分布，不把答案写进控制

### 4.1 共同源池与物理资格

从合法原始motion数据及**原BFM无接触真实rollout**建立源池。motion只提供候选，不直接把人体q/qdot强塞入MuJoCo；源必须经过当前机器人、PD、支撑与实际状态恢复资格，保存真实q/dq、接触solver、action、previous-z、所有历史和RNG。不能只恢复q/qdot，也不能将不同policy后的腕位置重新对齐成不同伙伴。

至少12个真实source lineage/片段祖先，含站立到步态建立、行走减速/停止、初始姿态差异；同片段近邻时刻、左右镜像和随机扰动均归同祖先，不冒充独立样本。原生运动ID不进Actor/critic预测身份。源池不足不造12个新hash凑数，报告DATA_SUPPORT_BLOCKED。

transition按原真实物理定义：从root平面速度<.05m/s的稳定姿态，到>.15m/s并持续≥.2s且出现有效左右支撑转换，或对应减速事件。脚步事件必须用真实足接触和离地/落地，不用外力启动时钟。登记t*前[-1.0,-.1]s、附近[-.1,.3]s、后[.3,1.0]s窗口；先锁规则再选源，不能按新训练结果挖最有利的起点。

**防止初始化伪成功：** 模式、名义z_R、伙伴起点/随机流都固定的同态无力对照必须保留；转变前源若无力也自然继续走，则该源只能证明延续/维持，不能证明“受力触发启动”。真正启动结论主要来自更早静止源与未用于transition训练的新source。

### 4.2 三个比较对象

|方法|学习|reset分布|
|---|---|---|
|B0 原始BFM|无，原checkpoint固定|统一评价池|
|U：Direct-FT/Uniform|§1–3原生FB继续训练|同一个源池的完整合法时段均匀采样|
|T：Direct-FT/Transition|与U逐模块同初值、同更新/数据总量/伙伴/名义z/奖励|50%采transition前后窗口（窗口内前/中/后比例1/2、1/4、1/4），50%仍从完整合法时段均匀采样|

**U/T唯一主差异是reset sampler。** 不同时给T更多接触数据、motion专家、reward、更新数、特权或不同z课程。replay抽样两臂同一规则：按g与实际接触阶段保持预锁配额、层内均匀，不因T表现好再重采高收益行；不同reset自然造成的访问分布差异是待检验效应，逐阶段记录。达到相同环境steps不等于有效交互曝光一样，独立报告真实active/post-release/stance-step与parent覆盖。

两臂串行，seed同一固定值681117，source/伙伴外生配对随机流预生成；真实闭环状态允许不同，不伪造逐值一致。B0不加FiLM或临时新控制器。旧Goal-only只作附录参考，不进入主假设胜负。

## §5 训练规模与更新节奏

首轮目标量级为**每臂524,288条真实50Hz transitions**，不是承诺该量已足够。最终N须在工程吞吐测量后、正式Actor行为结果与Dev未查看前按本节规则锁：从{131072,262144,524288,1048576}中选择在**每臂12 job小时学习子帽**内按保守全链成本可完成的最大相同N；两臂同N且不超过1048576。默认优先524288，选择其它值须记录固定算式和硬预算原因，不以结果好坏调N。若连131072/臂都无法在预算内完成，则T0_REPORT，不启动不对症的又一轮32k训练。

T0 probe只用于真实load/physics/完整FB更新/IO量测及正确性，不挑学习效果；样本和权重不得偷偷混入正式U/T初始化。每臂正式从同源原BFM和fresh优化状态重新开始。

统计分别给环境steps、真实5ms physics、optimizer更新组、sample exposure/replay unique、独立motion祖先和事件数，不将并行环境数乘两次。训练固定终点；N/4、N/2、3N/4为train-only安全/功能报告节点，不据Dev选择checkpoint。

每个逻辑训练轮采1024真实新行→8组完整更新→持久化提交。8192首轮Actor冻结的F/critic启动只是既定算法暖身，不需要再次考旧G17动作排序。Actor进入正式更新后即尽早看真实控制效应和安全；不让critic单独训练成新项目。

## §6 单卡Pro5000 80GB：优先提升真实吞吐，不用塞满显存冒充利用率

### 6.1 资源权利

仅用户指定GPU0/UUID一张卡，核实actual total VRAM，不因名字推断容量。允许在该卡上使用全部可用算力，但**预留OOM/原子保存空间**：planned峰值allocated+reserved≤实际总显存88%，且实时free≥max(6GiB,8%总显存)。不占其它GPU、不抢其他用户进程、不改驱动或共享环境。共享磁盘free保持≥100GiB；新增stage输出上限建议**120GiB**，模型、source、replay等实际大小先测，实际不足时停止报告，不删旧研究资产。

### 6.2 高吞吐实现

第一选择复用已验证MuJoCo后端、原robot/solver/timestep，**多个隔离CPU physics workers + 单GPU批量Actor推理与学习 + GPU/固定页内存replay**。VRAM不是MuJoCo CPU步进的替代，实际CPU核与内存不足必须报告，不伪称80GB就有几千并行环境。官方已暴露online_parallel_envs/buffer_device等接口可复用，但现有contact callback必须逐worker独立，禁止共享mutable Task/force/RNG/optimizer。

纯硬件筛选按worker数{1,4,8,16,32}递增，受CPU/内存上限约束；测固定已存源和固定命令的端到端 throughput、P95 reset、推理等待、完整FB更新、IO及显存，固定准则选最优且不超过前述护栏者。若加worker增益<10%或等待/峰值恶化则停止扩容。不改变logical batch、sampler份额或学习率以“提高GPU使用率”。logical batch1024的pairwise FB/白化项**不能未经全批数学等价验证就拆microbatch**。

replay优先放GPU（内存允许时），容量上限min(1,000,000行,2N)，实际bytes/row测量。只对原始非数学缓存做无损压缩；不把接触小信号改成低精度。首版FP32、固定TF32/矩阵精度；AMP/compile/CUDA graph默认不改变，若源训练本来使用则锁原设置并作回归。切换精度或后端不是本次纯工程自治；**不以MJX/Isaac迁移作为本轮前置长项目**。若MuJoCo实际吞吐无法满足N最低要求，提交结果和单一迁移提案，不无界重做引擎。

### 6.3 完整实测与预算锁

至少完成1024真实接触行、连续若干完整联合更新轮、包含g0/g1与至少两个source/实际release、两个完整checkpoint和一次新进程恢复。必须同时给出：s_collect/N、s_reset/episode、s_update/group（不是actor单独forward）、s_checkpoint、s_eval含prefix/恢复、V/RAM、输出增速。

锁定投影：`C_arm = C_cold + N*c_collect + n_reset*c_reset + (8*N/1024)*c_joint_update + C_checkpoint_all + C_train_protection`，各项不得重复计并行wall。用多个块P90与至少25%余量做估计，再选择§5 N。总throughput含准备、编译/冷启、存盘、失败与评价，不用旧PPO单步速度直接外推完整FB训练。

首正式16384行/臂再测完整增长；如果剩余预算不能完成已锁N和评价，安全保存并报告。不得因为看到reward在涨就增N，也不得悄悄减N后仍宣称原预注册实验完成。

## §7 新阶段硬预算提案与授权方式

**这是资源控制上限，不是尚未测得的训练时长预测。** 旧stage已结束，本轮不能借旧余额。建议新stage上限**40 job小时/120GiB/16,000,000个全流程5ms ticks**；上限包含在项目240h总账内。按当前保守136.293h，全部用满约176.293h，仍须运行前从实时ledger核对，不以旧数字覆盖新增花费。

|包|上限job小时|所含内容|
|---|---:|---|
|S0 身份/源池/真实force与原始数学/全链试车和吞吐|6|包括原checkpoint加载、独立workers、接触扩维、FB真实更新、源码/标签/恢复资格；不是独立Teacher工程|
|U完整训练|12|正式N、reset/actual physics、全FB/critic/Actor优化、每轮提交、train保护、所有冷重启|
|T完整训练|12|与U同N及同更新配置；成本单独实记|
|独立完整评价与有限归因|6|新父组、B0/U/T、同态no-force/stop/continue，原生保旧，视频；不读取旧sealed|
|证据核验/数据表/正式结项|2|含所有CPU后处理和保守未逐项IO|
|纯工程修复reserve|2|一次/根因，全部失败、修复和回归，不另增额度|
|合计|40|不自动用满；S0实测后锁可完成的N/评价分配|

并行worker是同一个训练job，主阶段wall只计一次；同时独立记录GPU active-time、各worker CPU、累积physics和全部失败，用原项目ledger规约避免既漏账又双计。16M tick与N近似关系是50Hz样本×4，再额外算source/prefix/reset/reference/eval，不能只数训练行。

**发车程序：** 用户签署完整v1后授权S0→匹配U/T→固定终点评价连续执行；阶段通过不用每次等确认。仅S0无法取得最低训练规模/源池或数学资格、科学变化必要、硬物理/权限、预算或恢复阻断时停报。不得在签署前先试GPU。若用户只批准S0，应明确只S0；旧方向同意不替代这40h具体授权。

## §8 真实行为评价：不要让“从快成功的状态接着走”冒充学会启动

### 8.1 固定评价设计

在任何正式训练前，按**motion/source lineage**划分新的train/qualification/eval，旧已暴露Dev仅作工程对照、旧sealed完全不读。新eval至少24个独立source+partner父组；同motion近邻时刻、stop兄弟、K变化或新seed不能冒称独立祖先。至少一半从未用于transition近成功窗口的更早自然静止状态起步，另半覆盖真实转变附近和已经运动的状态；若独立源不足，缩窄科学结论并在开训前报告，不造ID。

每个父组以相同完整初态/同z/g构造continue牵引、可观测stop、no-force三分支（stop/continue共享真实前缀），每个方法B0/U/T各运行；主域≤216个完整回合，平均任务段约6–10s再真实release后4s，不截掉失利阶段。另保留12个原生g0站/走/小转向片段×3方法。名义z不随真实force答案变化。

所有触发从真实传感/物理记录定义：t_contact、t_stop实际伙伴减速、t_release实际施力解除、step实际足部离地落地与支撑交换。phase只供train critic/reward或评价，不能送Actor。评价P95耗时纳入§7，不够则开训前锁较小且有解释力的样本数；正式后不能挑有利回合。

### 8.2 主结果

1. **受力引起的可用跟随：** 真实腕/身体沿引导方向推进；对长距任务腕有效推进/伙伴实际推进≥0.8，active RMS负担≤7N、RMS几何lag≤.20m（明确不同于旧固定8cm任务）；持续饱和比例、p95/peak和未截断力另报。
2. **真正全身运动：** 长距关键子集root沿引导方向≥.20m，至少两次左右交替有效step/support transition；承重足滑移和足材料点轨迹同时报告，禁止拖滑而无步态被记作迈步。
3. **停止/解除：** 完整事件后两秒内root平面速度降到<.05m/s并持续≥1s；release恢复4s内额外root路程≤.15m、yaw相对release起点最大连续偏离≤15°，与同态B0参考并列。原生g0 walking不套此“解除后静止”门。
4. **任务完成＋质量交集：** 原实际关节超界0.02rad硬终止不变；跌倒/非法状态/躯干硬界沿 pinned原实现；真正walking需使用已资格的足支撑时序，瞬时no-support逐tick记录，持续>0.05s为本新任务安全阻断（必须用无力原生walking先验证规则，不把接触切换一帧当跌倒）。loaded-slip原峰≤.45m/s及p95/累计材料位移分别报告，不调整法向阈值掩盖失败。上述是本新行走任务公开定义，不追改旧E/I判决。
5. **因果反事实：** 同态有力与无力在早期自然静止源产生不同的可重复root/腕响应；无力g1不应沿预置transition自动行走。真实无力是否自发继续运动是分类变量，不能从均值扣掉后强称“力触发”。
6. **原技能保持：** g0无接触原生tracking/goal效果相对原BFM，原动作任务成功率不降低超过10个百分点、父组运动误差中位不恶化>20%，无新硬事故。不能以guidance收益掩盖彻底丢失原词典。

目标结果不是只降低训练reward。报告每父组全部任务指标和置信区间；单seed、有限祖先只能称pilot证据。

### 8.3 可证伪判读

- **直接微调成立的必要证据**：U或T相对B0在自然起点的联合任务成功率至少+20个百分点，并有多祖先真实迈步/停止/释放成功，无运动保持/安全退化。小样本区间仍跨零时仅称趋势，不宣称可靠泛化。
- **sampler额外价值**：T相对U达到同样预算下成功率≥+10个百分点，或配对lag/负担改善≥10%且成功率不低、不损害安全；至少跨4个不同source lineage方向一致，配对分组不确定度公开。未达则接受“本域transition采样未获额外证据”，不隐藏U。
- T仅在转变附近成功、从自然静止源失败→只证明续接能力，不能宣称学会自主启动。
- 两臂都只站立或arm-only降低力→本配方未获得所需全身能力，不增训凑成功；查执行更新/目标冲突/数据可达性后另定科学决策。
- 结果显著但只在K40→标伙伴域依赖，不把软伙伴当模型泛化。
- U/T均有用而T无增量→保留直接微调路线，弱化sampler假设。
- 成功不自动等于完整处方性FB；下一轮才设计共同执行能力上的显式交互参数与B_I/F_I/z_I对照。

## §9 每轮恢复、证据保存与工程自治

每个1024真实数据＋8完整更新的轮次，**提交可恢复状态成功后才记completed**：Actor/F/全部critic/discriminator/targets、全部optimizer及Adam step、冻结B/normalizer身份、replay有效数据/采样指针、每worker真实Task/solver/action/力滤波/上一z/历史/RNG、global scheduler和成本。未来control等价所需的有效solver数组必须按当前后端恢复；不复制错误的旧数组白名单。

原子临时写→hash/readback→fsync→latest指针。最新与前一完整代始终可恢复；预登记N/4、N/2、3N/4、终点、首有效学习轮、安全失败前后完整包永久保留。本**新stage**明确允许在最新两代与预登记锚点都严格读回后，对非锚点的过期中间包按已锁manifest保留规则回收，留下其hash/更新/样本/成本日志；**不声称永久保存每个历史轮的所有张量**。不能回收旧项目资产或失败证据；用户不接受此存储策略则先修订输出预算。replay与源存档不可变/增量去重，完整指标保留，选定评价保存全部5ms物理；训练不用为每一帧保存重复全solver巨包。

必须注入一次普通中断和一次事务中途失败；恢复最后完整轮后用同scheduler/RNG验证后续数据/更新，无双计/漏计。当前语义同后端同shape需严格等价；并行顺序与旧单环境本来不同，不许把两种轨迹长期逐值不等称训练错误，也不许借此放松**同运行配置的重启等价**。

纯工程可自主：IPC/启动/路径、隔离worker、无损存盘/装载、元数据计数；程序为冻结现场→根因→最小修复→源/模型完整性＋同输入数学与状态恢复回归→原阶段续跑。**同一根因簇整个新stage仅一次**，复发/超过2h总repair池/完整性不过/触科学变量即停报。不得改sampler/lr/EMA/reward/action/批量损失数学/phase/训练规模/阈值/权限/后端物理。任何真正安全失败不是工程错误。全部Engineering Errata、成本和首失败写里程碑，不静默修复。

## §10 门语义三列表与预写失败分支

每门落地到机器可读gate_semantics.json，列字段、单位、时间窗、坐标系、比较模型/父组、pass_case/fail_case来源SHA、证据等级和唯一后继。下表是合同级定义；新域尚未有真实PASS的门必须写PENDING，不制造历史成功。允许基于历史实际量构造清楚标记的单元fixture，但不能以fixture代替真实物理资格。

|防什么|怎么算|已有正反例／缺证；失败后自动一步|
|---|---|---|
|加载时原权重被重置|setup后实际参数SHA＋零新增列前向与原动作一致，F/B键齐全|原E/I原生identity检查是历史PASS参考；官方setup含weight_init是必须拒绝的顺序陷阱fixture。FAIL→只读键/首差异，不开始采样|
|偷偷继承旧机制|模块/optimizer列表中无旧selector/FiLM/U/aux预测头/旧model SHA|原E/I当前目录含这些模块，若被本轮导入即FAIL；本轮尚未实测PASS。FAIL→依赖审计，不动旧文件|
|输入泄漏|仅改变隐藏anchor/未来phase，Actor的同输入动作不变|旧前缀/隐藏字段干预可复用验证方法；未来字段进Actor的注入fixture应FAIL。FAIL→停并列依赖链|
|force项只让机器人硬顶或瞎跑|真实f、v_ref、v_w、dt分项一致；无力v=0成本最低、有力静止响应成本高、同向限速响应降低该项|f=10N时v_ref=.2375、v=0与v=.2375可作PASS/FAIL算术fixture，非行为实证。FAIL→只读reward积分，不改系数|
|转变起点泄漏答案|同态同z有力/无力、以及自然早期起点；来源身份/preceding history真实|历史M41“好状态可续、坏状态难救”说明须区分启动和保持；新方法PASS未有。FAIL→同源已登记冻结对照，不再挑成功帧|
|并行worker共享状态|固定已存输入/独立Task/Force/RNG、共享模型只读、actual replay next-state正确|QA1 Adam alias旧失败是拒绝例；本轮需worker间污染fixture和新进程重启实测。FAIL→按工程一次修复|
|真实关节/支撑风险|实际q excess>.02rad或原fallen界；walking持续no-support>.05s|B19 mean .02756rad旧FAIL；旧无接触合法基线q excess0为PASS例。原walking支撑新门未测即BLOCKED，不擅自放宽|
|yaw与路程混淆|unwrap max excursion不是TV；root路径不是net位移|旧none TV39.323°但range3.730°应过偏航门；旧A128连续1081°应FAIL。FAIL→旧5ms时序只读分解|
|拖着滑不是迈步|≥2左右交替实际离地落地/支撑、root>.20m、stance滑移另判|H0手动而脚仅毫米不算全身迈步；本新任务真实PASS未来待测。FAIL→逐脚接触/材料点表，不能改任务门|
|仅降低力没跟上|progress≥.8、active burden/lag、安全与停止交集|旧E/I case17腕5.97cm/伙伴28.54cm仅.209，明确FAIL；新整任务PASS未有。FAIL→保留负结果，不追加Dev反选|
|更新没有改变执行或丢原技能|按实际Actor参数/同状态输出/真实rollout及g0原生保旧判，不以loss下降晋级|旧E/I127更新但无共同任务增益为FAIL动机；原BFM保持作为正参照，不预写新PASS。FAIL→限定train诊断，不新开Q考试|
|存盘遗失末轮|最新完整round原子可恢复，注入故障拒载、重启模型/Adam/replay/RNG/physics一致|旧97D仅step0是FAIL；既有严格原生重启逐值PASS可借工程方法。FAIL→最后有效包读回，禁止续新采样|
|OOM/预算超限|实时VRAM与RAM/CPU/磁盘、N预测与全flow ledger不超§6–7|用户80GB非实测性能；历史旧PPO耗时不是本新FB速度。FAIL→保存停止、报告实测，不擅自改batch/loss或加卡|

失败若是本轮尚未学到行为的**科学阴性**，按固定阶段结项；不能把所有坏episode都当工程异常中断，也不能忽略真实硬安全。训练早期可出现不成功且未违反机械安全的探索，完整入库、正确终止，不删除负样本。发现安全硬终止率异常或新机制导致重复硬事件时停止该臂并一次只读归因，不继续用随机seed绕过。

## §11 交付、接力与终止

阶段独立命名`bfm_ctft_v1`，个人worktree/分支实施；不改共享main、旧部署资产、原BC500或原sealed。发布issue只在Bridge，执行代码由Codex经用户签署后在研究分支实现。

必须交付：
1. 方法/参数/原生代码差分/源权重SHA与冻结表；至少一张实际数据流图说明contact如何进入Actor/F、B固定何处；
2. 完整source lineage与transition标注、无force反事实、伙伴定义、信息权限、采样与时序锁；
3. 单卡80GB真实VRAM/CPU利用、并行吞吐、完整FB/update/IO/恢复测试、N选择算式、预计及实际成本；
4. U/T固定终点与全部旧B0比较、自然启动/转变附近/停止/恢复/原生保持分层；负例、任务质量交集、有效步态视频与独立祖先数；
5. 每次工程错误、科学停止、完整checkpoint/实际replay/目标政策与收支；不能把结束作业说成研究通过；
6. 明确下一决策：保留直接微调／sampler无增量／数据不足／目标冲突／执行质量失败。不得自动续训、另开Teacher、加入新z_I、增加接触部位或真机。

**签署待办：** Yi确认本科学范围、40h/120GiB/16M ticks新stage上限与新stage检查点保留策略后，Codex可在合同内连续推进，不在每个普通里程碑重复等待；如果用户只是要求先完成S0，则仅给S0结果与实测后规模锁。当前状态为待审阅，不运行任何新训练。

