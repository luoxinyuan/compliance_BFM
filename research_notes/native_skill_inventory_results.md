本轮完成原生BFM接口与已有资产盘点。**当前确实有goal / tracking / reward三类latent生成接口，但还没有证据充分的“局部右臂技能 + 全身技能”可直接从同一standing起点切换执行。首个缺口是经过当前checkpoint验证的原生命令实例和进入/退出行为，不是再写一个腕部控制器，也不是BFM没有全身能力。** 本轮没有物理rollout、模型前向、Student/PPO训练或新的latent搜索。下面区分源码能力、实际资产和已完成的物理证据。

**真实模型/执行合同**

- 当前资产仍为原`FBcprAuxModel`，权重SHA256=`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`，已重新计算匹配；不是换成另一个BFM checkpoint。z维数256、`norm_z=True`，非零z由`project_z(v)=16*v/||v||`归一化。**范数合法不保证行为稳定、技能语义或安全切换**；代码也没有自动验证这些性质。
- 原actor只读`state(64) + last_action(29) + history_actor(372)`，输出29D action；原适配器/PD在50Hz执行、MuJoCo200Hz。当前actor无需未来reference，亦不直接读取force；contact条件来自待学习的高层Student。backward编码器使用`state + privileged_state(463)`，目标姿态/全身特权信息只能用于离线指令或教师生成，不能塞进Student输入。
- 原PD命令映射仍是`q_PD=q_default+clip(action,-1,1)*scale`（原适配器先乘5再裁剪±5），不是任意29D关节位置权限。按已验证action contract，腰yaw数学命令范围约±21.01°、腰roll/pitch约±11.94°；右肩三个轴/肘/腕roll约±125.64°，腕pitch/yaw约±21.34°，这些项offset为0。**这是接口范围，不是各技能实测活动范围或机械安全证书**；不同关节的机械限位及真实闭环仍须单独核查。当前非站立原生候选的真实动作范围尚未知。
- `act(...,mean=True)`取均值；`sample_z`只是球面随机采样，不能称为随机抽取已验证技能。模型没有“输入技能名/自然语言即执行”的接口；下列名字先解析为reward函数，再用数据求z。

| 原生入口（当前源码） | 合法latent如何产生 | 时序与reference依赖 | 当前真实可用性/边界 |
|---|---|---|---|
| `goal_inference(next_obs)` | `project_z(backward_map(next_obs))` | 一个完整目标姿态观测可离线编码为固定z；不需要每步完整未来motion。原示例每100个50Hz步换goal，即约2秒 | API存在；verified standing已实际运行。不是Cartesian wrist/root目标解算器，也不保证精确到达某姿态 |
| `model.tracking_inference(next_obs)` | 对backward embedding作向前序列均值后归一化 | 当前checkpoint `seq_length=8`；在50Hz参考序列上最长约160ms的窗口。需要参考全身运动及其时序 | API存在；未发现本地已导出的tracking z与相同standing起点物理验收 |
| `humanoidverse/tracking_inference.py`实际示例 | **没有调用上述方法**；脚本内部另定义L=1的backward→project_z | 先取整条reference、用`obs[1:]`编码，每20ms换z；**重置到该motion的首帧qpos/qvel**再跟踪 | 不能把这个示例当作“从当前standing自然进入walking”，也不能把其50Hz序列直接当旧Student的10Hz held-z合同。存成z序列仍依赖离线reference与播放相位，不等于contact conditioning |
| `reward_inference` / `reward_wr_inference` | 对采样状态计算B，按reward加权求和后归一化；后者另乘`softmax(10r)` | 离线从数据推断一个固定z；执行时只需actor当前观测，不必提供未来reference | 所需`checkpoint/buffers/train_reduced`和`train`均不在当前下载资产中；也没有可直接加载的`reward_locomotion.pkl` |

goal目标通过原`get_backward_observation(...,velocity_multiplier=0)`生成，包含局部/heading规范化全身特征、关节状态及root高度；不能把world XYZ anchor直接当作这个API的输入。原goal示例可以从default reset循环切换姿态，但不提供本项目共同起点的安全认证；原tracking显式reference-reset更不能替代该认证。此次没有直接执行三个原生CLI：它们默认还会导出ONNX/写入model_folder或跑整段示例，并非本轮所需的有界只读盘点。

**现有技能/动作证据**

| 对象 | 已有证据 | 不能据此声称的能力 |
|---|---|---|
| verified standing z | #19原actor/PD闭环6秒，无力、无跌倒，min root约77.983cm，max tilt2.525°；之后各实验共同起点已复现 | 这条z来自default reset的`goal_inference`，**不是**已下载的`move-ego-0-0` reward latent |
| S0/S1局部接触响应 | 有真实小幅±X物理结果及边界；BFM由Contact Projector产生z驱动 | 它们是已训练的高层接触策略，不是原checkpoint自带的命名技能库。此次保留，不训练、不拿S0/S1冒充原生抬臂技能 |
| A/B/C全身goal行为 | [C/D结果](https://github.com/GaoE05/robot-research-bridge/issues/21#issuecomment-6516895613)确认goal在固定教师组合中带来独立收益 | A/B/C均叠加直接右臂修正；C不是已经验证的latent-only原生前伸技能。D只运行4.34秒并安全停止，缺完整hold/release |
| `raisearms-l-m` | 当前reward脚本真实列出、CPU解析为`ArmsReward`；左腕高度目标60–80cm，右腕≥100cm，同时鼓励站立和低运动速度 | 只是reward目标；没有对应z/rollout。它不约束“仅右臂”、不规定世界+X前伸，也不保证身体不动 |
| `move-ego-0-0.3` | 当前脚本真实列出、解析为`LocomotionReward`，目标身体朝向的前方COM速度0.3m/s | 目标速度不是实测速度；没有对应z/stand→步态进入和停止记录 |
| `move-arms-0-0.7-l-m` | 真实存在的移动+抬臂reward组合，目标0.7m/s和上述腕高 | 不是“已可安全前移且保持右腕接触”的技能；更不能因为名字组合成立就宣称接触跟随成功 |
| LaFAN参考motion / 原goal目录 | 40条30Hz、29DOF reference，时长约102.167–273.100秒；21个goal索引/名称均与数据相符 | 参考中有walk/run/dance/fight/jump/fall-get-up，不代表本checkpoint从任意stand都能稳定复现这些动作，更不代表已有contact触发转换 |

盘点检查了实际model-view及项目run/output目录：没有发现`reward_locomotion.pkl`、`goal_reaching.pkl`或`zs_*.pkl`；当前checkpoint目录只有model文件。完整/10s-clipped LaFAN均存在，完整数据SHA与历史一致；完整文件40条记录均没有policy action，**不是原reward脚本要求的replay transition buffer**，不能直接改个路径替代它。

38个脚本中的reward任务名全部通过实际CPU parser检查，包括locomotion、arm-height、move+arms、rotation/spin、crouch/sit；这验证了命令定义，不验证控制性能。额外注意：解析正则虽写了`h/x`，当前`REWARD_LIMITS`只定义`l/m`，不能擅自宣称`h/x`也可计算/执行。

**对现有goal候选的有限静态核查：** 按原helper的50Hz索引换算到30Hz原始motion时间，而不是误把JSON帧号当30Hz帧。21个目标均在界内，motion名与ID一致，`pose_aa`导出的29DOF与数据`dof`一致。相对已保存t3秒standing的参考关节RMS变化范围：右臂26.10–54.58°、左臂22.74–49.21°、腿11.05–62.02°；这些都是**参考姿态变化，非BFM输出或真实运动范围**。例如`dance1_subject3_4024`的右臂/左臂/腰/腿参考变化为40.01/44.27/1.74/11.05°，`walk2_subject1_2588`为42.61/30.11/13.35/16.54°。目录没有被验证为“局部右腕响应”的目标，也没有这21个z从共同stand起点进入/返回的记录。不能仅从名字或这些幅值挑一个“安全技能”宣布可用；也不据此断言这些goal必然不安全或BFM做不到。

**本轮验证结果与首个缺口**

本轮完成真实CPU资产哈希、源码合同、40条motion/21个goal元数据与38个reward解析检查；模型前向0、MuJoCo物理步0、策略更新0。检查了全部7张GPU，盘点时GPU1/5有空闲资源，故停止原因不是固定只看GPU5或无卡可用。**尚未建立一对任务相符、独立于直接关节残差、能从共同standing进入并返回的原生命令实例**；reward路线首先缺匹配checkpoint的latent或可重算buffer，goal/tracking路线首先缺上述技能语义及进入/退出证据。没有把缺失数据自动等同于能力缺失，也没有为“必须跑一点”去随机试goal、重新做IK或扩Student半径。

**有合格技能对以后，最小接触切换与学习方案（条件式设计，未执行）：**

1. 先把两个候选在相同standing机械状态和本体/动作历史下的进入、保持、回到standing资格测清，包括真实腕运动方向、肩/腰/root参与、支撑变化、命令/力矩限幅及退出连续性。资格依据是实际行为，不是reward名字。若候选只有抬腕而不产生所需前向局部让位，就不能把它硬套为右腕+X教师。
2. 任务只用一个standing motion、一个右腕接触方向：较小引导先局部响应；若前向接触持续且按关节本体可估计的臂伸展/局部响应不足，则允许已验证的body skill参与。用小/较大引导配对，并比较固定local、固定body、接触条件切换；固定相同伙伴规律但记录真实XYZ力。触发依据只能是force/proprio/history，不能用anchor位移、任务幅度标签或全局时钟。量化跟随/力、主动肩腰root贡献、足支撑与稳定，不能把被拖走算作学会转换。阈值和过渡时序要在资格数据后、正式对比前固定，当前没有给出假装可直接运行的参数。
3. Student保持小MLP和action-alignment思想；教师只输出经过验证的原生BFM命令，不叠加Jacobian/IK直接动作。数据必须覆盖local、body、进入/退出和无接触保持，保存10Hz决策对应的全部5个50Hz真实观测与29D action/q_PD，先做动作可表达检查再监督拟合。Student输入仅真实force和规范化state/last_action/history_actor；不输入特权pose、anchor、未来reference、技能标签或时间。技能/阶段标签可作为离线监督，不能当部署输入。预先留出onset和身体起点；离线成功后才做闭环，失败优先检查分布偏移，必要时一次小规模教师标注聚合，不回到PPO。
4. **输出接口必须显式作科研设计选择：** 旧S0/S1的standing附近0.3 tangent与0.2秒gate原样保留。若原生技能需要另一组合法z，未来新Student可考虑只输出两个已验证技能之间的选择/过渡坐标，而非机械放大旧球面邻域；其过渡latent和release必须先有物理证据，不能默认两个安全端点的插值也安全。停止阶段应由接触消失及身体状态因果触发，并按真实停止轨迹对齐监督，不能默认无力后0.2秒归standing适合所有身体技能。若教师采用50Hz tracking，旧10Hz held-z无法默认承接，必须先验证节拍匹配。当前没有实施这些接口变化，也不主张把已有尚未验证的motion序列包装为无reference部署。

**唯一推荐下一步：补齐并资格验证当前checkpoint的最小原生reward命令对，暂不启动Projector训练。** 优先为源码真实已有的`raisearms-l-m`和`move-ego-0-0.3`取得**同一权重SHA**下可追溯的latent（或其可重算的官方replay样本），再作一次共同stand起点的原actor/PD进入—保持—退出验证，先报告它们究竟是否形成局部/身体两类有用行为；若语义不符则不进入接触训练。不能直接借用另一checkpoint的同名z。本轮未下载资产、未对外联系作者、未执行这项下一步。选择reward路线的理由是其固定z执行不依赖完整未来motion，比目前整段reference-reset tracking更接近本项目的部署目标；这仍是候选路线，不是成功保证。

当前还未确认官方发布端是否能直接取得这两个同checkpoint latent或对应buffer；本轮只读目录元数据请求未成功（网络访问失败/超时），没有下载任何模型或数据。因此上述是明确的资产/复现资格验证步骤，不是承诺“补一个文件便已有可靠技能”；若官方没有匹配产物，需要确认其可复现推断输入，不应偷换其它checkpoint。

复现：个人分支`gaoe05/bfm-native-interface-audit`，基于`40aff65cf7b63dd218887eb100ab9b67d5e02cc9`；实际CPU盘点脚本commit=`fdaeb5b9a492472f81f8b8e52a55864c89a5344a`。关键新增仅`research_notes/native_skill_inventory.py`和本报告，原BFM/教师/共享源码零修改。P=`/mnt/data/temp_user/yi.gao/projects/compliance_BFM`；个人worktree=`P/bridge-v2-owned/trees/issue-21-native-interface/repo`；产物=`P/bridge-v2-owned/runs/issue21-native-interface-20261030A/{inventory.json,inventory.log,reward-parser-check.json}`，包含模型/源哈希、所有候选元数据和实际缺失路径。未改S0/S1、C/D及旧结果，不push、不运行Bridge。本轮到此停止。
