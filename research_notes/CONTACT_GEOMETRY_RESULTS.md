# Contact Geometry → Translation / Rotation — 有界阶段结果

2026-11-03，Issue37。**完成，全部实验退出；S0/T30继续保留，没有可靠的对称转向新基线。** 本轮推进了真实双点物理、共享contact-slot Student与运动来源定位，但没有把曲线位置跟随误写为身体旋转成功。

## 核心判断

1. 旧转向任务的末端切线小、heading代价小是真实因素；单腕曲线不能唯一约束身体朝向，实际左右路径也不等于相反的外力矩。扩大到明确的对称曲线、修改目标后，PPO仍未获得一致的heading收益，故不是单一reward系数已解释全部失败。
2. 单点表达接触点位置需求，身体朝向是另加软偏好；刚性torso双点通过一个共同SE(2)引导框架表达平移和yaw。两点不等于完整6D姿态约束；固定总5N也不意味着力臂/力矩/做功相等。
3. 主动双点平移成立；原生BFM可执行小幅左右转腰。可靠的接触驱动左右迈步转身或统一旋转反馈仍未成立。
4. 一个共享ideal-force Student已经同时执行旧右腕与躯干双点，但旧任务存在退化，双点组合的收益主要是平移，不能宣称学会了目标旋转，也未证明它充分利用力偶而非合力。
5. 值得保留：S0直行/停止、T30、共享Student的双点平移研究版本、原生转腰行为源。不能推广：A/B PPO作为全局转向Teacher、goal监督初始化作为保旧专家。

**唯一下一建议：** 以已经有物理证据的小幅原生转腰为新行为源，针对“加入旋转后接触前就偏离原平衡/运动分布”的明确问题，做保留无接触稳定域与旧运动行为的同状态动作约束及有限真实访问状态聚合；先形成保旧的左右旋转反馈专家，再迁移到现有共享contact slots。不增加force cap，不先扩更多点或大角度迈步，不再无界扫描reward latent。本轮没有自动执行该后续训练。

## 数据、模型与执行合同

完整服务器根：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/contact-geometry-20261103/`。
小型可审查结果：[geometry_artifacts/results.json](geometry_artifacts/results.json)、[模型身份](geometry_artifacts/models.json)、[资源与全部失败](geometry_artifacts/resources.json)、[学习曲线](geometry_artifacts/learning-curves.png)。原始qpos、决策、观测、标签、权重保留服务器，不进入Git。

|角色|实际runtime commit|输出|
|---|---|---|
|双点初始资格|eb5eb03eb4ab1a167647ee2e3083ffd6340dae66|B-prepare-02|
|单腕A/双点B Direct PPO|b96fe456fb77d3d98602772d98d23652d0613c42|A-train-04、B-train-05|
|纠正时序的motion正控制|6be319c3cd3f696c2aa2506a899477c1413d90fe|C-prepare-07|
|共享Student训练及评估|596b598749db82d17b2bc97e08dec76d013095c1|C-final-08|
|原生goal正控制|72dc8937c644446f35c7beb600f577b5751b0dab|C-goal-09|
|goal监督Direct初始化|704de7e00736cc6845222672c17cc6e5350d8770|C-goal-10|
|冻结最终评估|940377c1fb3271abb7ca3ec5aae2f04a70e9c422|A-final-11、C-final-12、B-final-14|
|真实qpos渲染|87ed6cd6801ecccfdebe84680e08a10363fd3acb|videos-attempt02|

- BFM SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`，冻结原actor/PD/权重；所有有效物理run均记录`original_bfm_unchanged=true`。
- T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，旧`stop-sensing-20261103/B1-train-02/job/best.pt`。
- S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`，旧`direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt`。
- A20 `189e2fa24b6a2fdb3cecb8c14fb4f51071334d27ab487fab73e81248e3fbd610`，A-train-04/job/best.pt。
- B20 `62004810623df6ef8d5164ed04d3793cec5998fc57c6486012a5b5abdfc0fce8`，B-train-05/job/best.pt。
- 共享Student `c055315215615f496f86f22fbbf7df5fbc20f0dc119892cf01cbe03c29157001`，C-final-08/job/shared-r0.pt。
- goal监督Direct `2059302f4b72c204b020309f316d17d46dd8cb7718d1ceaa655185eca592941e`，C-goal-10/job/goal-initialized.pt；**未推广**。

BFM50Hz，原PD/物理200Hz；高层10Hz，保留原5个actor tick的合法z过渡。没有在线M10 Student、CyclicFeedback或新关节控制器。名义motion资产只作为既有能力诊断来源。

双点在同一`torso_link`刚体，局部位置`[0,+0.13,0.15]`和`[0,-0.13,0.15]`米，不是上臂。K80/D8，各2.5N；单腕5N。每tick先计算各点速度与力，在body惯性COM处累加`[sum F, sum((p-xipos)×F)]`一次，原finally减去同一wrench，保留外部既有wrench。每tick断言总force-norm≤5及减除正确。诊断yaw moment统一关于机器人整体subtree COM；做功`sum Fi·vi dt`来自同tick实际点速度。真实施力仍是双向保持接触Tether。

Teacher1088维：保留T30旧825维语义前缀，加逐点短历史/固定mount编码/当前特权误差速度/当前朝向误差。旧前缀有合力摘要，但新增逐点分布未丢弃。T30扩展时新权重为零，是**跨接触几何迁移执行基线**，不是已针对力偶训练的专家。
Student1639维：465本体/历史＋256自身已执行z＋100×3槽位×3轴力历史＋3×6 mount编码。每个历史样本保留原slot/link/局部点，世界力在读取时旋至当前pelvis系。mount-present是已知连接装置，不是active真值。无anchor、伙伴速度、目标yaw、时钟、实验ID或Teacher状态。实际G1力传感配置仍未知，不能称真机部署已成立。

## 旧任务和reward审计

源码重算旧新左切线11.769°、同前缀改向5.833°。旧heading项`.1*motion_weight*(1-cos(error))`，T30左条件平均heading cost0.000478、force cost0.389；右0.000076/0.603。它确实弱，但这不是唯一根因证明。

旧T30轨迹的平均Fy/关于COM的平均yaw moment：新左−1.230N/+0.037Nm，新右−2.905N/−0.198Nm；同前缀左−1.481N/−0.067Nm、右−2.084N/−0.135Nm。左右名字并不代表相反力或力矩。旧做功由保存点位置差分估算，力/位置存在≤5ms时序偏差；新日志改为同tick功率。完整审计见old-geometry-audit.json。

新开发单腕曲率±0.55、82cm，总转角±25.84°；封存±0.48即±22.55°。双点开发±15°、20cm平移、30cm/+12°组合；封存±18°及26cm/−10°组合。封存onset4.4/ramp8.1；新Student纯平移24cm/onset4.45/ramp8.05。此前暴露条件仅为开发，不冒充新盲测。

reward逐点force使用`1.5*n*sum||Fi||²/25`，不惩罚抵消后的合力；两点均分满预算与单点满预算归一量相同。误差、相对速度、限幅前需求也逐点计算。单腕heading平方误差尺度20°、权重0.35×距离因子；双点0.5×(0.8 torso＋0.2 pelvis)平方误差，接触保持期仍有效，不按近零速度关闭。其他稳定/连续性/终止惩罚保留。参数在新训练前固定。新RL项在`rl_stats`；旧控制日志显式命名`legacy_reward_terms_not_used`，不可误当优化目标。

指标定义：接触误差为`100*sqrt(mean_active_time(mean_i ||anchor_i-point_i||²))`厘米；sum-force RMS为`sqrt(mean_active_time((sum_i ||Fi||)²))`，不等于合力RMS。所有作用点真实XYZ及符号偏差保留。表中“实际转角”取运动段末相对3秒ready状态；不是释放后的最终朝向。

## 两组有限Direct PPO：位置收益不等于heading成功

A为单腕曲线/直行/局部域，B为双点旋转/平移/组合及直行/局部域；它们不是同任务算法消融。T30同初始化、新优化器、每组40×768决策；3worker、std0.008、actor LR1e-5、解析joint KL限0.02，未调force cap。每5批多条件开发评估，用最大相对成本＋平均成本选择，A/B都选20；保留初始/最佳/末轮，终检不重新选模型。A开发score1.800→1.788，B1.857→1.794，末轮均退化。新任务/reward/input共同变化，不把结果单独归因某一权重。

下表每格为**接触3D误差cm / sum-force-norm RMS N / pelvis heading RMS°**；单点时sum-force就是force norm。

|未见或回归条件|冻结T30|A20单腕训练|B20双点训练|
|---|---|---|---|
|新左22.55°曲线|8.682 /4.249 /16.89|6.629 /4.141 /18.60|6.555 /3.642 /23.37|
|新右−22.55°曲线|8.302 /4.441 /9.94|7.963 /4.586 /11.84|9.043 /4.644 /10.14|
|同前缀后左改向|5.715 /3.480 /14.70|5.742 /3.556 /13.33|6.048 /3.445 /16.76|
|同前缀后右改向|6.167 /3.841 /4.91|6.544 /4.058 /5.46|6.465 /3.756 /4.34|
|原82cm直行|5.826 /3.599 /7.87|5.876 /3.597 /5.86|6.159 /3.528 /7.74|
|15cm慢局部请求|4.121 /3.107|2.563 /2.108|2.826 /2.235|

此处15cm/ramp7.8是本阶段慢局部请求；旧ramp4.5局部回归另列Student表，不能混用。

双点封存+18°：T30/A/B误差2.981/2.977/3.022cm、sum-force4.000/4.000/4.009N，torso实际转角−0.32/−0.27/−0.61°。−18°：2.021/2.022/1.972cm、3.261/3.262/3.180N，torso−5.39/−5.37/−5.64°；Standing本身已有约−6.19°，不能把这当学会主动右转。纯旋转未改善，PPO局部收益不足以替换T30。

## 原生能力与学习瓶颈的对照

先纠正旧候选调用：左右motion的period不同（1.24/1.52秒），按各自period和原进入3–4.5、执行4.5–10.5、退出10.5–12时序重测。当前M10 `task`变体（含既有手臂/腰部调整，不冒称未经修改原motion）在双点纯旋转负载下：左9.863cm/4.977N、torso反向−13.87°；右61.037cm/4.983N并前移约78cm。它们不是可靠原地转向源；无负载两候选最终yaw也同为负向。初次period错误结果保留，不能将其当底座能力证据。

随后仅做两个明确、合法的原生`goal_inference`腰yaw目标±15°，同standing完整起点、原PD、同Tether，**无IK/关节动作覆盖**。这是限定的表达正控制，不是最终接触反馈策略：

|±15°开发|Standing|T30迁移基线|原生goal实际执行|goal监督Direct|
|---|---|---|---|---|
|左：误差cm / sum-force N|2.664 /3.900|2.533 /3.841|1.576 /2.535|2.511 /3.847|
|右：误差cm / sum-force N|1.501 /2.423|1.641 /2.648|0.678 /1.089|1.022 /1.615|
|torso相对ready转角左/右|−1.23/−5.59°|−0.42/−4.80°|+9.23/−14.52°|+2.37/−11.75°|

goal主要是腰部相对pelvis约±10°旋转，足部总路程约0.4–0.6cm，没有建立新的支撑转移/迈步转身。无外力goal对照也产生相应转腰，因此不是外力被动扭出来的；但它本身是目标指令正控制，不能声称只凭接触完成选择。

在剩余预算内用2条goal真实轨迹＋6条旧T30保留轨迹、1288标签、4000监督更新初始化同Direct网络，执行仅当前特权观测→z。新±12°/onset4.65/ramp6.4事先固定、不选checkpoint：左goal/网络1.495/2.244cm、2.409/3.600N；右0.703/1.072cm、1.137/1.696N。右向部分学到，左向不足。

更重要的失败：该网络旧直行5.826→53.310cm、平移4.998→12.385cm，未推广。保存专家状态上BFM q_PD平均RMS差只有约0.044–0.078°（抽查128级样本），却不能保证闭环保旧。直行q_PD差>1°持续100ms首次在3.76s，实际关节在3.955s，**早于4.25s接触**；root差>1cm到5.45s。阈值仅是定位标记，不是新成功门槛。两条闭环分离后不再同状态，故证据支持初始稳定域/闭环分布敏感，不能唯一归因网络或样本数量。完整first-divergence与alignment保存。

## 共享ideal-force Student：已实际训练与闭环

预先选择T30已有效的双点平移/组合域；旧腕保留S0自己真实执行的标签，不把较差Teacher标签强加给S0。8个独立回合、1227标签、4000 episode-balanced更新、seed1307；从S0精确输入映射初始化（腕部raw输出parity<1e-5）。同权重z直接监督并检查BFM动作一致性，不做恒等投影优化。

|条件|S0或Standing|T30|共享Student|
|---|---|---|---|
|旧正常直行|S0 6.530 /3.698|—|7.280 /3.764|
|旧早停止|S0 7.330 /3.745|—|7.719 /3.772|
|旧晚停止|S0 6.892 /3.852|—|7.646 /4.183|
|旧局部15cm/ramp4.5|S0 3.680 /2.611|—|4.445 /3.216|
|新双点24cm平移|Standing14.761 /4.556|3.960 /3.333|4.740 /3.495|
|新26cm/−10°组合|Standing16.155 /4.583|4.718 /4.226|4.178 /4.018|

数值为cm/N。共享策略有真实双点任务收益，原腕仍能执行，但旧能力退化，**不替换S0**。新组合Student的torso实际转角约+0.66°，目标−10°，所以主要收益来自平移，不是旋转合格。不能仅看误差4.18cm就宣布转向成功。

同状态接管4次：两个Student回合在7/11s接管前qpos/qvel/force逐值完全一致。平移7s接管全接触误差6.375→4.026cm、sum-force4.142→3.314N；11s收益很小。组合7s误差4.530→4.007cm但force3.902→4.244N；11s两项均退化。接管前相同，不代表接管后所有标签有效，**本轮未盲目聚合这些混合质量查询**。

## 逐点、全身与恢复质量

新组合Student两点平均Fxyz分别`(+1.238,−0.369,+0.342)`、`(+0.015,−0.072,+0.727)`N；每点力RMS2.219/1.872N，触各自2.5N上限比例67.95%/35.81%。合力RMS3.694N不等于sum-force RMS4.018N；净yaw moment均值−0.136Nm、RMS0.195Nm；外力总做功−0.327J。新平移外力做功−0.736J，Student root前移25.30cm，Standing仅2.59cm，支持主动响应而非纯被动拖动，但不把负功单独当完整因果证明。

新组合Student root接触段位移`(+24.88,+1.11,−0.75)`cm；T30为`(+21.63,+4.76,−5.35)`cm。Student最小root高0.744m、tilt峰8.08°，没有记录到双足同时无支撑、关节超限或力矩饱和。足部支撑状态变化8次；滑移速度p95约0.0104m/s、峰0.877m/s，峰值仍保留，不能宣称无滑移。新平移支撑变化10次，滑移p950.0111m/s、峰0.906m/s。

新组合q_PD最大20ms单关节变化32.17°（左踝pitch、13.26s），真实关节最大5ms变化1.61°；新平移为41.13°（右踝pitch、7.6s）、真实5ms1.52°。这是不同量、不同节拍；不把目标跳变误写成实际关节瞬时转动，也未重复无依据平滑或改共享PD。

组合释放后5秒Student root速度RMS0.0103m/s，位移`(+0.24,+0.00,+0.84)`cm，T30为0.0312m/s及`(+2.41,−2.62,+5.40)`cm。新平移Student恢复RMS0.0340m/s，允许合理收步。释放后anchor距离不计任务失败；无接触腕/双点对照只有毫米级root漂移，无完整新技能自动播放。所有分期XYZ、腰/肩/手/root、足部、目标与实际关节见metrics。

## 工程修复、预算与复现

保留失败：reset重建Tether/physics callback导致首个缺日志尝试失败；修复为每reset重装。A初启开发no-contact成本0使相对score不合法，在0更新时主动终止并加0.1分母保护；不是物理/学习失败。C默认train_C缺失在资格运行前报错，改显式初始化。旧right motion period调用错误单列，不隐去。首轮视频maintenance无XML资源失败，后显式只读使用已有个人runtime资产，未下载。离线分析误把标签列表当轨迹已过滤；物理结果不受影响。

实际累计受监督实验/渲染 **8990.33 job秒**（失败也计入），输出约**4.29GiB**；最多GPU1/2/3三张，启动前CPU/RAM/I/O实时核验。GPU0/5/6他人任务未操作。A/B各30720高层RL决策，合计61440；训练物理1226016步、actor/PD306504步，另standing prefill207600物理步（评估另计）。新增Expert监督4000、Student监督4000，非RL样本。两组PPO均完整40批，保留全部开发候选/初始/最佳/末轮，无按终检选择。剩余预算不自动发起下一研究。

最小复现：只读打开protocol、models/resources及具体run/launch，使用记录的个人runtime和现有项目`.venv/bin/python`；再次执行物理须重新取得阶段授权/资源快照。`geometry_final_supervisor.py --stage final --arm A|B|C --kind direct --gpu-index <空闲卡> --iterations 0 --seconds 1200 [--resume <已核SHA模型>]`对应已封存比较，不启动Bridge。源码角色：geometry_contact物理、geometry_env逐点输入/reward、geometry_train PPO、geometry_student共享监督、geometry_goal/goal_teacher原生表达/初始化、geometry_analyze逐点分期、geometry_render真实qpos视频。

## 本地视频与下一步

Mac已交付且验证：`~/Projects/compliance-bfm-diagnostics/videos/contact-geometry-20261103/`，仅3条MP4＋INDEX。
1. 单腕新左弧T30/A20：位置改善，heading仍差。
2. 双点未见组合T30/共享Student：平移收益与旋转未完成同时展示。
3. 原生左转腰/监督Direct：能力源有效，但学习承接不足的失败。

1×、实际qpos、完整接触与5秒恢复、固定世界＋俯视；显示每点force/anchor、净yaw moment、目标/实际朝向。SHA/25fps/时长/完整解码及画面已检查。模型与数据仍在服务器，Mac不堆原始日志。

本轮的能力边界不是“必须更大力”或“BFM完全不会转”，而是**已有有限转腰行为如何在Direct闭环中保住稳定域与旧移动能力，并进一步由受限接触观测可靠调用**。尚无对称迈步转身、任意多部位同时接触或真机证据。长期路线保持单/双点平面→多部位分别/组合→高度与移动→未见接触配置/身体状态泛化。
