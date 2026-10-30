# BFM全身技能来源阶段：原生goal身体前伸资格

**阶段决策：采用已验证的原生goal身体前伸作为下一阶段的有限姿态转换动作来源，暂不依赖当前locomotion z，也不继续手工关节控制器校准。** 本轮新证据是去掉所有直接关节修正后，原BFM仍能主动协调肩/腰/腿、从完整standing进入并返回，并在大幅前引接触中明显减少三维跟随误差。它可作为一个可执行的身体姿态primitive，尚不是精准、低力、鲁棒的接触跟随教师。下一步应检验接触触发的进入/退出，而非立即训练Student。

## 来源核查与取舍

读取#18 v3.6及#21最新部署审计，刷新团队origin。共享checkout保持`6023c6e`；团队main为`c002f394f0b77d2f8a85356667fec5eda0f4a1af`，新增累计latent策略配置/实现，未整合进本实验。团队REPORT与WHOLE_BODY_RUN_REPORT只给出冒烟与启动证据，明确没有有意义的学习收益；ACCUMULATING_TRAINING描述新更新语义，不是可靠全身技能资格报告。

核查了作者公开issue26全部现有答复、团队受控源码/报告、个人pretrained/native_reward_assets，以及此前保存的HF不可变revision清单。没有获得独立已验证的同权重locomotion z、完整初始历史/PD与物理轨迹包；不代表作者或团队私下没有。HF API本轮访问失败，不能宣称重新查全发布目录。没有进入其他用户私人目录、联系他人、随机抽z或下载新大资产。来源及范围存于`skill_source_sources.json`和run/sources。

|来源|真实物理证据|身体参与|接触适用性|首要边界|本阶段取舍|
|---|---|---|---|---|---|
|当前reward移动z|同权重6秒进入/执行/退出，部署复现通过|真实root/支撑转换|方向/连续性不足|侧漂，常z q_PD跳变；最终限位未解决|保留为能力证据，暂不用作定向教师|
|raisearms-l-m|原生左右腕高度语义100%，稳定返回|手臂为主，但有腰/身体变化|更适合垂直抬腕方向，尚未做接触资格|不是纯右臂或水平前引技能|保留，不硬套+X任务|
|A/B/C组合goal教师|C保持误差0.884cm，因果D支持全身goal价值|明显身体参与|当前任务精度最好|C依赖较大直接关节修正，旧受限latent匹配未通过|不作为latent-only教师标签来源|
|本轮纯原生goal末端|新增无力+接触有限资格，3×6秒|肩前移、腰/腿额外指令，双足承重|接触期误差/力有收益|保持误差6.45cm、力仍触5N上限|推荐为有限身体姿态primitive|
|团队PPO / tracking资产|报告为冒烟/启动；reference资源存在|未获得当前匹配的可靠同起点轨迹|尚未建立可复现证据|不能把启动、motion名字、reference-reset当技能资格|本轮不执行、不作为替代成功证据|

## 固定候选与运行合同

模型SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。直接取旧A/B审计`same-state-forward.json`末端index40保存姿态，使用原`goal_inference`重新编码，和旧native z逐值差0；范数16，距standing66.9587°。来源文件SHA `e1c0dae3356f566a4769c473c89e1eeeffa0a035a1fd0ed80ea77ac45e7af85d`。目标姿态最初来自以前的几何教师，本轮没有重新求IK、改目标或追加优化；它不是作者发布的命名动作，也没有把参考姿态当成已执行的运动。

使用原`act(obs,z)`→29D action→原PD，actor50Hz、原MuJoCo200Hz。只在已有standing与该native goal之间使用已经采用过的norm16球面插值，没有原锚定差分，也没有Jacobian/关节残差、target clipping或关节平滑器。模型仍根据当前本体/动作历史闭环。原权重、共享代码与S0/S1不变。

运行前固定三分支、原物理停止条件及资格筛选。无力技能先执行：0.75s进入、2.5s保持、0.75s返回、2s恢复。预先要求完整稳定回合、保持平均腕前移≥5cm、肩前移≥3cm、躯干姿态变化≥5°、同状态额外腰q_PD RMS≥1°，最终root速度≤0.02m/s、最后1秒q_PD步长≤2°。这是本实例的实用筛选，不是全身任务必要性的证明。

通过后执行standing_contact与goal_contact。两组同一完整t=3s standing机械状态/历史、原右腕+X18.6819cm几何任务、4s smoothstep移动、1s保持、t8释放、t9结束；K80/D8、5N cap、seed0。goal使用单一native endpoint的球面进度，与伙伴移动阶段配对，释放沿同一路径0.75s返回；没有每步姿态IK或关节反馈修正。该进度使用特权任务时序，是教师资格实验，**尚未实现根据真实接触自主触发**。

三个共同起点/3秒前缀全部精确复现；新standing接触分支与旧几何任务baseline的qpos/qvel/root/wrist/force/action全程差0。新旧共享资产身份没有混用。固定状态指令预检最大步长1.0082°、最大raw action绝对值0.4940，仅为有限预检，最终判断来自下面真实物理。

## 无力技能：独立于接触和关节补偿的身体协调

保持段腕平均ΔXYZ=[11.382,5.773,−3.398]cm，肩平均ΔXYZ=[8.145,−2.644,−5.298]cm，腕相对root平均ΔXYZ=[13.935,7.205,−1.665]cm；保持末rootΔXYZ=[−2.392,−1.481,−1.817]cm。躯干姿态峰值28.19°（含yaw），腰yaw/roll/pitch最大变化[16.35,6.41,12.78]°。

同一真实观测下，相对standing latent的额外腿/腰/右臂q_PD RMS为4.964/3.627/8.883°。外力全0，仍出现显著肩/腰运动，支持主动身体协调；不是被外力拖动。不能据此声称迈步：全回合双足均有>1N正载荷，没有支撑转换。

无力全回合最低root76.221cm、pelvis tilt最大8.181°；进入/保持/退出/恢复最大20ms q_PD变化为1.221/1.774/1.445/0.757°，实际关节20ms最大变化1.158°。最终root速度0.000746m/s，最后1秒q_PD最大0.0324°，最终腕ΔXYZ=[0.218,−0.189,0.852]cm。原预定资格全部通过。

## 接触真实对照

真实接触阶段按Tether.active统计，释放后的零力不算跟随收益。移动/保持/释放分别799/200/199个200Hz样本；另有一个实际active的浮点时钟边界样本纳入全接触汇总，一个最初未active样本不计接触收益。两分支使用完全相同规则；旧baseline逐帧相同，少量汇总末位变化来自明确的active分段规则。

|控制|移动3D误差 cm|保持3D误差 cm|移动/保持力RMS N|保持末肩/root ΔX cm|稳定性|
|---|---:|---:|---:|---:|---|
|原standing|8.926|15.084|4.083 / 5.000|2.210 / 1.791|完整、双足承重、未跌倒|
|纯native goal|4.315|6.447|3.489 / 5.000|9.768 / −1.526|完整、双足承重、未跌倒|

移动/保持误差分别降低51.66%/57.26%；全接触3D RMS由10.459降至4.820cm（−53.92%），全接触力RMS由4.283降至3.841N（−10.33%）。但保持力均为5N、触cap比例100%；移动触cap由54.44%降至18.27%。不能只报告误差降低而说实现低力跟随。

保持XYZ误差RMS：standing=[15.033,0.476,1.148]cm，goal=[4.851,3.684,2.115]cm。保持真实平均力XYZ分别[4.983,−0.159,−0.383]N与[3.748,−2.861,1.663]N；目标标签+X不代表实际纯X受力。

goal保持末腕ΔXYZ=[13.946,3.600,−2.127]cm，肩=[9.768,−3.763,−5.513]cm，root=[−1.526,−2.598,−1.735]cm。保持段腕相对root平均ΔXYZ=[15.518,6.013,−0.406]cm；同状态额外腿/腰q_PD RMS4.847/3.649°。躯干最大变化26.98°、腰最大[16.15,5.62,12.61]°，支持真实身体参与，但仍不是完成固定肩不可达目标的证明：原hold<1cm判据未过。

goal接触全回合最低root76.299cm、最大pelvis tilt8.586°，两分支全程双足>1N正载荷，无实际关节越限、非足地面接触、action/torque饱和或原环境失败。goal移动/保持/释放q_PD最大变化1.223/0.195/1.504°，实际关节20ms最大1.121°。足接触切向速度峰值0.279cm/s。

**释放边界不能省略：** 本接触6秒回合仅含1秒释放，其中0.75秒回到standing命令，最终root速度仍0.03442m/s（standing为0.00506）；最后腕离起点[2.714,−0.070,1.062]cm。因此只能说已安全开始返回，不能说接触释放后已完全停稳。无力实验有2秒恢复并已停稳，不能替代接触后的更长恢复证据。未为了得到更好终值延长/重跑这次固定回合。

## 能力边界、接口与唯一下一任务

首个不可跨过的现有边界是：**这个native goal具有可执行的躯干前伸能力，但会伴随Y/Z运动；在原大幅纯+X任务上仍有6.45cm保持误差和5N持续载荷，原1cm成功判据失败。** 当前不能称已得到精准低力接触教师，亦未验证迈步或多起点鲁棒性。接触结束的停稳证据还有上述时间窗边界。解决这些问题若再靠追加关节控制，就回到了本阶段明确放弃的支线。

最适合后续Contact Projector的动作通道是原**256D norm16 native-z→冻结BFM→原PD**，将新问题限制在这条已验证standing↔goal球面技能路径上。用一个有界技能进度α∈[0,1]参数化合法z，可以先学习进入/保持/退出，避免先在整个球面盲搜。教师实际action均直接来自同一个BFM，因而在该命令路径上可表达；仍应保留action/q_PD对齐以及更简单的latent/进度蒸馏对照。

这不是把旧Student半径偷偷放大：本goal距standing66.96°，不在旧约16.7°输出邻域内。只凭角度也不能证明旧邻域内不存在等效动作。本轮没有改变S0/S1或Student输出定义。后续若采用α路径，应作为新的显式输出合同立项。actor仍50Hz；当前资格中进度也50Hz变化，不能直接宣称旧10Hz held-z已等效。应保持已验证命令节拍采集/训练，或先单独验证10Hz决策与50Hz过渡的合同。Student输入仍只能是force/proprio/action history；目标姿态、anchor、任务时钟和未来轨迹只用于教师生成。

**唯一下一任务建议：真实接触触发的standing→有支撑躯干前伸→standing。** 固定这两个latent端点与现有过渡，不再优化reward或IK；只检验何时进入/退出。先做有限的接触门控教师资格，再决定是否监督训练。

建议预先固定原大幅+X伙伴轨迹及5N cap；接触onset使用两个开发时刻和一个独立未见时刻，并加入无接触与短脉冲对照。比较standing、固定时钟回放和只读真实force/proprio/history的接触触发。门控规则在运行前固定一次，不按结果反复调。释放后统一留2秒评估恢复。可证伪标准：未见onset下相对standing全接触3D RMS至少降低30%、全接触力RMS至少降低5%，同时没有无接触误触发、没有跌倒/支撑丢失，2秒恢复后root速度≤0.02m/s；报告反向力、保持力和XYZ代价，即使总体指标通过也不抹去原hold<1cm失败。若这些条件不能同时达到，就否定这条路径作为本任务教师，停止该支线；不靠加关节补偿或放大训练量救结果。这是有限姿态转换任务，不把固定双足称为步态转换。

## 版本、资源与复现

本阶段第一次且唯一数值启动完成三组6秒分支及三个3秒前缀，记录物理共27秒（不含构造/reset），Student/PPO更新0。GPU0（RTX PRO5000 72GB）、原BFM CUDA，原MuJoCo CPU；wall131.402s，采样owned VRAM峰582MiB，aggregate RSS峰7.15GiB，低于预定3600s/4GiB显存/64GiB内存/8CPU范围。owned PID54167及子进程已退出，没有停止他人任务。

唯一准备期工程失败是从共享checkout创建私人commit worktree时该object不在共享对象库，未启动模型；改从包含该commit的私人repo创建worktree。失败和修复记录保留，不改共享Git历史。

个人分支`gaoe05/bfm-skill-source-qualification`，基于`59fee50ae4cc9ff817d0151ced34eb3e244056a4`；实际运行commit `17d80d5e30d3c77bd9d769d409e1ec0da5d3162d`。三项配置/时序/隔离测试通过，AST/JSON与git diff检查通过；源文件/权重hash、GPU模型前后tensor及参数版本均保持，goal重编码、完整初始状态和baseline复现均通过。

`P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`：
- 代码`P/bridge-v2-owned/trees/issue-21-skill-source/repo/research_notes/skill_source_*`、`test_skill_source.py`。
- 数据`P/bridge-v2-owned/runs/issue21-skill-source-20261030A/`；全轨迹`physics-attempt-01/outputs/{goal_free,standing_contact,goal_contact}.json`。
- 指令身份`native-command.json`、资格`basic-skill-qualification.json`、基线`old-contact-baseline-parity.json`（均在outputs）；指标`analysis/physical-summary.json`。
- 来源` sources/`及`analysis/skill_source_sources.json`；预算/失败`compute-ledger.json`、`preparation-failure.json`；最终commit与清理结果见`stage-final.json`。
- Mac小文件`~/Projects/compliance-bfm-diagnostics/issue21-skill-source/`。本轮未下载新大资产、未push或启动Bridge。原结果和权重保留，阶段结束停止。
