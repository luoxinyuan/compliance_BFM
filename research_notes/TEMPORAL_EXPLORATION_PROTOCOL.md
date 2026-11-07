# Issue49：原生动态运动时间连续探索（正式采样前预登记）

2026-11-07。[Issue49](https://github.com/GaoE05/robot-research-bridge/issues/49)承接已结项Issue48，不重启其任务、不修改其best/终检结论。当前阶段从准备、资格/概率验证再到正式训练，实时状态见CURRENT_STATE与私有ledger。用户提示本身不代表实验已经运行。

## 固定基础、候选与预算

Pro5000 GPU0，最多一个受监督实验job、3个单线程MuJoCo worker；不动GPU1–6及他人任务，不用PPU。先核查CPU/RAM/IO/磁盘再启动每个GPU job，禁止CPU替代。新硬上限：18000累计experiment job秒、12GiB新增输出、98304真实高层RL决策；其中Student/C0训练0。worker含在所属job内不二次叠加，失败/资格/概率/评估/视频计时。当前仅CPU准备，启动后记录精确PID/start_ticks/输出/commit。

只复用同BFM站立、goal5/15、lean与motion资产；原reward移动负例不重新推广。最多3个新增原生motion物理正控制：原full-rate的72cm/8s低速接触、同reference half-rate自由执行、同half-rate接触；包含1.5s进入、持续及1.5s回standing、完整释放。full-rate自由行走复用Issue48保存证据。half-rate只是有限能力诊断，不因结果好就静默改一臂训练源/协方差；不扫描新seed、goal、reward或phase gain。时钟native控制仅资格参考，不是在线Expert、监督答案或on-policy样本。

新A/B均从Issue48原生初始化`d399174cc2ddbd3efb858953653c9134495070948341381565f5f918f58da5cd`开始，不用C32作为初始化，更不继承I/C0/T30。它们均为冻结参照。原生441时刻/7回合BC背景保留，不重复BC；fresh阶段critic/Adam，两个seed4901/4902在配对内一致。BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`，原PD/256D Direct/10Hz高层/50Hz actor/200Hz physics/K80D8/10N主与5N回归不变；网络、normalizer、reward、力历史和课程均不变。两臂都使用Issue48 C的可逆均值head坐标与**同一**满秩raw协方差`Sigma=.008²I+.15²DᵀD`，同4方向balanced-v2。不是新raw-vs-coordinate或方差能量比较。

## 实际动作、条件概率与梯度

Actor网络仍1088→128→128→256，输出均值，参数数目与Issue48 C相同。定义白化均值`mu_theta(x)`，raw=`L*u`一次后原norm16合法投影；L固定可逆。A rho=0，B rho=exp(-.1/.5)，不扫描rho。

记录实际上一命令`u_(t-1)`、上一真实观测`x_(t-1)`：

`p_theta(u_t | actual history) = Normal(mu_theta(x_t)+rho*(u_(t-1)-mu_theta(x_(t-1))), sqrt(1-rho²) I)`。

**更新时两个mu都用当前候选theta重新计算并保留梯度。** 若缓存旧residual当常数，会漏掉上一mu的梯度，不是该历史条件策略的正确似然。PPO replay wrapper `[current1088, previous1088, previous_u256, valid1]`共2433D只携带真实因果记忆/概率上下文；不是把新特权/任务ID加入actor。Actor与critic实际网络仍读原1088D。critic保留原current-observation近似，不额外读取精确AR residual；这是共同网络设计下B价值逼近的已知限制，不隐瞒为完整噪声状态critic。

每个episode、每个冻结policy-version采样batch的首决策，rho置0，直接从stationary N(mu,I)启动；只重置探索记忆，不清空身体、solver、BFM/动作/last-z或真实接触历史。source prefix是完整真实执行。policy-version边界最长每worker256决策，远长于本次0.5s相关时间。延长若发生，从实际last checkpoint恢复actor/critic/Adam和torch RNG，物理worker通过新真实源前缀重建，不声称接续旧实时MuJoCo；source RNG按预定offset1600000更新，两臂相同，避免重复首段source随机序列。

固定theta的标准化residual满足`e_t=rho e_(t-1)+sqrt(1-rho²)xi_t`，首项stationary N(0,I)，因此**单步边际探索噪声**协方差相同Σ；B给定历史的条件创新协方差更小。不能声称内生机器人状态不同之后，实际动作总体协方差或同状态全历史条件协方差也相同。实际力仍每200Hz由物理重算。

完整保存u、raw、当前/上一obs、上一u、conditional mean/std、noise、innovation、valid/reset、version、time与真实轨迹。oldlogprob对应实际conditional分布，不能用独立Gaussian补标。原3e-4联合logprob/.02 joint KL/clip/Adam回滚不放宽。先做NVIDIA FP32 batchone/batched、两步联合密度分解/梯度、head梯度`(1-rho)/.008`、stationary协方差/lag1，以及作废clone原PPO更新；不是离线成功即闭环成功。正式首batch仍受原断言约束。

确定性开发/终检只执行raw_mean(current)；不读取rho/noise来选技能，没有在线原生控制器覆盖。当前仍1088D特权Teacher，能看当前伙伴误差/相对速度/active；不是force-only、受限观测Student或真机感知。

## 课程、选择、延长与停止

共同沿用Issue48 native_env的standing/reach/walking真实源前缀与零/无需求、近中远、停止/继续、moving需求改变课程。首次每组16×768，两个seed共4组49152决策；每8批开发。纯回合源编号、变化时刻、最终距离与未来不进actor。源执行阶段不冒充当前policy的RL数据。

新开发分数使用已有物理cost的**绝对值**`mean(cost)+.25max(cost)`，不除接近零的初始cost。训练reward不变。严重退化保护使用物理单位：无接触root max(5cm, initial+3cm)、torso initial+10deg；近/中/arm误差允许initial+max(2cm,.2initial)，其他active initial+max(5cm,.15initial)；force initial+1N、限幅+15百分点，以及原物理安全边界。它们是停止/回滚保护容限，不是宣称成功必须达到的统一数值。initial包括在选择中，完整输出各族/最差/分段，不将平均收益抹去代价。

4组首段全部结束，只有**同一探索方法在两个seed**均存在通过上述保护的开发选择，且满足long误差较initial≥10%下降，或moving-continue/stop平均误差≥10%下降、对应力不超过initial+.5N，才允许**全部4组等预算**延到32。还须按每个首段最大实际job耗时×1.25估计4个延长job，并保留3500s终检/视频，整体不超过18000s/12GiB。否则全部16停止。不能依封存单独延长B/某seed；无效不自动追加PPO/候选/Student。

新留出组合预封存于temporal_protocol.json：6.1/23.2/73.5cm、onset5.31，长ramp8.45，同前缀8.76s后平滑停止/继续；arm源5.83s与moving源6.34s后当前需求变化。旧Issue48新条件现已暴露，仅回归。封存开启前全部选择/延长锁定，不用封存修补。完整评价移动/维持/撤销/恢复的XYZ误差/力/饱和、腕相对肩、root/足部/支撑/滑移、实际q/dq与q_PD，释放后旧anchor误差非任务指标。命令不同、latent接近或critic EV不代替成功。

真实不安全、概率不一致、资源权限或总预算不足则保存停止，不能放宽守卫或改公共BFM/PD/main。最终2–4条真实qpos/1×MP4与INDEX验证同步Mac，保留负例和固定checkpoint身份；阶段结束统一更新Issue/Master/research，不自动进入下一阶段。


补充（正式PPO前固定）：本版rho施加于全部256D白化噪声（含isotropic floor），研究整体时间连续性，不把收益单独归因于4条候选轴。封存共143个执行单元：4组best/last加原生初始/C32/I/C0各11均值条件132，4个best各2次固定随机长距8，standing3；重复函数/continue条件不作为独立任务。视频固定4组末轮（非终检挑选）：近距、长距、seed4901同前缀停止/继续、seed4901 moving源双视角。末轮与开发best分别报告。采样数在优化之前单独落盘，优化失败不能把已交互样本记0。
