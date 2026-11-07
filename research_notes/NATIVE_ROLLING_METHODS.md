# Issue50 — 原生候选与短程物理反馈方法

这是原生能力来源与特权反馈获取实验，不是新PPO或Student训练。原Issue49结论、所有旧高层模型与失败保留。

## 冻结合同与身份

- BFM SHA256：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；同一模型内normalizer，FP32，原PD不变。
- 合法高层256D z，范数16；10Hz高层命令、五个50Hz原BFM子步、200Hz真实MuJoCo。高层子步按真实上一z执行原slerp，不另加关节控制器。
- 真实右腕双向保持接触Tether，K80/D8，总cap10N；每个物理tick由当前几何/速度重算受力。没有修改anchor追随机器人、提前断接或降低cap。
- 原replay 26,675,212,288bytes，完整SHA256 `46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b`。复用现有资产，没有下载。
- 实际滚动runtime：`32b3502262399409dd9091b067c89ba619d20873`。代码在个人隔离worktree，运行树不用于同步编辑文档。

## 原生来源不是旧高层actor

局部goal来自已保存的离线机械可达姿态，经同权重BFM goal_inference得到合法z；离线IK不在线执行。goal_reference SHA `09f819a474402639bbf3b5f5d645686e51d56d572b519f112d3c1f4401005a2a`。

walking来源为已有同权重goal推断的motion latent序列，含M10已记录的离线手臂/腰姿态修正。它不是未修改的官方reward技能，也不是在线M10 Student。reference SHA `211a192ddf6e3d534a1ef273c22e9cf977510ccb235e1c88dc2f0b24e8d6f2c0`。

I/C0、native初始/C32仅加载作冻结参照及恢复校验。实际滚动候选生成不调用它们的actor/隐特征、CyclicFeedback、force-to-motion规则或在线IK。

## 三个预固定状态reward

同一个原采样器seed5001抽150000个有效successor；复用已审计的physical row t与observation(t+1)合同，不修改buffer。令body线速度为v，根部角速度z分量为w：

1. `r15 = exp(-((vx-.15)/.10)^2 -(vy/.10)^2 -(w/.25)^2)`。
2. `r22 = exp(-((vx-.22)/.12)^2 -(vy/.10)^2 -(w/.25)^2)`。
3. `r15_arm = r15 * exp(-((wrist_z-shoulder_z+.22)/.25)^2 -(pelvis_tilt/.35)^2)`。

普通embedding为`sum(r_i B(s'_i))`；加权为`sum(r_i softmax(10r)_i B(s'_i))`，随后各自调用原project_z。没有把归一化z线性混合冒充原embedding组合。保存sample indices、每个reward、未投影embedding及6个合法z；与原reward_inference/reward_wr_inference API逐值对照最大差0。

本轮150000 unique重标记状态，含原API验证共450000 backward-map examples，3configs、6z。外部force、anchor、伙伴状态不属于这些B输入，未声称零样本推得接触策略。

## 速度对照与初始化

相同72cm位移、onset5.0s、完整standing接触起点、相同K/D/cap。smoothstep ramp的平均速度为.09/.15/.22m/s，峰值各为1.5倍，时长8/4.8/3.2727s。两个保持秒、四个释放恢复秒。

资格对照在onset前真实standing；motion/reward来源1.5s进入，伙伴完成运动后1.5s按时返回standing。这个退出脚本只用于来源资格，不宣称是自主反馈Expert。不同速度回合不比较累计return；报告active和moving/holding的RMS及单位时间负担，并记录不同进入占比/累计yaw漂移的混杂。

## 同状态短程物理分支

保存完整MuJoCo integration、solver warmstart与约束缓存、q/qdot、支撑、BFM实际obs/历史、真实上一z、原PD/action历史、逐点力/接触历史与自身记忆。额外保存接触measure闭包中的对齐原点；否则预测分支可能污染真实anchor原点。

每个候选从同一快照恢复，实际执行BFM/PD/Tether物理。恢复后同一三个100ms命令的全部physics/control记录逐值一致。不是qpos-only reset，也不是可微物理rollout。

## 搜索与滚动权限

一次24个高层指令=2.4s，每轮7候选，1轮有限random shooting：standing、arm15、ordinary r15、ordinary r22、完整motion、平移后的上一计划/初始渐进计划、一个连续参数扰动。没有根据封存表现追加reward/seed扫描。

sequence使用4个站立→来源alpha节点，位于高层节点0、7.6667、15.3333、23；可选motion或reward-z基底，motion phase-rate .5–1.4，arm goal混合0–.2。协议旧字段`node_spacing_decisions=6`未被实现读取；实际以np.linspace均匀四节点为准，不能误写成固定6步间隔。上一计划按已执行100ms平移。

这是本轮有限搜索覆盖，不是完整256×H全局优化，也不代表最终策略必须永久受固定菜单/小半径限制。自身phase只是实际选择motion后累计的名义控制记忆，不是真值步态相位。

预测仅取当前伙伴位置/速度，使用明确的常速度外推，当前连接状态保持。所有未来stop/release/change元数据被删除。legacy heading reward原读最终distance，预测统一改用固定名义.72尺度，而非各回合未来终点；真实物理reward不变。预测期关闭onset重新对齐，真实onset仍与当时实际腕部对齐一次。

候选按原geometry_env物理reward在2.4s内积分排名；包含逐点力/误差、限幅前需求辅助、相对速度、heading、滑移、倾斜、q_PD连续性及无接触idle代价。无终端学习价值函数，无未来轨迹oracle。unsafe预测拒绝。最后只实际执行首个100ms命令，再从真实新状态搜索。

预测分支、实际第一步及后续真实执行分别保存。搜索动作不是当前Gaussian的on-policy样本，没有PPO likelihood或训练收益宣称。

## 严格共同前缀

两个真实回合使用相同初始化、当前输入、随机提议序列与自身记忆：继续72cm/.15；另一支在7.8s开始.7s平滑减速，之后保持2s、释放恢复4s。分叉前不得向搜索提供后者停止信息；比较完整真实physics/control和输入/上一z/第一指令。分叉后才评价停止响应。

## 证据边界与执行成本

2.4s常速度模型会在实际加减速时失配；重规划可纠正但不是全局最优保证。有限basis/粒子/源相位限制、单组真实停止/继续验证、没有神经策略泛化证据，均需保留。

搜索约23秒墙钟才生成100ms动作，远慢于实时。这一轮用于验证可获取的反馈与未来摊销标签，不是可部署控制器。后续Student仍需只读本体/实际动作、接触位置与理想力/可维护记忆；仿真真实力不等于已验证G1传感。

外部实例重启造成挂载范围和现有runtime变化。旧cu118/MuJoCo3.3.7结果单独保留；当前cu128/3.3.3只做必要差异smoke，BFM/PD与backend内精确前缀通过，12概率回归最大.00019837<.0003，无放宽或PPU修复默认化。不同backend结果不当严格因果对照。

## 保存的控制记忆与未来标签使用

MPC自己的phase、完整候选参数、真实上一z和所选首指令在`searches.json`逐次记录。底层Direct Task沿用的`native_memory`字段为零值，不代表实际MPC源相位；不能用这个legacy字段伪造Teacher历史。未来从Student状态查询时，必须按其真实动作/身体历史建立合法查询记忆，不能照搬另一条Teacher轨迹的相位。

本轮保存的1088D privileged query与1639D restricted ideal-force query用于将来权限审查和标签对齐，不表示本轮已经训练Student。受限输入不含anchor误差、伙伴真值速度、任务clock/active真值或内部候选模式；接触槽位存在只表示已知安装，不等于接触激活。

## 复现入口（需独立资源授权，不自动发车）

使用上述实际runtime与本地现有原模型/资产manifest，隔离stage布局`<stage>/repo`、`<stage>/runs`、`<stage>/ledger.json`。原生推断、资格与滚动都写新目录；禁止覆盖已保存运行。

```text
project .venv/bin/python research_notes/rolling_reward.py --output <stage>/runs/reward-inference-01
project .venv/bin/python research_notes/native_rolling.py --mode qualification --output <stage>/runs/qualification-NEW
project .venv/bin/python research_notes/native_rolling.py --mode diagnostic --output <stage>/runs/diagnostic-NEW
project .venv/bin/python research_notes/native_rolling.py --mode rolling --output <stage>/runs/rolling-NEW
```

上述为入口，不跳过`rolling_job.py`所实施的空闲GPU/内存/输出与累计墙钟保护。实际执行以对应launch/ledger记录的命令、源码commit及protocol SHA为准；文档/统计提交不冒充物理runtime。依赖同checkpoint私有模型/replay与已保存原生reference，不把源代码可读误称为无需资产即可完全复现。


## 为什么没有新PPO，Teacher仍有反馈收益

本轮优化的是每次当前状态下的有限候选序列，而非网络参数：完整快照→7条2.4秒真实物理分支→累计原reward/拒绝unsafe→恢复快照→仅执行最佳首100ms→读取真实新状态再规划。候选可以不完美；其组合与执行时机由本轮物理评分确定，没有写死先臂后脚。7提议/窗口/源基底是搜索覆盖限制，不是未来Direct256D策略永久菜单。对应固定runtime的native_rolling.py:126–190与236–246，效果和质量边界仍以既有完整报告为准。

新讨论明确区分原生来源、序列搜索、PPO学习及受限观测迁移；也允许另行设计未来可见Teacher/critic上界，但当前实测不读未来，未启动新实验。[完整方法讨论第9节](CONTACT_LEARNING_RATIONALE.md#9-issue50讨论澄清候选物理搜索ppo与未来信息)。
