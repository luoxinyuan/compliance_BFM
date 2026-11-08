> 2026-11-08本有界阶段已完成，最新结论见[P4000_PPO_RESULTS](P4000_PPO_RESULTS.md)与[CURRENT_STATE](CURRENT_STATE.md)。下文保留当时方法/预登记，不代表仍在训练或等待发车。

# Issue52：均值学习瓶颈的有界诊断

2026-11-08用户授权在训练继续时排查，不修改当前配对训练runtime，不新增第二seed。先做保存数据NumPy审查；GPU部分仅在当前两方法精确owned队列退出、GPU确实空闲且预算足够后运行一次。不是新PPO/Student、planner或控制器。

## 已完成的CPU检查

固定四个保存批次1/7/15/23，三worker共3072决策，不含首批value预热。保存数据独立重算GAE与记录阶段均值相符，最大误差约1.96e-7；不是重新进行全套接口审计，也不证明critic优势真实无偏。移动期force成本占91.0%，减速90.7%，不支持force权重太小。移动决策约30.1%在其实际接触行全限幅，减速38.1%；这些完整active饱和决策的force项固定6.0，其他uncapped/跟随/朝向等成本仍有差异。这里“接触行全限幅”允许边界决策含部分无接触行，因此保持期force项可低于6，不能将其误报为物理cap不一致。

这支持“部分区间主要force项区分度变弱”，不是总reward完全无信号或必须增力；更不是Gaussian force sigma过小，当前force项是平方成本。不同状态混合的reward/优势关联不能作动作因果结论。详细数值见p4000_signal_evidence.json。

## 已预登记、待GPU空闲执行

复用既有实际访问状态与完整前缀。首先在同一批输入比较初始化及两方法8/16/24/32/last的累计raw变化、径向/切向份额、合法z方向；角度用double atan2诊断，避免极小角度FP32 acos量化。再用真实上一z、全部5个原BFM观测，比较同状态BFM action/qPD，先用实际源命令复现已存qPD。固定观测的动作回放不叫物理rollout。

随后在原已暴露开发条件normal72的6.2s、stop66的停止后.8s，重放源模型dev24实际raw前缀，逐值核对真实features/qpos/qvel，完整BranchState solver/BFM/force/动作/z历史在每分支恢复。只做2.4s真实闭环窗口，对比P4000均值、两方法32批均值、放大或反向学习修正、原.008和.016的同随机序列采样。放大修正和噪声都是诊断，不变更任何正在训练的分布、PPO概率或生产接口。

记录真实XYZ力/误差、限幅、root/torso/pelvis/足部、qPD与5ms真实关节变化、支撑/滑移与边界；窗口reward和共同critic bootstrap分开，后者不是地面真值。所有raw/physics私有保存，不作为PPO on-policy数据或正确专家标签。比较短窗口不代表全回合改善，更不按结果重选封存checkpoint。

最多600GPU-job秒，包含在原24k/10GiB上限，额外RL训练决策0，必须剩余>1100秒（含500报告预留）才发车。若源前缀不匹配、qPD复算不同、配对训练失败或资源不足，保存失败并停止诊断；不放宽检查、不自动重跑。精确协议见p4000_diagnosis_protocol.json。
