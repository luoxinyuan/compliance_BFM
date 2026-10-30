本轮唯一累计修正实验已完成。**固定 B 全身 latent 路径，新增有界累计右臂反馈后，完整 6s 稳定完成；保持段三维 RMS 从 5.880cm 降至 0.884cm（−84.97%），通过原 hold<1cm 判据，全接触 RMS 从 3.862cm 降至 0.969cm（−74.92%）。** 这支持“已有身体响应加持续局部反馈，能够完成这个前伸任务”的有限假设。它仍是特权教师，依赖显著直接关节修正，**不是 latent-only Student 或 contact-conditioned motion transition 已成功**。本轮未训练 Student/PPO，未扩大旧 Student 范围。

固定设置与唯一方法改动：

- 精确复用 [原 B 轨迹](https://github.com/GaoE05/robot-research-bridge/issues/21#issuecomment-6515834114)作对照，未重跑 B。新分支完整机械状态、动作/本体历史与 B 相同；3s 无力前缀 qpos/qvel/wrist/root/force 最大差全部 0。新轨迹逐个 50Hz 控制样本的 executed_z 与保存 B **逐值一致**，伙伴 anchor 也逐值一致。固定 +X18.6819cm、seed0、4s smoothstep、1s hold、t8s release、t9s结束，原 K80/D8、5N cap、BFM/PD/Tether 和 10/50/200Hz 不变。力和 BFM action 随新实际状态自然变化。
- **教师算法修改，不是 bug 修复：** 仍用原 bounded Jacobian 算子，每100ms增量每关节≤0.04rad、L2≤0.08rad、damping=.02；从已实际施加的修正累加，投影到总界并丢弃被拒增量，不保留隐含积分量。每关节累计≤0.20rad（11.46°），L2≤0.40rad；50Hz施加时速度≤0.40rad/s/关节、L2≤0.80rad/s，所以每20ms新增修正≤0.008rad（0.458°）/关节。正反误差都可让修正退出饱和，没有硬编码肩关节目标。
- **边界先于物理固定：** B实际关节与命令的最小机械余量为0.926rad；每关节总界取 min(0.20rad, 半个观测余量)。原增量/0.1s提供0.4/0.8rad/s速度依据；0.8s cubic release的峰值导数系数1.5，0.20/0.40rad总界允许在该速度内归零。首次观察到接触关闭时冻结已施加修正，按1−smoothstep在0.8s内归零并停止积分；原B的 **latent 0.2s release路径没有改变**。在线还与原机械关节范围（保留.02rad余量）及原归一化action范围相交。

E为真实 wrist–anchor 3D RMS，单位cm。XYZ列均为分量RMS；完整原始方向、力和轨迹已保存，释放后force=0不计作跟随收益。

| 教师 | 移动 E；XYZ (cm) | 保持 E；XYZ (cm) | 全接触 E (cm) | 移动/保持力 RMS (N) | 接触力峰值；保持触cap比例 |
|---|---|---|---:|---|---|
| 原B非累计 | 3.163；[2.910,1.212,0.256] | 5.880；[3.398,4.745,0.717] | 3.862 | 2.637 / 4.749 | 5.000N；14.5% |
| C累计 | 0.989；[0.554,0.559,0.597] | **0.884**；[0.365,0.525,0.611] | **0.969** | 0.894 / 0.992 | 2.292N；0% |

移动E改善68.74%，但移动Z RMS由0.256增至0.597cm，保留该退步。C保持力峰值1.545N，平均真实力XYZ=[−0.009,−0.115,+0.081]N，B为[+2.720,−3.841,+0.509]N；平均值接近零不能代替力RMS，C仍有正反向波动。**保持误差并未处处消失：** C中位0.836cm、p95=1.407cm、峰值1.610cm、末端1.139cm，仅69%保持采样<1cm；按事先固定的RMS标准通过，不把它改称逐时刻精确跟踪。

身体、支撑与释放：

- 保持末端腕ΔXYZ：B[+15.338,+4.694,−0.380]cm，C[+18.852,+0.494,−1.012]cm；肩ΔXYZ：B[+7.297,−1.088,−4.337]，C[+6.597,+0.157,−4.310]cm；rootΔXYZ：B[−1.533,+0.659,−1.399]，C[−2.290,+1.508,−1.527]cm。C腕相对root位移[+21.142,−1.014,+0.515]cm。并非靠root整体前漂来填补目标距离。
- C保持末腰yaw/roll/pitch实际变化[16.25°,5.25°,12.68°]，最大躯干旋转22.87°；同实际观测下，B-goal相对standing的腰/腿 q_PD RMS为3.57°/5.55°，新教师对非臂关节的**直接**增量严格为0。身体由原BFM闭环参与，新增局部修正也可能间接影响身体状态，不能作简单因果分账。
- C保持期100%的实际腕位置都在以**初始固定肩**为球心、41.039cm机械链长上界之外（峰值43.347cm）；B保持期为0%。配合真实跟随过关，这比仅报告“躯干动了”更明确地支持当前任务利用了身体参与。仍然双足支撑，无迈步/支撑转换，不能称motion transition已实现。
- 两组均无跌倒、非足触地、实际关节超限、action或torque饱和，双足正载荷100%。C/B root最低76.500/76.591cm，最大pelvis tilt5.103°/5.450°，足接触点切向速度峰值0.622/0.611cm/s。几何COM支撑裕量为正，不把它当作动态稳定性证明。
- C累计修正于t8.02s观察到release后平滑衰减，t8.82s精确归零；自身最大50Hz增量0.458°，没有积分残留。但**整体命令并未变得更平滑**：接触段最大q_PD step由B3.993°变成C4.284°，发生在右腕yaw，全部来自相应实际状态上的BFM输出，该关节直接修正为0。名义t8s释放过渡窗口最大step为B3.051°/C3.904°；其中C3.904°在t≈8.00边界的右腕yaw，实际观察到接触关闭后的最大step为C3.464°（右肩pitch，B3.051°）。因此累计器限速不等于整个冻结BFM控制链限速。真正无力的释放采样单独评估，未用于跟随改善。

**新增直接控制承担多少：** C接触期右臂修正RMS5.43°（B1.48°），保持期6.67°（B1.73°）；单关节峰值11.46°，7关节L2峰值0.330rad（18.92°），未触0.40rad总范数界。第一次单关节累计触界在t4.80s，保持期肩roll有40%控制样本触0.20rad界；活跃决策中52%有增量被投影拒绝，未继续隐式积累。

在C的同一批实际观测上，保持期BFM-goal相对standing的右臂q_PD RMS为11.90°，直接修正6.67°，两者RMS比为**0.561**；相对最终右臂新增命令17.16°的RMS比为0.389。接触期对应前一比值为0.689。按预先记录的RMS比较，直接修正没有在接触期/保持期超过BFM-goal右臂增量，但绝非可忽略的“小补丁”；这些不是物理运动贡献百分比。释放期因goal在.2s回退而直接修正持续.8s，该比值达到1.17，直接控制在释放过渡占比更大。新增控制的实际执行值与目标值最大误差1.67e−7rad，非臂直接增量0。

**首个边界和研究决策：** 原假设得到一次有限的物理正证据，但仍有厘米级波动、较大的直接关节修正及整体命令跳变；尚未验证这些实际动作能由拟用Student受限latent表达，也未验证从contact/proprio/history学习后可复现。不能把累计教师的成功当成BFM latent-only可达性已经通过。**到此结束手工goal/IK校准，不继续加控制补丁。下一步仅建议在本次真实轨迹的移动、保持、触界及释放代表状态上，做冻结BFM的受限latent→实际action/q_PD对齐可达性检查；通过后才考虑contact-only单场景Student。** 若关键动作不可表达或只能靠大幅直接关节通道，则停止这条教师路线，回到现成稳定BFM技能/动作或任务专家来源选择，而不是先放宽Student范围。这一建议未执行。

实现、失败记录与复现：

- 个人分支`gaoe05/bfm-accumulated-correction`，从`a3f0aaad8d605ef0407ca2b2846c33e65b235951`建立。初始固定协议/实现commit `f42e0221023875af9a23724073fbb5aeae964622`；实际物理运行commit **`7a48d90e956d5ce4352b8af703f50cf1b48b2b12`**。核心diff是新增累计器、10Hz提议/50Hz施加及日志、原B轨迹复用和同状态command拆分；原BFM/Tether/PD/旧teacher源码零diff，旧轨迹零覆盖。
- 两次启动完整保留：attempt-01加载原模型并复现250条保存B命令，误差≤1.01e−7rad；预检发现完整±0.20rad机械关节可行盒超出部分手腕原归一化action范围，**未执行接触分支**。这是新实现漏纳入原action范围的工程问题。commit `7a48d90`补齐原action范围交集并加数值余量，未提高任何上限，7项CPU测试及250条保存命令边界检查通过；attempt-02唯一实际6s接触分支完成，无物理停止边界。没有基于物理收益调参/重跑。
- launcher SHA256=`60ef044a96a60bbafbba744ff5dc193504b51cf82fee597a06fa0920e1e8f820`；实际controller SHA256=`ac9c2f3288388c871f140617e224fecbfb3f35e38bbf3a12ab040dc73d326a98`。使用commit和manifest中的完整source hashes识别两次实现，不能仅用未变化的launcher SHA区分。
- P=`/mnt/data/temp_user/yi.gao/projects/compliance_BFM`，worktree=`P/bridge-v2-owned/trees/issue-21-accumulated-correction/repo`；R=`P/bridge-v2-owned/runs/issue21-accumulated-20261030A`。`R/attempt-02/outputs/accumulated.json`保存200Hz物理、50Hz原BFM/standing/最终动作与修正、10Hz累计提议/被拒增量；`accumulated-analysis.json`、`supplemental-diagnostics.json`、`protocol.json`、`precheck-decision.json`、`accumulated-pd-audit.json`及完整起点、原B SHA引用均保留。各attempt还有source/manifest/stdout/stderr/遥测；原失败未覆盖。
- 8个初始新Python文件静态编译、7项CPU控制器测试、保存轨迹数值审计通过；q_PD重算误差≤1.49e−7rad，原PD力矩≤9.06e−6Nm。两次墙钟共211.767s；4 CPU，RSS峰值8.836GiB，GPU5峰值566MiB。仅新增6s接触分支+3s无力前缀记录，无策略更新。两个owned进程组3983709/3985470及子进程均退出、GPU5回到1MiB，共享源码快照完全一致且checkout干净。原BFM参数哈希/版本不变、梯度为空；S0/S1 checkpoint SHA复核仍为`a5daf853…809d`/`4c2eb34f…dac`。没有push、没有Bridge或后续训练。
