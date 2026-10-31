# M5 完成：首项 FIBER-inspired Contact LPP 已实际训练并通过有限新条件闭环

**结论：已有一版真实有效的接触 LPP 技能。** 以已有 native goal＋bounded 局部反馈为任务专属专家，不再等待“统一通用高层教师”。完成专家动作投影、可部署网络训练、两轮 Student 实际访问状态重查询/聚合，以及固定最终 checkpoint 的真实 MuJoCo 验证。新接触组合中，两版策略均显著降低真实跟随误差和受力；关闭 learned residual 会失去一部分收益，说明不是仅播放 α。

**范围仍是一项 standing 起点的右腕前向引导技能，不是 general whole-body compliance。** 没有训练第二技能或统一 Flow Planner，没有把位移/时序变体算作两种技能；没有新 PPO。M0–M4、旧 Student 和失败完整保留。

**方法与可部署性。** 继承 M4 `f538511` 和当前原 BFM/PD/Tether。教师的高层进度由已有真实力因果 gate 驱动，替代 M4 的全局时钟进度；这是明确的教师算法组合变化，不是工程 bug 修复。保留原门槛/迟滞，不为测试调规则。低层仍为原 BFM 反馈加原10Hz bounded arm correction（每关节≤.04rad、7D norm≤.08rad，50Hz释放fade），anchor 仅用于训练专家。

Student 为1425→128→128→257小MLP：465D原生 `state/last_action/history_actor`＋320帧真实XYZ力历史；输出一个学习到的native进度＋256D原native-centered tangent residual，球面norm16。保留已资格的native端点/原局部范围，因为本任务已可表达，不把这套范围当作以后全局限制。无anchor、未来、全局时钟、实验ID或教师内部状态输入；只在原因果残差gate中使用自身事件经过时间。部署无专家/Jacobian调用、无逐状态latent优化、无额外关节执行权限。

参考 [FIBER §III-C、Appendix C-D](https://arxiv.org/html/2610.24675v1)：动作目标转为同一物理q_PD，通过冻结actor反传到latent，真实on-policy采集后再查询专家。本版保留原BFM-Zero/50Hz接口，用projected native z先验；不是AH-BFM、论文B(s)正则或10Hz大Flow Planner的完整复现。PD映射使用本模型已验证的ratio5，没有照搬论文不同部署配置。

- 临时标签：原160步有界action projection；部署时完全移除。
- Alignment损失：右臂q_PD MSE＋0.1非右臂MSE＋0.01 projected-global-z MSE；Projected-z对照仅用global-z监督。没有直接用教师α作为监督标签。
- 同数据/网络/初始化/采样预算，3000＋2000＋2000更新，各7000次；Alignment最初500步采用共同z warmstart，随后6500步实际action alignment。每轮AdamW重新初始化为预定方案，lr3e−4、batch64、seed0。
- 初始1605标签＋两轮各2610访问状态标签＝6825。两策略的真实访问状态共同纳入，含无接触与完整8s恢复。保存所有中间checkpoint，不以终检选模。

**固定物理合同。** 完整相同3s standing状态进入；原K80/D8/5N、50Hz actor/PD、200Hz物理。anchor取无力参考在各onset的腕坐标，策略共用固定伙伴轨迹，不伪造相同force。开发：onset5.0/6.2、+X15cm、4s ramp、7s接触、8s恢复。训练前封存：T1=onset7.1/+13.5cm/3.5s ramp/6.5s接触；T2=onset5.7/+16.5cm/4.5s ramp/7.5s接触，各8s恢复。T1/T2是新组合，非未见身体配置或统计广泛泛化。

下表为全接触 **三维误差 RMS cm / 实际力 RMS N**；无接触和释放后的anchor几何误差不计作跟随效果。

| 控制器 | 开发回归 +15cm | 新 T1 | 新 T2 |
|---|---:|---:|---:|
| Standing BFM | 9.003 / 4.411 | 7.942 / 4.408 | 9.953 / 4.415 |
| 真实接触规则 | 3.921 / 3.177 | 3.612 / 2.929 | 4.248 / 3.440 |
| 反馈专家 | 2.967 / 2.404 | 2.707 / 2.195 | 3.219 / 2.606 |
| LPP Action Alignment | 3.042 / 2.461 | 2.650 / 2.146 | 3.423 / 2.769 |
| Projected-z | 3.041 / 2.460 | 2.725 / 2.207 | 3.405 / 2.755 |

Alignment在T1/T2相对Standing误差下降66.63%/65.61%，力下降51.31%/37.27%；相对强规则误差下降26.62%/19.41%。Projected-z同样有效：没有一致方法胜负，不能据一个seed宣称Alignment优越。所有Student主接触的5N限幅比例为0，Standing约68.7–69.6%。

不能只报一个标量。Alignment的T1/T2 XYZ误差RMS分别为[2.154,1.177,.998] / [2.674,1.654,1.354]cm，实际平均力XYZ为[1.334,−.517,.624] / [1.920,−.808,.892]N，峰值3.359/3.972N。保持阶段3D误差仍2.429/3.603cm，未消除所有误差，也不是毫米级IK控制。

**全身物理响应。** Alignment T1/T2腕末位移XYZ=[13.123,2.021,−.700] / [14.725,2.690,−1.548]cm；肩X=9.113/9.358cm，rootX=−2.040/−1.920cm，躯干旋转26.898°/27.152°。这是双足支撑下的手臂—躯干/平衡协调，不是迈步。无接触两Student的qpos和腕部轨迹与Standing逐值相同；接触/无接触配对在onset前也逐值相同，未出现接触前自主转熟悉姿态。

**真正的投影贡献。** 在T1、同一固定checkpoint保持学习到的进度，关闭残差后进行真实物理重跑：

| Student | 正常误差/力 | residual OFF误差/力 |
|---|---:|---:|
| LPP Action Alignment | 2.650 / 2.146 | 3.625 / 2.941 |
| Projected-z | 2.725 / 2.207 | 3.609 / 2.929 |

这是闭环组件消融，关掉残差后状态轨迹会自然分开，不假装所有时刻机械状态仍相同。它支持 learned latent correction 的额外任务价值；没有独立分离force与proprio通道，也没有证明general接触意图理解。

**动作表达与访问状态证据。** 三批标签各组最大的右臂q_PD RMS=.00458/.00480/.00545°，非臂=.04554/.03324/.04161°；最大单关节表达误差.05124°。Student所有执行z保持norm16。首版Alignment在原教师数据右臂误差.0504°，加入实际访问状态后上升至.4986°，更新后.0677°；第二轮新覆盖数据.1605°→.0595°。最终Alignment非臂.0419°；Projected-z最终臂/非臂.0764/.0572°。每轮查询保存实际obs、force、action/q_PD、源文件SHA；不是离线单点优化冒充部署网络。

**停止、恢复及运动质量。** 45条完整物理共893.2秒（含每条3s前缀）；无跌倒、无失支撑、无非足触地、无实际关节越界、无action裁剪或力矩饱和；双足全程有载荷。最终主Student最大单步q_PD变化≤1.521°，同一个峰值关节后20ms的真实变化分别记录，不能混作指令变化。全部回合最大q_PD步长2.017°来自教师。

Alignment T1/T2释放后2/4/6/8s root速度分别：
- T1：[.004368,.000582,.000258,.000161]m/s；
- T2：[.004746,.000557,.000264,.000179]m/s。

完整曲线保留明显的释放瞬态：Alignment峰值root速度.08365/.08812m/s；不是用某个2s数值称完美恢复。T2释放进度最大短时回弹Alignment=.02465、Projected-z=.04744，最后α>.01分别到release+1.26/1.56s；所有主Student在+2至+8s α均为0，未出现晚期重新动作。T2最后1.6s腕XYZ仍漂移[−.244,−.203,−.097]cm，稳定不等于绝对静止。短脉冲回归下两Student未启动完整全身进度（α=0），通过小残差响应；误差.433/.430cm、力.428/.425N，随后恢复。400ms教师退出约定仍只是操作性规则，不是人类意图真值。

**保留的失败与边界。**
1. 工程失败：collect1-attempt-01第一次Student前向发现CPU权重/CUDA输入设备不一致；t=3s、尚未接触。修复加载器后，用同一checkpoint重试。失败日志、运行SHA、103.26s资源消耗完整保留，不算科学结果。原result快照仍为running，但exit_code=1，最终审计按failed解释，未篡改旧记录。
2. 科学负结果：访问状态动作误差改善，并未带来单调的物理改善。首版Alignment开发误差2.782cm，固定最终版3.042cm（约退步9.3%）；Projected-z2.841→3.041cm。没有换回较好中间模型，也没有对终检补训。
3. 当前真正的能力边界是任务/控制覆盖：仍沿单一已验证前伸native路径组织反馈，不证明自由方向、多身体配置、连续技能选择或支撑转换。相邻新接触条件有效，不等于general whole-body compliance；旧u=.5及中断历史失败没有被宣布修复。

**研究决策。** M5的第一项有效LPP交付成立，可继续拓展，不再回到“必须先有通用Teacher”的门槛。唯一推荐下一阶段：沿同一前向物理引导，在两种已有自然身体配置中使用各自可查询的有效反馈专家，以本轮已跑通的LPP/真实访问状态聚合训练共享策略，并让策略保存自身可部署的初始姿态/指令记忆；预封新的身体状态×接触组合。检验是否能超越当前单一路径，而非再优化这条轨迹的毫米误差。不预设离散selector、不要求迈步、不以+Z教师作前置。此建议尚未执行。

**版本与位置。**
- 个人分支：`gaoe05/m5-contact-lpp`。训练/两轮聚合核心修正版：`3f2f0629336a8f28729770b91aeda26c3e3380ab`；固定终检runtime：`73613b4ce591d107613ec84c9d1859f198810969`。
- 源码：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/m5-contact-lpp/repo/research_notes/`。核心新增：`m5_stage.py`因果专家/无特权部署；`m5_learning.py`动作投影＋训练＋实际访问聚合；`m5_protocol.json`预算/条件/固定SHA；analysis/plot/audit复现统计。继承BFM/PD源0修改。
- 数据根：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m5-contact-lpp-20261031A/`。真实闭环：`qualification-*`、`collect1-attempt-02`、`collect2-attempt-01`、`final-attempt-01/outputs/*.json.gz`。训练、投影、源SHA、累计dataset：`train0/1/2-attempt-01/outputs/`。
- 最终Alignment：`train2-attempt-01/outputs/alignment-final.pt`，SHA256 `0e14d9f699540b3f5db979d70450fda80e07b307c9a087f8f140fe164d12fe86`。
- 最终Projected-z：`train2-attempt-01/outputs/distill-final.pt`，SHA256 `6475261f6109a348bfd50bab1c29f09d94f7f9f0602e7bdb44f507bb4b8c4891`。
- 统计/完整恢复图：`analysis/physical-summary.json`、`analysis/m5-final-curves.png`；首版统计另存`physical-summary-v1.json`。`compute-ledger.json`保留全部8次启动（7完成＋1工程失败）、日志、PID、版本和资源。
- GPU4累计2128.71s（35.48min），自有峰值710MiB、RSS7.483GiB；原5400s/8CPU/32GiB/8GiB预算内。没有CPU BFM替代、下载大资产、改共享环境、push或Bridge运行。
- 4项测试通过，原BFM每阶段版本/梯度/权重校验不变；原3s站立前缀逐值复现。源码、原权重及旧M1/M3 checkpoint最终核验见`analysis/final-audit.json`。

复现索引见`M5_EXPERIMENT_INDEX.md`，逐阶段确切命令在ledger。独立重跑须用新运行目录，不能覆盖或消费旧记录。当前训练和仿真均已停止，下一阶段等待用户决策。
