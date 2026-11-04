# CURRENT_STATE — 新会话先读

最后核对：2026-11-04 11:31 UTC。**新授权Transition-centered privileged latent RL正在执行；准备32个完整物理对照已完成，两组PPO已真实采样更新，尚无最终训练结论。** [Issue41](https://github.com/GaoE05/robot-research-bridge/issues/41) · [研究纲领](RESEARCH_OVERVIEW.md) · [预登记](TRANSITION_PROTOCOL.md) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。此前力范围A/B扩展训练不执行。本轮5/10/20N冻结诊断已获明确授权，不再待审批。

## 保留的事实与基线

总目标Contact-Conditioned Whole-Body Motion Transitions。T30→S0已建立有限域右腕直行/停止Teacher→ideal-force Student链路；ideal-force不等于G1已确认有精确外力传感。M38有限域双向转腰成立，不等于骨盆/脚步转身。M39-A/B保留直行与左右能力，组合进入/撤销仍不协调；[M39A](M39A_RESULTS.md)、[M39B](M39B_RESULTS.md)全部成功/失败保留。M39-B已结项，不原样续训，不自动推广新Student。

可靠模型（均不覆盖）：
- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- T30 SHA `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，runs/stop-sensing-20261103/B1-train-02/job/best.pt。
- S0 SHA `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`，runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt。
- D2 SHA `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`，runs/m38c-history-20261103/C-train-05/job/D2.pt。
- 本轮共同初始化M39-A best64 SHA `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`，runs/m39a-demand-20261104/B-ppo-03/job/best.pt。
- M39-B A64/B56 SHA和结果见M39B_RESULTS；未替换T30/S0，未删除。

以上runs前缀：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/`。个人长期审查分支`gaoe05/research`；具体最新文档提交由Issue41链接记录。源码发布不等于训练成功。

## 本轮合同与正在运行的任务

[transition_protocol.json](transition_protocol.json)启动前锁定。同1088D特权Direct绝对raw→合法256D z，10Hz高层/50Hz BFM/200Hz物理；冻结BFM/PD。无在线force-to-motion规则，无motionID、全局clock或未来路径输入。Teacher仍读当前伙伴误差/速度/yaw需求，权限不自动传给Student。actor与critic同观测。

A共同站立3s起点 vs B冻结源策略真实执行后接管；同伙伴需求分布、初始actor/critic/normalization/std、fresh Adam、raw保持.002及奖励。T30提供真实行走源，旧双向模型提供转腰源；接管不重置qpos/速度/支撑/solver/BFM/动作/z/力历史。源前缀成本单列、不计PPO样本。三类前缀独立重放的完整状态和随后100ms物理完全一致。

两臂主训练均5N，各64×768=49152决策，同seed4111，每16批开发，按预定多条件成本选择；两臂全部tensor初始化hash相同。首段A7/B4批时raw logprob复现误差1.22e-4、联合KL≤.02；这些是实现检查，不是学习成功。最终阶段、源相位、伙伴组合已封存，不按终检补训。B源前缀有额外真实采样成本，不能声称总物理完全同预算。

实际训练runtime `a8afaff087f535124d9e1b28be7ada57caa10bdf`：
- A worktree `trees/transition-standing/repo`，输出`runs/transition-curriculum-20261104/A-ppo-03/job`，GPU6，supervisor/worker以实时ledger为准。
- B worktree `trees/transition-source/repo`，输出`runs/transition-curriculum-20261104/B-ppo-04/job`，GPU1。
- 准备runtime `8d1a02754807eda84fc2842921c9577c56a9dafb`，C-final-02/job，原GPU5已完成。

上述trees/runs均位于项目`bridge-v2-owned`。运行worktree不回写。有限后续脚本只等待这两个明确job，成功且预算满足后锁定开发best做一次封存/保存轨迹渲染；无新训练/重试/扩展。详见transition_finalize.py及continuation-status.json，不是Bridge/watch服务。未执行部分不得写成事实。

## 准备结果与当前科学边界

32完整回合无安全boundary。冻结初始化在双点正向组合：5/10/20N总cap误差9.107/3.790/4.108cm、负担4.609/5.153/6.715N；负向7.896/3.943/3.945cm。直行5N误差5.665cm，10/20N反而6.533/6.542cm且负担上升。固定standing在20N也大幅被带动，因此增力改善不可全算主动学习，20N并非通用答案。所有观测仍/5、reward分母25，未静默重归一化。

原生单独左右goal约+9.68/−14.05°；简单walking/goal组合约44cm误差，不是有效组合专家。这是这个直接切换参考的失败，不能宣称BFM没有任何组合能力。B输入重建state最大差2.98e-8、privileged_state完全一致；B仅辅助诊断，不作为到达概率或奖励。

当前首要待回答：真实源状态课程能否改善进入/维持/撤销，胜过共同站立训练及直接切换；尚未得出训练结论。此前组合弱不能简单全部归因5N，也不能由此宣判底座不可表达。

## 资源、失败与交接

新上限：累计job-wall24000秒、输出15GiB、RL196608决策、最多3空闲获准GPU。实际首轮两臂总98304决策，准备/终检/渲染另计。以`runs/transition-curriculum-20261104/compute-ledger.json`为准，勿复用旧PID或GPU快照。第一准备失败因新worktree继承文档sparse checkout，缺运行源码；3.43秒/0物理，已在私有worktree物化同commit并核对全部PINS。第二次完成32回合，CPU保存数据分析另计。未改共享BFM/PD/权重/main、SoftSONIC或他人任务。

现有Mac视频：`~/Projects/compliance-bfm-diagnostics/videos/m39b-action-retention/`（4MP4+INDEX，已验证）；本轮目标目录`videos/transition-curriculum/`，**截至本记录尚未生成/交付**。新训练模型SHA须从实际开发selection与checkpoint核验，不以best.pt文件名代替身份。

新会话先读本文件/协议/Issue41，再核查实时ledger、进程、Git和模型SHA；训练中不改源码、不重复启动。完成后统一分析、核验Mac视频并更新Issue/Project Master。本轮不自动训练Student或进入下一阶段。Library2026-10-28仅历史快照。
