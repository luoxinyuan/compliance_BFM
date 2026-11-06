# CURRENT_STATE — 原生能力驱动的共享接触学习

最后核对：2026-11-07 CST。**当前阶段Issue48正在执行，尚未结项。Issue47已完成不重启；C0不再是主线前置。**

[Issue48](https://github.com/GaoE05/robot-research-bridge/issues/48) · [Project Master18 v3.48-WIP](https://github.com/GaoE05/robot-research-bridge/issues/18) · [方法](NATIVE_CONTACT_METHODS.md) · [预登记](native_protocol.json) · [长期纲领](RESEARCH_OVERVIEW.md)

## 当前目标与权限

原生goal/reward/motion→有先验的共享接触反馈RL→真实源状态transition→后续受限观测Student初始化与自身闭环RL。新actor/critic/normalizer/optimizer不继承I/C0/T30高层权重，不把其输出/隐藏特征作为在线输入或默认标签。它们仅冻结对照。不是随机重学BFM/平衡，不是部署时force-to-skill状态机。

冻结BFM/原PD/256D Direct/10Hz→5×50Hz→200Hz；K80D8、主总10N、5N回归，原geometry物理reward。当前仍为1088D**特权Teacher**（当前伙伴误差/相对速度、实际力、本体/真实动作历史）；没有Student训练或真机授权。后续Student必须排除anchor、伙伴真值、active、clock、任务ID和未来，ideal-force不代表硬件传感已具备。

## 已完成、正在执行、尚未执行

- 已完成：NVIDIA batched/batch-one FP32原概率检查及KL/Adam回滚复验，最大联合logprob差≤1.23e-4<原3e-4；拟合后真实输入两臂均9.16e-5。无PPU逐行路径，原阈值不变。
- 已完成：首批6个逻辑候选、7个真实资格回合。standing稳定；5cm goal伸臂1.587cm/1.303N、root最大0.68cm；15cm为4.740cm/3.886N。lean30为13.080cm/8.838N，不当优质30cm专家。native motion任务版能进入/持续/退出，但72cm负载33.092cm/9.075N且推进过多，只是运动先验。旧reward move-ego侧漂/指令跳变负例保留，没有新大资产下载或reward seed扫描。
- 已完成：fresh高层网络用7个原生回合的441个维持/恢复时刻记录作2000更新初始化，不是441个独立回合。排除隐藏时钟选择的进入/退出帧；低raw MSE不代表闭环成功。新初始化连接零需求会多走、长距又推进不足；无接触root约2mm。问题完整保留，不先无限BC修补才开始RL。
- 正在执行：GPU0普通Gaussian已完成32批/24576决策，开发最佳仍iteration0（未胜过原生初始化）；原生方向协方差Gaussian组已真实启动，同样32批/24576决策；每臂首段32×768，相同seed4801、初始化、网络、课程、reward和决策预算。正式runtime见下方。源standing/reach/walking实际执行到不同阶段后同env接管，solver/BFM/动作/last-z/力历史不清零。零需求、近/中/远、停止/继续及移动起点需求变化均为正常分布。
- 已预登记追加：坐标组（最多32×768），仅改均值最后输出层的训练坐标，初始raw/Gaussian均值逐位相同、actor参数仍188928；隐藏层继续训练、无第二actor/半径限制。原两组保留完整比较；同一保存批次上的作废clone更新，原生方向均值改变量约9倍、正交变化和KL相近，这不是物理成功证据。原概率阈值1.22e-4通过。追加前封存测试仍未打开，三组总RL最多73728，不变更98304阶段上限。
- 尚未完成：三组最终开发选择、封存真实物理矩阵、真实qpos视频与Mac同步、科学结项。已登记仅在三组及supervisor完整退出后进行的97回合冻结评估/离线渲染队列（continue与long重复，不多算独立条件）；不新增RL或Student。Mac目录已确认可写，不等于视频已交付。

探索实现：A各向同性raw std=.008；B固定满秩协方差 `.008²I+.15²DᵀD`，D为按来源均衡的4个原生切向方向；白化Gaussian采样u，raw=L*u一次后原sphere16投影，old logprob对应实际u。最终均值执行不查询候选，不限制为菜单/小半径。**此对照同时改变方向与总探索方差，也改变概率信赖域的raw几何；即便B胜出也不能单独归因方向性。** 无时间持续proposal或SF planner。引导组正式开始前发现unweighted SVD被71帧walking压过3个静态goal，局部切向覆盖仅9–13%；改用arm5/arm15/lean/walkmean四来源正交基，秩/方差总量不变；新真实输入概率复验1.22e-4通过。首版保留，未看引导/封存结果改方向，普通组执行不变。

## 精确身份、路径与账本

- 原BFM SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- 新原生初始化 SHA256 `d399174cc2ddbd3efb858953653c9134495070948341381565f5f918f58da5cd`；路径`RUN_ROOT/native-initial.pt`。任何带best命名文件须核对selection.json/iteration/SHA，不推定训练成功。
- 当前训练runtime `c1a8cc3e50d2258aa6afbf8d3ecf7b889c6250d6`，个人实验分支`gaoe05/native-contact-learning`；运行树`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/native-contact-learning/repo`保持不动。此文档通过独立`native-contact-docs/repo`维护，报告commit不混作runtime。
- RUN_ROOT=`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-contact-20261106`；`ledger.json`的jobs/PID/start_ticks、`ppo-*/status.json`、`training.jsonl`、`selection.json`为实时依据，ledger最上层preparation是初始标记，不能覆盖jobs实际状态。
- 只用当前实时获准PRO5000 GPU0；其他卡/他人任务不动。上限24000累计job秒/15GiB/98304新增高层RL决策；C0专项为0。初步队列各32批共49152决策，不自动耗尽上限。
- 工程失败保留：新worktree物化问题0RL；冷字节码CPFS启动中止0RL，后用同源码PYTHONDONTWRITEBYTECODE=1执行成功。初始化/概率/失败/评估均计入ledger。不修改共享BFM/PD/权重/main/SoftSONIC，无真机。
- `postprocessing/status.json`及pid.json记录冻结评估/渲染队列。该队列不增加训练，不根据终检重选模型；若failed保留错误，禁止重复启动整个训练。
- Mac目标`~/Projects/compliance-bfm-diagnostics/videos/native-contact-learning/`：计划仅4条1×MP4＋INDEX（局部/长距、同模型共同前缀停止/继续、连接零需求含原生Standing物理对照）。目前尚未生成或验证，不能宣称已同步。

## 不可丢失的历史与基线

I `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`；C0 `ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c`；T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`；S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`；D2 `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`。不得覆盖或按任务拼checkpoint。

[Issue47完整历史](C0_ENGAGEMENT_RESULTS.md)保留真实接管：早期C0→I有效、晚期不可靠；C0能维持部分已组织walking；固定z对照支持高层反馈贡献。两seed仅4701/batch8在10N取得有限局部/长距共同收益，5N及另一seed不稳定，不推广为新基线。4701选定SHA `f23a6818dab277b775b41d5926641c359546cd6d5f6b2f2003a33514c446e2f6`；PPU服务器当前不可达，不能假装已在本阶段新条件重评。M41启动历史退化、混合监督C1/F/R的局部/长距取舍、旧Teacher→ideal-force Student有限成果全部保留。释放后不把旧anchor误差当主动跟随失败。

## 接力

先读本文件/OVERVIEW/Issue48/当前ledger，再核对精确PID与runtime。**本阶段已在运行，不重启Issue47、不重复启动原生初始化/PPO队列、不移动运行HEAD。** 开发与新封存分开；best与末轮都保留。若只在等待训练，可给出实际进度/ETA；不把启动或低离线loss写成成功。阶段结束核查原始物理、Mac视频、更新Issue/Master并FF整合gaoe05/research；不自动进入Student或下一C0专项。旧Library2026-10-28内容是历史快照。
