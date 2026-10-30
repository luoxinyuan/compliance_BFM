# 原生移动 latent 部署一致性与质量审计（2026-10-30）

本轮继承 #18 v3.5 和 #21 comment 6519711525。只审计固定 `move-ego-0-0.3`，未训练 Student/PPO、未搜索新 reward、未修改共享源码或原权重。临时 target clipping 是官方 deploy 接口差异的诊断分支，不自动替换科研基线。

## 1. 合同与数值证据

- BFM 权重 SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；旧 reward pickle SHA256 `4c4b811584807f2e6c0ddd9f6cb32f79ea2274781449b005e4c7e8e670f67374`。没有混入 HF 旧 model 的 z。
- 原 buffer SHA256 `46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b`，revision `62b4206d68e026de5e5dc7efb1529bccfb95164c`，沿用上阶段全 SHA 验证与限定版本的 legacy 时序读取适配。没有重下载大资产；确切生成 commit 和全部 reset 边缘语义仍未获得。
- 150,000 个原 seed 样本索引完全复现；原 global softmax(10r)、reward×weight、backward map、norm16 重建原 z，max absolute error=0、夹角=0°。固定 5,000 条物理字段重算 reward 最大误差 `2.98e-8`。没有改 softmax 分块归一化、温度或采样器。
- standing/抬臂/移动合计 900 帧保存观测的 actor 重放 max error=0。`last_action=5×previous_raw_action` 最大误差 `2.38e-7`；actor间隔20ms、PD 50Hz、原MuJoCo200Hz。直接50Hz wrapper步进完整复现旧6秒移动分支，qpos/qvel/root/force/action差异全0。
- 原生 reward 实际读取 `torso_link_subtreelinvel`，并在 pelvis yaw 坐标评价方向；旧表的 full-body COM 差分只是标明过的行为诊断，两种量不能混称。原 reward 奖励水平速度**模长**与软方向分数，不是严格 Cartesian 速度伺服。
- 原移动执行段原生速度模长均值0.29830m/s，但前向0.18777m/s、侧向−0.09005m/s，前向std0.19074m/s；只有13%样本落在0.27–0.33m/s速度带内。原生reward均值0.63937；heading范围−22.99°到28.39°。主要问题是方向忠实性/波动，不能简化为所有速度都不足。

## 2. 哪些原因已排除，哪些尚不能归因

**执行接口：** 原 BFM actor、wrapper、PD、MuJoCo、hard-waist配置与 pinned official main 的对应五文件一致（忽略末尾换行）；动作顺序/缩放、20ms调用、当前 hard-waist gains 未发现主链路实现错误。MuJoCo关节位置和速度是live属性，不能把未显式refresh误说成PD用了陈旧值。

官方 deploy 的默认配置属于旧权重，不可直接替换当前模型的 hard-waist 设置。按作者提供的同款 hard-waist 参数核对，Kp/Kd仅存在约`5e-5`舍入差；有效action scale最大差约`8.40e-4rad`（左腕yaw），不是膝/踝大跳变的同量级错误。作者确认新权重要匹配Kp/Kd，同时说明reward inference存在采样敏感性和塌缩限制：[参数](https://github.com/LeCAR-Lab/BFM-Zero/issues/17#issuecomment-4035650943)、[推断说明](https://github.com/LeCAR-Lab/BFM-Zero/issues/26#issuecomment-4966553366)。这些是来源说明，不等于当前z已经获得作者验证。

一个明确部署差异是：官方 deploy 对最终 `q_target` 做关节位置限位；官方main MuJoCo限制raw action和torque，但没有该最终目标限位。旧移动全轨迹的请求目标最大超配置界28.13°，涉及踝关节，0.759%的joint-time；**实际关节越限为0**。本轮以此为唯一物理接口干预，保留原z、PD增益、raw action/history、起点与时序。

**z生成质量：** 原z计算已逐值复现，并非计算出错就可解释质量问题。八个固定交错子集的z距全批12.30–14.35°，留一子集1.82–2.08°，两半11.62°；加权B resultant/加权B范数为0.10179。这是有限样本敏感性/方向抵消证据，不是“找到更好的z”，也不能单独证明候选失败原因。未执行或选择任何子集z。固定5,000条高reward加权物理速度为前/侧/模长0.2794/0.00169/0.3026m/s，未发现样本本身整体朝负侧向奖励的证据。

**状态/历史：** 常z并不意味着动作恒定，actor始终闭环。将相邻真实观测的state/last_action/history三组作Shapley代数分解：3.10s右膝+58.98595°目标变化分为+31.80219°/+4.88749°/+22.29627°，z贡献为0；4.00s左踝+36.479°中，last_action/history约+19.66°/+19.06°，z仅−0.06985°。这是混合观测的**代数归因，不是同状态物理因果实验**；不能由此证明某一特征有bug或状态分布必然失配。轨迹归一化特征abs的p99/max=2.45/4.46，而replay为2.99/28.22；标量范围重叠不能排除联合状态/历史分布偏移。

**目标与实际运动不同：** 旧完整移动回合实际关节每20ms最大变化9.31°、最大速度8.91rad/s、最大力矩96.88Nm，无跌倒/实际越限。3.10s右膝目标3.21°→62.19°，实际关节49.37°→44.58°；随后四个物理步力矩约57.68/7.59/1.43/−4.59Nm。不能用“59°目标跳变”单独宣告不安全，也不能忽略可重复性和退出质量。

## 3. 资产与推断来源边界

官方main固定commit `05ad6119a25b06c341c191a72ccf936e987f59cf`；deploy固定commit `de9e248ce8b315508aafaecc8f15def3e057fccc`；文件URL/blob SHA保存在 `sources/provenance.json`。核对原reward推断的150k样本、原reward_wr_inference及50Hz调用。官方直接reset/常z运行与本轮共同standing+slerp协议不是同一初始状态实验，不能声称已排除所有进入状态差异。

在本阶段限定的官方/作者资料、固定HF资产清单及个人pretrained/native_reward_assets中，没有取得作者或团队已物理验证的**同一权重** locomotion z + 对应轨迹。旧model的reward pickle不可混用。HF live目录再次访问失败（proxy连接拒绝/工具错误），保留失败；使用的是上一阶段已经保存的不可变revision清单，不冒称重新联网查全。没有无限搜索或自动联系作者。

## 4. 工程失败与保留原则

首次saved-data audit在`abs(Python list)`处失败；同时复制supervisor时误留旧输出根，使新失败目录落入旧阶段并在旧ledger附加一条。发现后未覆盖旧物理/权重，旧ledger按原stage-final中SHA逐字恢复为`e7b7b510a7144decdde0bf409b87009059992545304bab1ebae57d2cf6a1bb90`；新增失败目录仍保留，误写ledger备份与恢复记录存入本轮。52.46s失败计算计入本轮预算。修复类型处理和输出根后审计成功。

第一次target-limit分支用代数等价但浮点运算顺序不同的PD目标公式，进入段已有约`2.5e-8m`状态差异。因此该结果保留为实现受限对照，不用它单独作严格因果归因。随后预先记录一条修正对照：原source顺序构造目标、只对真正越界关节替换torque、其余返回原torque位值；不改限位/增益或物理参数。使用不同空闲GPU但型号相同，要求首次裁剪前完整状态与动作位级相同。

所有失败、旧结果、原始模型和S0/S1/C/D保持。实际物理结果、资源、版本与最后决策见附表。未使用Bridge、未push或启动后台循环。


## 5. 同条件真实物理对照及连续性

原t=3s完整standing快照，Tether关闭；进入0–0.75s、常z执行0.75–3.25s、退出3.25–4s、standing恢复4–6s。下表除标注全回合/最终的行外为执行段；所有位置相对共同起点。正式限位分支首次介入在相对1.36s，此前qpos/qvel/action及物理qpos误差全0。

|指标|原MuJoCo / native_50hz|官方目标限位（保持原算序）|
|---|---:|---:|
|原生前向 / 侧向 / 速度模长均值 m/s|0.188 / -0.090 / 0.298|0.190 / -0.094 / 0.302|
|前向速度std m/s|0.1907|0.1888|
|原生reward均值|0.63937|0.64852|
|速度模长在0.27–0.33m/s比例|13%|15%|
|root末端ΔXYZ cm|54.891 / -27.663 / -7.313|55.724 / -28.985 / -7.632|
|执行root最低高度 cm / pelvis最大tilt °|69.56 / 9.18|69.75 / 9.18|
|torso姿态变化最大 °（含yaw）|27.87|30.62|
|腰yaw/roll/pitch最大变化 °|9.254 / 4.348 / 10.820|9.238 / 4.559 / 10.401|
|双足 / 至少一足几何接触|3.20% / 92.60%|3.20% / 92.00%|
|左 / 右足最大抬升 cm|18.725 / 8.721|18.722 / 8.721|
|执行torque饱和 joint-time|0.110%|0.090%|
|全回合实际关节20ms最大变化 °|9.31|9.32|
|全回合最大 / RMS力矩 Nm|96.88 / 8.84|93.53 / 9.00|
|最终root速度 m/s|0.00241|0.00152|

|每20ms q_PD最大变化 °|原接口|目标限位|
|---|---:|---:|
|进入|21.23|21.23|
|常z执行|58.99|62.93|
|退出|12.72|29.89|
|standing恢复|36.48|7.28|
|最后1秒|0.341|0.134|

限位分支的恢复最大跳变从36.48°降至7.28°，但退出段从12.72°增至29.89°、恒定技能段从58.99°增至62.93°。不能只展示有利阶段宣称解决连续性。最终root ΔXYZ为原接口[63.12,−25.00,0.10]cm、限位[69.07,−25.05,0.002]cm；恢复是停住，不是返回世界原点。两组实际关节越限、非足地面接触、外力、raw-action clip和跌倒/预注册停止均为0。限位分支全回合最低root为68.87cm（在退出段），原接口为69.56cm；两组最大pelvis tilt约9.18°。这不是已证实的危险控制，但也不是可靠的定向移动教师。

首次代数重算限位分支也完整6秒，前/侧/模长0.190129/−0.093848/0.302052m/s、执行q_PD 62.9306°，结论与正式分支一致；它保留在`physics-attempt-01`，正式比较使用`physics-exact-attempt-01`。不按结果挑选候选。

## 6. 研究决策和唯一建议

**本阶段决定：先解决原生技能质量与部署合同，暂不将本移动候选接入Contact Projector训练，也不把已有抬臂技能强行当成定向移动替代。** 原生冻结BFM确实有全身移动/支撑能力；本轮没有证明它不能用于最终研究。已排除的主要工程解释是actor复现、动作坐标/历史缩放、50Hz时序和本地wrapper外层步进。已发现的最终目标限位差异，经过同状态真实物理干预，未解决方向和波动；不能继续堆平滑器来掩盖它。

首个尚未解决的质量边界是**当前同权重z在共同standing起点的方向忠实性与进出技能的闭环质量**。这不是仅看q_PD硬判失败；它由实际侧漂、速度波动、进出响应共同支持。由于尚无独立已验证的同权重locomotion参考，且未做官方reset/历史与本起点的交叉实验，不能把根因唯一归到reward推断或状态分布。

**唯一推荐下一阶段：取得并复现一个同checkpoint的已验证locomotion参考合同，再决定接触切换。** 最小向Xinyuan/作者索取内容：该权重SHA对应的reward z文件与SHA、生成seed/样本索引/预处理和reward命令、使用的代码commit与hard-waist PD配置、完整初始qpos/qvel及actor历史、短程50Hz obs/action/q_PD与200Hz物理记录。拿到后只做参考原起点与当前standing起点的固定交叉复现，以区分候选质量和进入状态；不先改reward、不无限抽z、不训练Student。当前缺这份正参考，应明确等待决策/资产，而不是再把本轮已完成审计当下一轮任务重复。若团队没有该参考，需另行决定是否做预注册有限多样本z稳定性研究或改变技能来源；本轮不自动进入该分支。

## 7. 实际执行、测试与位置

四项回归测试通过（类型处理、输出根、目标限位不污染raw action、原生PD算序）；各脚本JSON/语法及git diff检查通过。900帧GPU actor重放、原z重建、完整物理基线复现及限位前位级复现均通过；这些比mock测试更直接。原模型tensor hashes前后一致，policy updates=0。

GPU按当时资源选择，saved-data和首次物理使用GPU6，最后修正使用空闲GPU1，同型号RTX PRO5000 72GB；无BFM CPU fallback。实际4次数值启动：一次审计失败、一次审计成功、两次物理成功（2+1个分支）。累计监督wall=325.23s，小于3600s；采样owned VRAM最大624MiB、aggregate RSS最大29.70GiB，均未触及限制。GPU峰值为轮询观测值，不是绝对连续峰值。新物理3×6s+3×3s前缀=27s记录，不含reset构造。四次owned进程均退出；不操作他人任务。

个人分支`gaoe05/bfm-locomotion-deployment-audit`，从`14a323eb6081e35a4f3e928e8d48b7d5567757bf`继承。saved-data runtime `aa8acc8d212378446adc3f4c7f739d7bf4af0a6f`；首物理runtime `dc70c19913eb86cb4af8eb6a6b33f202afc55ccf`；精确限位runtime `75effca1f8e9e56020ef6f16db2520f58c9360c0`。最终归档commit由Git和stage-final记录，未push。

`P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`：
- 代码`P/bridge-v2-owned/trees/issue-21-locomotion-audit/repo/research_notes/locomotion_*`、`test_locomotion_audit.py`。
- 本轮数据`P/bridge-v2-owned/runs/issue21-locomotion-audit-20261030A/`。
- saved audit：`audit-attempt-02/outputs/{saved_trace_replay,jump_attribution,inference_sensitivity}.json`。
- 物理`physics-attempt-01/outputs/`、`physics-exact-attempt-01/outputs/`，分析`analysis/physical-comparison.json`、`analysis/actual_vs_target.json`。
- 来源` sources/`、预算/失败`compute-ledger.json`、`output-root-correction.json`。
- 误根目录的失败仍在旧run的`audit-attempt-01/`，未删除或覆盖。旧run ledger恢复SHA与旧stage-final一致。
- Mac小文件镜像`~/Projects/compliance-bfm-diagnostics/issue21-locomotion-audit/`；服务器保留完整raw trace。

阶段完成后停止，未启动下一研究阶段。
