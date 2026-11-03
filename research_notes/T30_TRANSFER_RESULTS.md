# Teacher30 transfer / recovery / motion quality — 2026-11-03

## 科研结论与决策

**完成有限迁移与局部恢复闭环，但没有获得可全面替换旧r0的新Student。** Teacher30不是在所有Student状态都更好：S0早停止2/5、晚停止5/5接管满足预定筛选；S1warm共10/20端点通过，只有4个两端均通过的小窗口，最终20条访问状态标签。前缀物理差均为0，不是初始化不一致。

S2在四个开发接触条件均优于同预算S1，但新的早停止退化：S0 6.594cm/3.645N，S1 6.762/3.578，S2 8.295/4.509；新晚停止S2则由S0 7.682/4.201改善至6.741/3.878。正常开发S2 7.429/4.058仍差于S0 6.505/3.698。因此保留S0基线，S1/S2为有不同取舍的研究版本，不根据终检选统一赢家。

停止、保持、释放不能合并成一个成功标签：新晚S2保持4.76cm/3.59N优于S0 6.15/4.12，但释放后root路程6.00cm大于S0 2.20cm；新早S1释放路程1.24cm小于S0 8.00cm，而S2保持几乎持续饱和（4.92N）。无接触各Student稳定保持，局部请求保留并改善；76个回合无记录的物理终止/双足同时失去支撑，不等于动作质量已解决。

相同4k更新/数据/seed：有力S1warm新早7.679cm/4.259N、新晚6.934/3.718；真正无力N30 11.385/3.881、12.019/4.646。新早N30力较低却误差大，不能称更好柔顺。换更好标签没有自动弥补感知与闭环差距；没有证明无力路线理论不可行，也没有模拟新的本体噪声或提高cap。

首个明确边界是**Teacher恢复域局部、标签拟合与闭环泛化不等价**。S1warm正常条件首次持续腕轨迹偏离Teacher自身轨迹超过2cm/200ms约7.14s；这只标记分离，不等于该时刻是错误原因。移动期同状态首个50Hz插值步q_PD RMS差：S0均值0.175°、S1matched0.122°、S2 0.110°，更接近Teacher仍未保证更好物理。此误差仅对应第一插值子步，不冒充整100ms动作承接质量。输入辨识性与长期分布偏移未被完全分离，不能直接判定唯一根因。

20条访问标签在S2续训每batch占32/128，实际抽样约96000次；这是非常局部的数据重复利用。开发改善/新早退化支持覆盖或过拟合风险，但本轮没有独立随机化该比例，故不声称已证明其因果。没有继续用封存失败补训。

**唯一建议：下一轮先对“访问状态覆盖/重复采样权重＋旧行为保持”做一次受控迁移对照，以S0为保留基线，目标是保住正常牵引并迁移有效停止反馈；不追加当前S2训练，不先改force cap或PD。** 可并行准备现有缓弯的有限进入/退出资格，但当前不能把S2当成熟共享底座直接扩大连续heading训练。此建议未执行。

本阶段不训练Teacher/BFM，不修改PD；真实Tether上限5N。冻结Teacher30自身反馈改善的历史结论保留。本报告区分Teacher自身执行、Student状态接管、标签拟合和Student物理迁移，不能相互替代。所有模型是仿真研究模型，G1精确腕部XYZ外力硬件未知。

## 固定实验与身份

主运行 `8af5a87cf6684469d28a99a11e156ecdb0ba9575`；独立质量对照 `23309461185f0b10cdc6dfc291c8665f6213fa80`。协议[t30_protocol.json](t30_protocol.json)、源码[t30_stage.py](t30_stage.py)。阶段[Issue35](https://github.com/GaoE05/robot-research-bridge/issues/35)。全数据目录：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/t30-transfer-20261103`。

Teacher30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`；旧PPO15 `92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe`；S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`；BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。由实际文件SHA检查，不凭best文件名认定。

Student 1025维：465本体/已执行动作历史、300实际历史力、256自身合法z、4固定腕部安装编码。没有anchor、伙伴速度、任务时钟、active真值或Teacher隐藏状态；BFM observation只含state/last_action/history_actor，其history仅角速度、重力投影、关节位置/速度/动作。N30物理删除力列及对应归一化/网络输入，725维，重新监督训练；没有外力矩估计或接触真值替代。200Hz物理、50Hz原BFM/PD、10Hz高层命令及原5子步slerp不变。

Teacher30六条独立回合943标签；旧PPO15数据824标签保留正常/无接触/局部行为。每条标签同一10Hz决策时刻的真实输入与合法z，没有重复近恒等投影。固定seed805、网络、归一化、Adam3e-4、batch128。S1warm4000更新，随后分叉相同模型和新Adam：S1matched追加3000原新混合，S2追加3000并将每batch32条Teacher-own换成恢复域Student标签。两者总学习7000更新；实际共享warm计算仅记一次。N30与S1warm同4000更新/数据/seed。固定末轮，不按封存评估挑checkpoint；单训练seed不支持统计普遍性。

新封存：82cm/ramp7.8，早onset4.31/stop7.65/decel0.9/hold3.1，晚onset4.16/stop9.65/decel0.7/hold3.3，释放后6s；其余为已暴露开发/回归。伙伴轨迹真实改变施力，三轴力自然产生。

## 物理结果

单元为全接触3D RMS误差cm / 真实力RMS N；无接触的0不是主动跟随成功。完整XYZ、阶段、饱和、质量和失败见[t30_artifacts/results.json](t30_artifacts/results.json)。

| 条件 | Teacher30 | S0 r0 | S1matched | S2 | S1warm (4k) | N30 (4k) |
|---|---:|---:|---:|---:|---:|---:|
| normal | 5.802 / 3.599 | 6.505 / 3.698 | 8.040 / 4.168 | 7.429 / 4.058 | 8.360 / 4.230 | 10.516 / 4.172 |
| early_stop | 6.321 / 3.698 | 6.863 / 3.611 | 6.724 / 3.563 | 6.407 / 3.385 | 7.019 / 3.648 | 10.553 / 4.474 |
| late_stop | 5.557 / 3.304 | 6.457 / 3.716 | 6.359 / 3.604 | 6.099 / 3.582 | 6.389 / 3.124 | 11.033 / 4.413 |
| local | 3.513 / 2.575 | 3.675 / 2.611 | 3.449 / 2.463 | 3.292 / 2.378 | 3.398 / 2.427 | 6.233 / 3.906 |
| no_contact | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| sealed_early | 7.128 / 4.144 | 6.594 / 3.645 | 6.762 / 3.578 | 8.295 / 4.509 | 7.679 / 4.259 | 11.385 / 3.881 |
| sealed_late | 5.873 / 3.642 | 7.682 / 4.201 | 7.383 / 4.385 | 6.741 / 3.878 | 6.934 / 3.718 | 12.019 / 4.646 |

## 接管与聚合范围

完整Student前缀重放到接管时刻，保留BFM历史、实际动作、自身z和真实物理状态。所有接管前qpos/qvel/force逐点比较；没有只恢复qpos或虚构Teacher历史。接管判据为剩余接触force RMS+3×error(m)较低，释放root路程不比继续Student增加超过3cm；释放后用root路程+腕速描述性组合。此操作性筛选不是通用恢复真值，必须同时看分项取舍。

- `S0take_early_stop-recovery.json`：2/5端点满足预定筛选；完整前缀最大差0.0；接受区间[]。
- `S0take_late_stop-recovery.json`：5/5端点满足预定筛选；完整前缀最大差0.0；接受区间[]。
- `S1take_early_stop-recovery.json`：5/10端点满足预定筛选；完整前缀最大差0.0；接受区间[[8.7, 9.1], [11.1, 11.5]]。
- `S1take_late_stop-recovery.json`：5/10端点满足预定筛选；完整前缀最大差0.0；接受区间[[8.7, 9.1], [9.399999999999999, 9.799999999999999]]。

实际聚合20条标签，来自2条Student回合：['S1warm_early_stop', 'S1warm_late_stop']。仅取两端接管都成立的0.4s窗口；这些区间之间或其他状态没有恢复保证。少量标签重复采样的风险完整保留，不将端点的轻微改善夸大为大范围恢复。

## 阶段指标与恢复

| 模型/条件 | 阶段 | 3D误差cm | 力N | 饱和% | root位移XYZ cm |
|---|---|---:|---:|---:|---|
|Teacher30_sealed_early|move|7.946|3.889|50.6|[19.02, 8.23, -2.24]|
|Teacher30_sealed_early|decelerate|8.126|4.922|84.4|[22.25, 0.21, 0.6]|
|Teacher30_sealed_early|hold|5.748|4.162|15.5|[6.89, -5.27, 1.67]|
|S0_sealed_early|move|8.481|3.992|53.7|[23.06, 9.02, -2.25]|
|S0_sealed_early|decelerate|6.298|4.479|25.6|[16.45, -2.07, 0.15]|
|S0_sealed_early|hold|3.725|2.905|5.0|[-2.1, -0.22, -1.7]|
|S1matched_sealed_early|move|8.451|3.930|51.2|[23.03, 11.21, -2.37]|
|S1matched_sealed_early|decelerate|7.378|4.747|52.2|[17.32, -3.41, 0.06]|
|S1matched_sealed_early|hold|3.932|2.668|11.5|[-0.18, -1.07, 1.87]|
|S2_sealed_early|move|8.490|3.931|51.8|[19.57, 10.8, -2.05]|
|S2_sealed_early|decelerate|9.592|5.000|100.0|[21.14, -1.66, -0.35]|
|S2_sealed_early|hold|7.647|4.920|72.9|[9.78, -4.76, 2.38]|
|Teacher30_sealed_late|move|7.135|4.175|46.9|[57.95, 4.62, -3.03]|
|Teacher30_sealed_late|decelerate|4.949|3.917|17.9|[14.15, 1.79, 2.33]|
|Teacher30_sealed_late|hold|3.055|2.428|0.0|[2.74, -3.56, 0.68]|
|S0_sealed_late|move|8.612|4.136|54.9|[56.91, 4.98, -1.57]|
|S0_sealed_late|decelerate|6.352|5.000|96.4|[2.12, 0.73, -3.18]|
|S0_sealed_late|hold|6.145|4.119|40.2|[14.76, 0.5, 4.48]|
|S1matched_sealed_late|move|8.008|4.130|54.7|[58.35, 3.77, -1.2]|
|S1matched_sealed_late|decelerate|4.956|4.110|0.0|[5.94, 4.08, -0.95]|
|S1matched_sealed_late|hold|6.693|4.828|53.2|[-0.87, 1.51, -1.8]|
|S2_sealed_late|move|7.870|4.014|50.3|[58.07, 3.43, -1.7]|
|S2_sealed_late|decelerate|4.831|4.077|10.0|[3.71, 3.65, -1.02]|
|S2_sealed_late|hold|4.764|3.592|20.0|[2.6, -1.52, 0.2]|
|N30_sealed_late|move|13.791|4.390|64.7|[55.9, 3.9, -2.25]|
|N30_sealed_late|decelerate|6.245|4.842|56.4|[14.55, 0.84, -1.25]|
|N30_sealed_late|hold|9.526|5.000|100.0|[11.04, -2.82, 3.51]|

释放后的几何anchor误差及force=0不计主动跟随收益。完整释放路程、腕速、支撑变化见逐阶段质量数据；支撑布尔翻转不等于准确步数。无接触也检查姿态/位移，不以force=0代替保持资格。

## 目标与真实动作质量

旧Teacher30四个保存条件的大目标跳变都发生于移动期右踝pitch，41.75–53.92°。同状态计算分解：只换z为0.44–1.54°，只换BFM观测为41.38–53.47°；重构保存q_PD误差0。它指向原BFM状态反馈输出，不证明是独立物理因果或BFM实现bug。实际前一个20ms关节变化2.42–3.75°，不能称关节瞬间旋转50°。完整分析另记录后一个20ms变化、q/dq/目标差、力矩/限幅、滑移、root/腕加速度。

固定0.1s高层因果z平滑（原PD不变）的有限物理对照没有整体收益，不采用、不追调tau：

|条件|原误差/力|平滑误差/力|
|---|---:|---:|
|normal|5.802/3.599|9.076/3.940|
|early_stop|6.321/3.698|9.174/3.875|
|late_stop|5.557/3.304|8.592/3.380|

晚停止hold有改善但整体移动误差变大，正常条件hold及释放变差；不能只取局部改善宣布连续性修复。

## 新模型身份

- S1warm: `2a4e8babb81cad35a127d8feeb4bfe594bb93915d476375b37d87bcc2e8e9cff`，`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/t30-transfer-20261103/A-prepare-01/job/S1warm.pt`，本次fit 4000更新。
- S2: `2ecfff8e3ff8ee70e2b8889cea85820552747a9c69f47f616294401339a27b1d`，`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/t30-transfer-20261103/A-prepare-01/job/S2.pt`，本次fit 3000更新。
- S1matched: `11b82954b10f68961d05f32a74b16fb30e79efc58f428b9852572268974744ad`，`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/t30-transfer-20261103/A-prepare-01/job/S1matched.pt`，本次fit 3000更新。
- N30: `49d9aa94114b3e5eadb679796b43160725d72830da5353a63b4d4db2d809bf83`，`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/t30-transfer-20261103/C-final-02/job/N30.pt`，本次fit 4000更新。

## 资源、复现与限制

账本`compute-ledger.json`记录真实卡号、GPU/内存峰值、固定源码、进程、墙钟及退出。解析/汇总脚本为`t30_analyze.py`、`t30_label_audit.py`、`t30_report.py`；物理数据及全部失败不入Git，公开的是精简数值。不要重复运行已消费的stage覆盖目录。视频来自真实保存qpos，双固定世界/俯视、1×，含真实force/anchor；从ready t=3s到完整释放结束，省略的是共同无交互预初始化，不剪掉失败。

本阶段只有单训练seed、少量窗口恢复证据；近似接触信息与Teacher特权信息差异、有限覆盖及闭环分布偏移尚未被独立完全分离。不能由新标签未全面获益证明无力不可行或BFM没有能力。

平面下一候选仍为已有同权重M10缓弯/heading，已有standing进入证据，不等于Teacher30在当前停止偏离状态可安全切换；高度候选已有下沉/回升，属于保持连接Tether，不是已验证单向按压。此轮不新增大范围技能训练，见STOP_CAPABILITY_REVIEW。

## 执行核算与工程记录

14000次新增Student optimizer updates，0RL更新；76个完整物理回合（含30次Teacher接管），236888个记录的物理步、11866个高层控制决策，**这些不是RL训练样本**。共同预初始化另计，墙钟已包括加载和初始化。含3次真实qpos渲染，累计受监督job-wall2293.69秒；结束时约1.39GiB输出，均低于18000秒/10GiB/16000更新授权。GPU1/5/6独立运行，所有owned已退出；未操作他人任务。

一次只读分析在运行文件尚未写完时读到gzip EOF；修正为只分析已登记完成的summary，重跑分析成功。它没有修改runtime、训练、模型或物理数据。没有把分析读取竞争误记为科学失败或重跑实验。

Mac实际交付3个MP4及一个INDEX：`~/Projects/compliance-bfm-diagnostics/videos/teacher30-transfer-20261103/`。01为封存早停止S0/S1/S2失败对照；02为封存晚停止Teacher30/同4k有力/无力；03为原Teacher与未采用平滑的正常开发负对照。保留完整交互至释放后6s，非挑选seed或参考动画。
