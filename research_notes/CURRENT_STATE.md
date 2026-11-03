# CURRENT_STATE — 新会话先读

最后核对2026-11-03。**M38有界实验已收尾，全部owned任务退出；保旧左右旋转Expert的完整目标尚未达成。** 未启动PPO或新Student，未自动进入M39。M11 continuation及其他旧训练保持停止。

入口：[研究纲领](RESEARCH_OVERVIEW.md) · [M38完整报告](M38_ROTATION_RESULTS.md) · [Issue38](https://github.com/GaoE05/robot-research-bridge/issues/38) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。上轮[Contact Geometry报告](CONTACT_GEOMETRY_RESULTS.md)/Issue37、T30迁移/Issue35和heading/Issue36不覆盖。

## 可靠基线与本轮结论

- **S0 / T30不替换。** 有限域右腕直行、减速、保持和释放Teacher→ideal-force Student成果成立。释放后旧anchor距离不算任务跟随失败。理想仿真力不等于已确认G1传感能力。
- 旧1639D共享contact-slot Student已具备双点平移收益，但旧腕退化；保留其研究身份。双点仍是同torso_link的`[0,±.13,.15]`m，各2.5N、总5N，不是单向按压。所有原BFM/PD/权重保持冻结。
- **找到并修正真实标签冲突。** 旧旋转源和T30保留数据在t=3s完整输入/BFM观测逐值相同，目标z却差2.236°、q_PD RMS约0.246°。统一接触前来源后目标逐值一致。最初极小非零cdist数值由逐元素检查纠正；冲突不是仅凭冗余表示推测。
- 新因果一致native-goal源真实左右±15°：1.387/0.810cm、sum-force2.143/1.299N、torso+10.17/−14.70°，主要转腰，不是迈步转身。
- 同网络、同T30初始化，两轮各3000更新的同状态q_PD保旧监督完成。r1左较好/右不足；r2开发左+4.85°、右−11.15°。但旧直行T30 5.826cm/3.599N→r2 9.751/4.200，双点平移4.998/3.997→9.188/4.417，**没有可靠保旧**。
- **封存新±13°失败边界明确。** r2左3.446cm/4.139N、torso−7.18°反向；右0.773/1.241、−10.75°有效。相同新左native源1.369/2.130、+8.41°有效。r1/r2不能按方向拼选冒充共享成功。新16cm/−8°组合r2也退化。
- 源状态r2平均q_PD约0.05°仍不足以保证闭环。旧goal接触前大分叉缓解，r2旧直行持续q_PD差>1°首次4.32s（接触4.25s）；实际访问域偏离仍在。T30前缀只能部分改善直行，右旋转反而弱化。
- **有限精确接管有效。** 左/右7s及直行5s，接管前qpos/qvel/force最大差均0。接管后接触段左2.344cm/3.525N→1.605/2.491；右1.028/1.657→0.932/1.496；直行10.144/4.363→6.376/4.148。只界定这三处恢复域，不认定全部标签有效。旧三条多次归一化的非精确前缀结果保留并被正确重跑替代。
- 无真实力时没有按虚拟框架方向播放完整旋转，但有几度非任务yaw漂移；不得用无接触error=0声称无动作。纯旋转没有支撑丢失、力矩饱和；走路q_PD峰仍约46.7°/20ms，实际q峰约1.93°/5ms，未改PD或加入无依据平滑。

## 当前瓶颈与下一决策

当前瓶颈是**旋转动作学习后，旧运动与真实访问域的行为保持不可靠**；不是BFM完全不能转、必须增力或再加接触点。当前r2有部分真实收益，但不是可推广Teacher。两轮数据/采样还不同，不能把结果唯一归因loss或标签修正，也没有多seed统计结论。

唯一建议：下一次明确监督预算中，利用已验证接管域做受控真实访问状态聚合，对比仅增加原数据更新，重点量化旧行为与自身z记忆的闭环稳定性。先不进入M39连续切换或M40共享Student。现有6000监督更新已用完；总墙钟剩余不是额外监督授权。没有自动追加训练。

长期路线已写入纲领：M38保旧旋转Expert→M39连续平移/旋转→M40跨腕/躯干共享Student→M41同需求跨身体状态→M42多部位/高度/未见组合→M43真机传感与Sim-to-Real；后续是规划，不是自动执行授权。

## 精确身份、数据与视频

长期个人分支：[gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)。最终文档整合commit见Issue38结项链接；以下是实际运行源码身份。

- r1 runtime `fa11b4cc54bc5a57f3e15f58ad903efad01819b7`；r2 `02751071de6ebdc2e292c46c85833146caec6c58`；冻结终检 `0f390e10c4399f1db7a4eeebf36297f27ffc224f`；精确接管 `3352e2bc35e9b3313a0109f69a54f7c6eb34fbf9`。
- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- T30 SHA `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`：`bridge-v2-owned/runs/stop-sensing-20261103/B1-train-02/job/best.pt`。
- S0 SHA `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`：`bridge-v2-owned/runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt`。
- r1 SHA `7e09d30f8bda647b49c85124c6121af6c632547d1c135d032ebe960277e6e68a`，`A-goal-02/job/retained-r1.pt`；r2 SHA `310f744427bbe48ab914699044d678fe97d729b6bac96b98173b02dd6cba47f1`，`A-goal-03/job/retained-r2.pt`。均为研究模型，未推广。
- 服务器本轮根：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m38-rotation-20261103/`。[结果索引](m38_artifacts/results.json)、[全部资源与失败](m38_artifacts/resources.json)、[精确接管](m38_artifacts/recovery-exact.json)。旧所有checkpoint/轨迹原样。
- Mac：`~/Projects/compliance-bfm-diagnostics/videos/m38-rotation-retention/`，仅3条MP4＋INDEX：左右旋转源/r2及旧直行退化。真实qpos、1×、16.08s，SHA/时长/全解码/画面检查完成。

## 资源与交接

本轮1514.77累计受监督实验/渲染job秒，约0.92GiB，Expert6000更新、PPO0、Student0，47完整物理回合含3条已标记非精确前缀；首次sparse源码缺失在加载前失败并修复。轻量离线分析未作为额外GPU任务计数。训练用GPU1，渲染短用GPU2；全部owned已退出，GPU0/5/6其他用户任务未操作。共享repo clean，main/BFM/PD/权重/共享环境不改，无新资产下载、无Bridge/watch、无真机。

新会话先读本文件、纲领及Issue38，再实时检查Git、ledger、模型SHA、资源。不得凭旧空闲快照启动，不重放旧任务；普通工程在新授权内自主推进。Student继续排除anchor/伙伴速度/目标yaw/clock/active真值等特权；当前T30/r1/r2是特权Expert，不混淆部署权限。Library中2026-10-28快照属历史，实时状态以本文件及GitHub为准。
