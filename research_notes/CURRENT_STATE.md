# CURRENT_STATE — 新会话先读

最后核对2026-11-05 CST。**Issue45冻结I adapter对照已完成并停止；没有PPO或后续训练排队。**

[完整报告](ADAPTER_RESULTS.md) · [协议](adapter_protocol.json) · [聚合证据](adapter_artifacts/physical-metrics.json) · [Issue45](https://github.com/GaoE05/robot-research-bridge/issues/45) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。长期纲领RESEARCH_OVERVIEW不变；Library2026-10-28是历史快照。

## 目标与可靠基线

Contact-conditioned whole-body behavior selection/modulation/transition：真实接触与身体状态决定合理局部手臂、躯干/root和支撑运动，不预设先臂后脚。冻结BFM/PD。有限直行Teacher→ideal-force Student、双向小转腰及历史失败全部保留；不宣称实机已具备精确力传感。

- BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- I `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`。
- T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`。
- S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`；D2 `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`。
- Issue44 C0局部有效但长距失败，C1救回walking却失去局部优势；[原报告](PRIOR_RECOVERY_RESULTS.md)保留，不按任务拼checkpoint。M41启动历史退化证据仍成立。

## Issue45实际结论

1. F全参数与R冻结I+raw条件adapter，两个配对seed，同750数据/恢复标签/损失/保持权重/批次顺序，每个固定2000更新。R初始确定性输出精确复现I，训练后I及归一化未变，无人为小半径。
2. 冻结参数不等于闭环保旧。新9cm F1/F2/R1/R2误差4.671/4.820/4.566/4.780cm，C0为2.784cm；新65cm分别6.012/6.292/6.657/5.135cm，I5.495cm、C0失败32.024cm。两种参数化都未实现最佳局部/长距共存；R没有一致优势。不替换I/T30/S0。
3. 新9cm R1左足路径29.33cm、腕相对肩前移1.79cm；C0仅.38cm足路径、相对腕前移8.07cm。允许合理迈步，但不能仅以root移动判断合适分配。
4. 共同前缀stop/continue在7.05s前物理/观测/命令一致，7.1s出现首个命令差异；证明需求改变后反馈调整，不是预知最终距离。F/R仍读特权几何，不能称force-only。
5. 自然arm-ready由真实native前缀执行，保留历史；身体组织改变但未证明更优可达性分配。原6.5s源其实为后向步态准备，已如实更名保留；补充8s真实前行发展诊断，不称盲测、不据此重选模型。
6. 预固定Teacher I的独立ideal-force Student：883标签/7完整训练回合，2000更新。合法1639D输入无anchor/error/伙伴速度/clock/active/Teacher隐藏信息。新9cm4.232cm/3.471N，新65cm7.404cm/5.943N（I5.495cm/4.534N）。有限迁移可行，长距分布差距仍在。现实力传感未知。
7. 132完整物理回合，无安全终止/失支撑tick；q_PD20ms峰60.287°与实际q5ms峰3.332°区分。全XYZ、支撑、滑移、恢复曲线保存；释放后不以旧anchor误差判失败。

## 下一决策（未执行）

不把adapter作为已证实解法，也不再默认增加混合监督即可共存。唯一建议：下一次有界真实反馈优化围绕共同前缀及近/远请求，从明确固定的可行基线开始，保留可靠walking域并检验局部解能否保留。先用已有数据确认reward是否偏好有效局部解；若需运动经济性项，单独可审查对照，不和架构/课程一起静默改变。需要新授权，不自动启动。force-history网格/真机/扩大任务均未执行。

## 版本、数据与成本

训练/主评估runtime `2673837a1f92cdc9ff0861c7ac1be767a930374a`；补充源/渲染 `3f7a1e6be9d5f40326e01f0468cd3b8f2170a645`。个人长期分支`gaoe05/research`，本次文档最终SHA见Issue45；文档commit不是训练runtime。

- F4501 `52956080217e6346b1a8b196ad411cbc84a07e248a8e747eeedf2dcc775e12f6`
- F4502 `9e009414dcac884f8bfc39566e5c8ea6a9ca3831f47f5e56e0a29ea0421f3462`
- R4501 `298af356d386f79377ca103b4fc82baecbf6e18a997a958633dade2cc7733a37`
- R4502 `d940c66409c1cec7f6e1ffec7c25589c99abf227a597c3bdb67447d4ec9d0561`
- Student `02f3c7685c41a3114814a537ae6a1bf861bdc686a65b89561bcda3cfe952fbae`

服务器run `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105`；Expert在各F/R目录`*-2000.pt`，Student在`probe/ideal-force-student.pt`。初始化/中间/最终及所有失败未覆盖。8000Expert+2000Student更新，0RL；ledger3735.67job秒（含保守故障收费），另约90秒CPU聚合及短审查；约2.03GiB，低于16000秒/10GiB。最多GPU2/5/6；本阶段进程已结束。

R4501监督器读临时空ledger出错，训练子进程正常完成；保留exit_code=null和reconciled_completed，不伪造OS退出码。原日志、677.53秒保守预算计费及原子写修正均保留；没有重复训练。

Mac `~/Projects/compliance-bfm-diagnostics/videos/conditional-adapter/`：3条真实qpos1×MP4+INDEX，SHA与完整解码通过。01同模型短/长；02共同前缀停止/继续；03自然源状态。完整负例保留，不新增一堆重复视频。

接力先读本文件、纲领、ADAPTER_RESULTS、Issue45；资源必须重新实时核查。共享checkout6023c6e、BFM/PD/权重/main/SoftSONIC及他人任务未改。无Bridge/watch/真机。M11初轮与continuation等历史实验身份不得合并改写。
