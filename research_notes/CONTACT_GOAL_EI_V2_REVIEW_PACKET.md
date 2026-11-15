# Issue66 — E/I v2修订审阅单（未签署、未运行）

2026-11-15。本轮只读取GitHub/源码/旧账，写文档；新物理、训练、Sealed读取均0。

正文：[CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_DRAFT.md](CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_DRAFT.md)。旧下一E/I草案保留，不改H0正判或任何旧负结果。

|Issue66审查点|本稿落点|还缺什么／当前状态|
|---|---|---|
|E可被Goal-only查表复现|§0、§5.3、§7：查表为主对照，必须真实超越菜单；不会把H0再开一次|新学习增量UNMEASURED|
|HOLD与旧卸载reward冲突|§2：FOLLOW旧成本逐值一致，HOLD替换3个冲突项，保留共同力/质量；给出固定公式及匹配排序门|新reward和理想自身里程计假设待Yi批准；未在代码中改reward|
|contact只经U或旧adapter|§3：真实力历史→零影响FiLM→原Actor末Block；同状态同z检查，隐藏字段干预|新接口待实现；不是把旧Actor说成已有接触输入|
|自由256D＋29D双频PPO过早|§4：先低层、再有限categorical；10Hz命令、.5s最短option、50Hz一次raw Gaussian；条件联合logprob、实际映射、KL/Adam事务|新采样数学待批、尚无概率/物理PASS；旧.01/.06不能移植叫PPO|
|stop窗口没跨事件|§5：跨实际减速、hold≥1.5s、release后4s；逐tick事件资格|完整新评价未跑，不以任务名代替|
|只有standing/前期小窗|固定arm-ready源、mid晚期高负担、真正stop/continue、双向Goal变更、完整恢复|不扩未资格walking，不用source hash凑独立|
|各自onset对齐会改变伙伴|§5.1新增明确待签的共同native参考anchor规则；所有Goal/方法用同伙伴路径|是显式科学初始化修订，非偷偷改物理；未实施|
|I只有漂亮表示无物理增量|§7.2：成功率＋2/16；两族J降≥10%、第三族≤5%退化；K分层/力误差/尾部/恢复共同保护；§7.3 probe＋因果消融|未测；无样本、天花板或消融不足均不得补过|
|阶段之间凭什么放行|§6共17行三列表：G00—G15及G01-P；每行有原始量/规则/已有例证边界和失败动作|所有新门UNMEASURED，不借H0的PASS放行新合同|
|失败又加一串Teacher/Q门|§7.4承接五分支；§9单根因一次等价工程修复，其余停报|没有独立Q70%或H0.1，不自动Student/额外seed|
|小预算签后扩|§8完整41,400秒/7GiB/950k ticks；正式64k＋工程8192条分账|原余额36,462.071743秒，差4,937.928257秒；提前请求新增独立6,000秒，未批准不能开正式训练|

## 预算审阅重点

H0结项R2已21,137.928257/57,600秒。完整新方案41,400秒已含本次文档项≤400秒，不能把这400在以后再加一次。若该项最终按400秒保守入账，当前R2变21,537.928257、剩36,062.071743；尚未发生的方案余额41,000，资金缺口**仍为4,937.928257秒**。

新增6,000秒建议作为独立新阶段预算、纳入项目240h；不改旧R2的57,600帽。总可用42,462.071743秒，计划41,400，余1,062.071743不自动变训练。若不批，不把完整试验缩成先开小段；退回改稿。P1端到端实测还须投影全100回合Dev、48 train sentinel与8消融，不是只测采集FPS。

## 五分支与优先级

1. E成功、I无增量：保留E，I不晋级。
2. E成功、I更好且表示/消融支持：I成为下一候选，仍非通用FB。
3. 两者失败：结束配方，按真实证据归因，不自动加训。
4. E失败、I成功：保留I正证，先排容量/数据/初始化/预算混杂。
5. 行为接近或probe/样本不足：证据不足，不按loss/rank补分。

硬安全/权限/恢复正确性优先，危险分支不为凑表继续。完整签署前，以上全部只是待审规则。

## 代码和事实依据（固定4c75ea7，不是新实现结果）

- [Issue66审查6889896067](https://github.com/GaoE05/robot-research-bridge/issues/66#issuecomment-6889896067)：H0查表反例、5.9–7.9<8.3秒、q_PD与动作质量、三项合同建议。
- [H0实际报告](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research_notes/H0_REPRESENTATION_AND_CAUSAL_ADDRESS_RESULTS.md)与[h0_results.json](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research_notes/h0_results.json)：E/I物理结果、能力边界、成本。
- [agent.py:Agent.action/update](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research/interaction_fb/agent.py)：旧robot-only Actor、接触U/adapter、旧clipped探索与旧Q更新；它不是本稿PPO实现。
- [schema.py:deploy/robot](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research/interaction_fb/schema.py)：现有角色权限、隐藏字段排除。
- [dual_track_task.py:native_terms/Task.step](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research_notes/dual_track_task.py)：真实20ms优化reward、5ms有效统计、真实终止与吸收尾。
- [geometry_env.py:student_features/contact_features](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research_notes/geometry_env.py)：1639D restricted数据路径、力历史、安装与特权项的分离。
- [h0_physical.py](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research/interaction_fb/h0_physical.py)、[h0_common.py](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research/interaction_fb/h0_common.py)：四候选、相位/实际last-z、目标/实际插值、24D未来结果、K40/K80公式审计。
- [R2门语义v1.1](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research_notes/R2_EI_GATE_SEMANTICS_V1_1.md)：yaw excursion不等于TV、none不等于release、同态恢复比较与证据等级。
- [QA2合同§9](https://github.com/luoxinyuan/compliance_BFM/blob/4c75ea74bd877e76853e3c49debf4673b40d2e1e/research_notes/R2_EI_G17_QA2_VALUE_TARGET_CONTRACT_V1.md)：原工程自治条款；此次保留边界与800秒修复池，不继承旧Q全局门。

**当前唯一下一动作：Yi审阅修订稿。没有开始新的资格、物理、训练或sealed评价。**
