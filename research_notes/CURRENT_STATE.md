# Current — Issue66 E/I v2：P1暂停，正式训练未启动


## 2026-11-15：独立审查后局部优化修订待签（未执行）

已按[Issue66独立审查6893703131](https://github.com/GaoE05/robot-research-bridge/issues/66#issuecomment-6893703131)完成存档/合同/源码只读核查。[事实审计](CONTACT_GOAL_EI_V2_OPTIMIZER_AUDIT.md)与[OR1可签修订稿](CONTACT_GOAL_EI_V2_OPTIMIZER_AMENDMENT_OR1_DRAFT.md)、[待签参数](contact_goal_ei_v2_or1_draft_lock.json)已归档。裁决：P1优化规则资格失败，不是正式E/I行为学习失败。原v2已签正文与原工程账本不改。

实际P3 option draws E83/I41；I只有3次FOLLOW且全部arm15。历史clip_fraction全0是pre-step记录，不能证明被拒Adam提议温和；未保存proposal后的完整surrogate/逐关节变化，不补造。当前没有新GPU网格/优化器/采样/物理，α1端点只有CPU修复。

OR1建议固定α[1,.5,.25,.125,.0625]，原KL.02/sigma/reward/物理不变；原Adam一次提议，缩参数配完整新moments事务。先两个归档锚点、可测执行步后才给有限新on-policy资格；依旧不直接启动32k/臂。联合100ms信用/selector耦合仍为机制候选，20ms分层替代只设计、不运行。

待签调整：新工程≤8192条，旧8192原账保留；新工作包总≤6000s（含本轮只读review400），另申请资金5000s；P1段10100s、P6段1400s、stage总47000s（原41000）、输出8.4GiB（原7），其他原帽及科学门保留。原正式训练完整成本待accepted-Actor吞吐/存盘实测，不承诺旧段额度必能容纳。

[新review独立费用凭证](contact_goal_ei_v2_or1_review_receipt.json)：400s整体保守上界，无GPU/物理；原stage7807.703624不覆盖，合并review为8207.703624；R2参考29745.631881/57600，原另批6000未使用。原P1 3837.064558、采样8192、physics201096不变。新科学工作等待Yi签OR1，无队列。


2026-11-15。Yi已批准v2、新增6000秒及自主修复实现问题。已修复并继续到原工程采样硬帽；现在需要科学优化配置审阅，不是等普通实现批准。

[P1权威报告](CONTACT_GOAL_EI_V2_P1_ENGINEERING_RESULTS.md) · [机器读数/身份/成本](contact_goal_ei_v2_p1_engineering_results.json) · [NaN修复](CONTACT_GOAL_EI_V2_P1_GRADIENT_ERRATUM_01.md) · [同提议诊断](contact_goal_ei_v2_step_diagnostic_partial.json) · [已签合同](CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_APPROVED.md)。

已通过：源修复12真实前缀/66对；mean/sample各12/12完整资格；4测试父组/24reward分支、11强Pareto pairs来自3有信息父组、父组等权100%同向。含实际停止、release后4s及同态原生恢复。原失败全部保留，不声称全域reward正确。

真正执行策略未学习：E4块4096、I3块3072；Actor接受0/32、0/24。只有辅助E16/I12与value32/24更新。tail/FiLM/selector权重逐值同初始化、相关Adam全空。Adam提议joint KL约.16–.26超过原.02，全回滚。原提议joint .188249485/Gaussian .188249483/categorical2.78e−9已逐值复现；缩放网格未完成、不改正式步长。

恢复：两臂新进程模型、已有Adam/RNG、有效物理/历史、后续采样与一轮更新逐值同；仅验证确已存在的预测/表示/value状态，没有训练后tail Adam可验证。

G07不通过：失败1024+E4096+I3072=8192帽；I缺第二P3块且仅5完整回合（E7）。不补采、不虚报4块、不正式训练。正式E/I=0、Dev任务=0、sealed未读，无Student或FB长跑；GPU0释放，无队列。

新增stage 7807.703624job秒；P1 3837.064558/4500；修复677.307783/800；201,096/950000 ticks；R2 29345.631881/57600，独立6000未用。下一是审共同可接受优化步及低层/selector尺度，不放宽KL/std/reward/物理。新规则和实际复测规模须另行明确，未执行。BC500、共享原BFM/PD、main及旧结项不动。
