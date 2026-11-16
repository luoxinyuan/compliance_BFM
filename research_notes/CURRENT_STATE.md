# Current — E/I v2正式预算修订已授权，运行准备中

2026-11-16。Yi已直接授权按实测修订完整正式训练/评价/存盘预算并发车，后续确认继续；[完整预算](CONTACT_GOAL_EI_V2_FORMAL_BUDGET_AMENDMENT.md)与[最新资源签署](CONTACT_GOAL_EI_V2_FORMAL_RESOURCE_SIGNOFF_20261116.md)已归档。原OR1工程资格与历史失败保留。正式仍是每臂P2 12k＋P3 20k，完整100 Dev、48 train sentinel、12 P3入口及最多8消融，不使用sealed。

本轮新增预算56,000 job秒、10GiB输出，combined stage上限74,000秒/18.4GiB；物理新增1.1M ticks、stage1.5M。完整成本区分实测与投影。当前正式训练0，未把预算批准或源码准备写成发车；新入口/评价/资源与项目历史总账核验后才启动。

## 以下保留历史快照

# Current — Issue66 OR1有界资格完成，正式训练未启动

2026-11-16。[最终报告](CONTACT_GOAL_EI_V2_OR1_FINAL_RESULTS.md)：D0/D1/D2及E/I新进程12条＋完整更新逐值恢复通过；新工程采样8192/8192，两臂各4块、Actor各接受12步。I最后恢复379.938秒/4145ticks/新增训练行0，GPU0释放。

输出和I恢复补齐授权已归档；原tick轮询超支及所有失败保留，不追溯改判。OR6按实测热块/IO投影尚未放行；不自动32k/Dev/sealed，不宣称接触行为或BI增量成功。下一项仅提交完整正式预算修订。

## 以下保留旧进度快照

# Current — Issue66 OR1：P3工程IO修复，按已批合同续跑

2026-11-15。D1 E/I完整通过；E的P3首1024行接受4次Actor、累计9次，入口6/6及none/zero通过，但actual-release期间监督器ESTALE中断，未宣称D2通过。用户已授权修复后继续。[工程修复与续跑02](CONTACT_GOAL_EI_V2_OR1_COUNTER_RESUME_02.md)只调整计数IO互斥、异常退出元数据和完整block1续跑。CPU并发/故障注入通过，真实恢复尚待运行。

剩余固定清单：E补release与第二P3块，I入口/两P3块/保护，两臂新进程完整恢复。已有采样5120/8192，剩3072；不重采首块、不改五α/KL/物理/科学门，不自动32k/Dev/Sealed。旧失败和成本保留，分段秒数超支按已签补充报告。

## 前次进度快照（保留当时事实）

# Current — Issue66 OR1：秒数补充已签，继续补齐D1/D2

2026-11-15。Yi最新明确批准[OR1分段秒帽补充条款](CONTACT_GOAL_EI_V2_OR1_SEGMENT_SECONDS_SIGNED_AMENDMENT.md)，原文逐字归档，执行源码340930d。**P1、D0/D1/D2、修复池仅秒数超支改报告制；stage47000秒、240h总账、原输出帽、新采样8192条、五α及全部科学门不变。** OR0按Yi决定接受现状、继续执行，具体资源私有记录。原OR1分账6000秒作为历史参考持续核账，不独立阻断合同内阶段。

已有实证：D0真实五α回溯PASS；新rollout E/I各1个P2块、各2次Actor接受，原logprob误差0，none/zero/实际释放恢复各3/3 PASS。见[此前实际续跑记录](CONTACT_GOAL_EI_V2_OR1_RESUMED_RESULTS.md)。尚不能称主动接触或BI行为收益。

D1现已完整通过：E/I各2048条、各2个P2块，Actor每块接受2/3次、累计各5次；第一接受后的none/zero/实际释放恢复各3/3 PASS。所有已有首块与合格检查复用，新增采样累计4096/8192。E/I断点严格恢复和同状态行为均值身份核查通过，未混用另一策略旧数据。

下一按原合同执行P3入口6项随机资格、各两个P3块、首次接受后3项保护与新进程12条＋完整更新恢复。为守原输出帽，新增无损expected-checkpoint引用与actual-delta格式；真实完整Actor/Adam/物理/RNG签名CPU测试逐值PASS，单bit差异完整保留；不冒称GPU恢复已过。详见[IO资格与成本](CONTACT_GOAL_EI_V2_OR1_LOSSLESS_VALIDATION_IO.md)。

OR1累计3850.106770job秒；D1为2573.457050（原2000、超573.457050），修复/IO343.165106（原100、超243.165106）依已签补充报告，非停工门。超支原因为早期两次IO故障/恢复、原超短作业时限、真实接受Actor回溯成本与必要的无损存盘资格；新投影见IO记录及实际后续sidecar。stage已计11657.810394/47000，采样4096/8192、次数/输出/科学门不变。

旧stage与R2引用不覆盖，项目去重总量缺口不虚报。

E/I完整P2、P3、接受Actor/selector后的新进程12条＋一整更新恢复及OR6仍待验证。正式32k、Dev任务、Sealed、Student/FB长跑均不自动启动；原BFM/PD、BC500、main不变。

## 历史记录（以下为当时状态，以本页顶部为当前事实）

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
