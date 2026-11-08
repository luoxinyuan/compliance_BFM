# CURRENT_STATE — Issue53小续跑结项：交互标签无整体增量，PPO按用户决定停止

最后核对：2026-11-09。**没有训练/队列，未启动下一阶段。** [Issue53及续跑](https://github.com/GaoE05/robot-research-bridge/issues/53) · [本轮结果](INTERACTIVE_INCREMENT_RESULTS.md) · [精确指标/成本](interactive_increment_evidence.json) · [长期目标](RESEARCH_OVERVIEW.md)。Issue52独立Expert PPO保持结项。

## 当前结论与冻结基线

- 有限域Teacher→ideal-force Student最小闭环仍成立。保留全部历史I/C0/T30/S0/D2及原#53结果；当前干净原生线基线是**旧数据BC500**，SHA `a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`。不是原S4000，也不与历史S0混同。
- 本轮复用#53修复后的34正确标签（2个Student起点＋32个Teacher相关后继），按原形态35%混合、1000监督，开发选500。固定6cm误差/力由BC3.200cm/2.671N→2.899/2.399，但59cm持续由10.794/7.201→11.689/8.400，停止9.127/5.916→9.345/6.092。无严重边界，但当前形态**未显示整体增量**；没有追加标签或加训救结果。交互1000负结果保留。
- 用户预定分支因此选BC500作为Student自身PPO共同起点。原BFM/PD冻结、actor1639D仅部署许可输入、fresh独立critic/Adam，无在线Teacher覆盖。seed5401第4批开发72cm改善至8.303/5.403，固定第7批却退化18.494/8.922；seed5402第4批8.463/5.447，用户要求在该节点评价期间提前停止。两组中期均有真实反馈收益，但**尚未稳定保留**，不把中间模型当末轮成功。
- 第二seed不是完整等预算终检，没有新PPO封存矩阵；不宣称双seed方法成功或失败。原6/59cm封存已在#53暴露，本续跑是指定固定复核，不称新盲测。当前不升级交互版或PPO替换BC500。

## 质量、失败与成本边界

- 交互版确实在走：59cm移动期root约49.62cm（BC47.53），但Y力约4.24N（BC1.68），保持力9.47N（BC8.57）。不等于失去walking；三维协调/持续负担更差。释放后不再评价旧anchor跟随，合理收步不判失败。
- 原#53跨reset接触回调闭包错误修复与40个受影响标签/模型隔离全部保留。本轮只用正确34，数据SHA3062cf5c...16b657，未恢复错误混合数据。
- 首次PPO调用漏日志raw_adv_std，768实际决策后退出、更新权重未保存；仅补原日志统计，实际保存批次通过原概率/KL/Adam断言。失败计费，原两seed8批共同缩为7批，之后用户再停第二seed；不扣失败成本重跑满额。
- 实际新增1000监督、0完整Teacher查询、9216 RL决策=失败768＋5376＋3072；含CPU预留约3876.93/12000 job秒，约.93/4GiB输出。第一组与第二组按用户效率授权同卡并行，cgroup20 CPU核/RAM约193GiB可用/GPU约80GiB核验，重叠225.94s，job-wall仍逐进程相加。所有owned训练/渲染进程结束。
- FB影子仍仅原#53资料：接触12状态F/物理Spearman均值−.208，接触前无实际力8状态−.121，top4分别保留5/12、0/8，无新增胜出。当前提名器暂停，不否定整个FB。原96查询的用途/分族成本见本轮报告，无重跑。

## 位置、身份与接力

- 个人长期审查分支`luoxinyuan/compliance_BFM:gaoe05/research`，最新文档提交见Issue53最终续跑评论。真实监督runtime5d8facce...、正确评价20880dbf...、两个PPO相同runtime **b1751077a1a5d59bb08a1a2386cbc7db3e8b20fa**，收尾/渲染1f2a95f...；文档HEAD不冒充训练源码。
- 服务器stage：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/interactive-increment-20261108`，ledger及user-early-stop保存；模型`runs/fit-01/interactive-500.pt`、`runs/ppo-clean-5401/candidate-{4,7}.pt`、`ppo-clean-5402/candidate-4.pt`。完整SHA、数据来源、成本、负例均在报告/evidence。原BC在上一stage`interactive-contact-pilot-20261108/runs/bc-fit-01/old_data_only-500.pt`，从未覆盖。
- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`，原PD/Direct256D norm16/10-50-200Hz/K80D8/10N及reward未改。Student1639D无anchor/伙伴真值/clock/任务ID/active真值/MPC phase/hidden；ideal-force不等于G1真机传感已具备。
- Mac `~/Projects/compliance-bfm-diagnostics/videos/interactive-increment/` **2条1×真实qposMP4＋INDEX**，已SHA/时长/全解码/代表帧核验；[视频解释](INTERACTIVE_INCREMENT_VIDEOS.md)。旧视频原位置保留，不再复制大量辅助文件到视频目录。

## 下一决策（未启动）

保留BC500；本轮收尾后停止。若再研究反馈优化，先在一个seed上让已出现的实际收益稳定保留，再做独立seed复核；不把更多seed、更多标签或更长训练当默认修法。用户已要求以后主动利用获准NVIDIA资源，启动前实时核查CPU/GPU/RAM/I/O，预算内并行有独立价值的工作并记录效率，不机械填满显存、不扩资源或实验主题。

新会话先读本页、OVERVIEW、续跑结果和Issue53最新评论，再核对实际ledger/进程/模型SHA。本阶段已消费且用户停止，不自动恢复。Git仅审核源码/配置/小型摘要，不上传权重/原始轨迹/凭据；不改main、共享BFM/PD、SoftSONIC或他人任务，不做真机。旧Library2026-10-28仅历史快照。
