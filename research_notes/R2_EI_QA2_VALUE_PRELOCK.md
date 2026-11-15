# Issue65 QA2 有限时域实现预锁

本文件在首次新价值头优化之前锁定；以已签Issue65正文为上位合同，不改旧QA1/G17。

- 标签仅取E3500真实stochastic μ；旧40条mean分支只作P0已暴露诊断。
- 输入为原 `privileged(observation)` ＋实际29D action＋实际插值z256＋实际zi32；原robot normalizer不变，不输入time、phase、parent或future spec。新Q专用标准化仅用fit父组，逐字段mean/std，std下界0.01；这是新小Q头的输入映射，不改变BFM/Actor输入。
- 单一共享remaining-H模型：输入拼接H/50，Linear256—SiLU—Linear256—SiLU—Linear1，输出 `-softplus(net) * H/50`。H0精确0。H1..50是V2自举必需的剩余时域，主报告仍为H5/25/50；没有增加超过1s的预测任务。
- 一个Q头、fresh Adam1e-4，无旧Q56/QA1权重继承。V1与四个V2版本同网络/初始化seed650801、batch64、固定1024组、MSE、target EMA0.01。V1用完整MC；V2 n=1/5/25/50，实际有效n=min(n,H)，bootstrap只取同一μ日志中t+n实际action/z输入；H=n无bootstrap。V1和V2-n50数学目标逐值相同，必须输出终点等价核验。
- 所有臂完全同一fit父组×物理phase×remaining-H抽样序列（seed650211），索引包及SHA在P1保存，不看成绩改预算/采样。父组—phase分层均匀，层内合法窗口均匀；不把帧数当独立父组。
- 原3500有6个独立父组。fit固定none/首near/mid/stop-continue四父组，heldout固定zero/最后partial-near两父组。旧数据的高低成本独立组数若不足G2，照实UNDERPOWERED，不调整split凑资格。它不构成P2前的循环G1门。
- P2在拟合前固定原train manifest索引6(mid hold)、8(stop hold)、9(near entry)、11(zero)，各不在3500父组内；具体spec/time/seed在不可变P1包。每组mean、±draw0、±draw1，候选seed650600+idx；共同续跑随机μ CRN650701/650702；前缀seed650500+idx。评价1s，40条上限，不补父组/seed。
- P2预测在真实评价之前冻结；候选首次20ms覆盖一次、之后冻结随机μ；初始完整状态、历史和RNG共享。候选CRN差和排序符号反转按正文UNCERTAIN保留，不删除。
- 只保存固定初始和固定1024终点，不按旧40挑节点。所有模型/targets/Adam/RNG/输入标准化/索引游标/完整source物理和replay引用保存，独立进程读回核验。新head干跑仅forward/backward，不调用optimizer.step。
- A0默认NOT_AUTHORIZED；即使μ价值合格，没有额外同π_mean后续资格也不执行确定性Actor更新。

预算：P0/P1后先用真实读取/标签/冻结head干跑/存盘重投影，再启动优化。V1≤1500s；四个V2合计≤1700s；P2≤1800s/26000ticks；各子额和总9000s/3GiB/38000ticks不变。失败与工程修复按§9单列，不挪用P2/A0额度。
