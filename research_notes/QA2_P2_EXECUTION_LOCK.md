# QA2 P2执行细则（模型终点冻结后、物理之前）

四个父组/time/seed已在首次V1拟合之前写入P1数据包；本文件仅明确执行接线，不重新选父组或阈值。

- 首20ms动作取mean/±draw0/±draw1。draw由各父组预锁CPU generator产生std.01、clip.06并在合法[-1,1]动作空间限幅。
- 首动作是明确物理干预，不从μ再采一次，故不消耗后续CRN随机数；其后49个20ms全部由冻结原μ实际Gaussian采样执行。两个预锁CRN 650701/650702对五动作分别重置相同外生随机流。物理内部RNG和source历史保持完整真实源状态；CRN只替换声明的后续探索随机流。
- 每个父组先真实source前缀，再冻结μ真实执行至预锁100ms边界。保存有效MuJoCo/solver、Task/callback/本体/力/z/动作历史、RNG与source identity。每分支从同一完整snapshot恢复，effective canonical状态和前缀RNG逐值校验，然后重置该CRN。禁用qpos-only拼接。
- 在各候选真实物理之前保存五个Q固定终点的预测。Q输入使用实际slot0、实际previous-z、真正当前高层target-z与首个插值z；首控制再断言同输入、action、z及q_PD一致。
- 每模型都报告，不按新物理结果选模型/补训。V1和V2-n50若模型/targets/Adams完全相同，明确是数学等价重复对照，不算两个独立成功。
- 真实分支均为前瞻评价，非训练数据/Expert标签/PPO样本。四独立父组、40分支的层级不混淆。
- 按正文：两CRN均值作G；回报差>.25 max(1,abs(meanG))则候选量级UNCERTAIN；动作差符号翻转则该对UNCERTAIN。所有对均保留，报告总对数、有效可辨对和不确定对；不能剔除后高准确率冒充≥4父组通过。
- G1≥70%且至少4个独立有效父组。G2高成本至少2父组/低成本至少2父组，全部父组门与训练拟合global/phase基线同时检查；不足则UNDERPOWERED，不能重选near时刻找低成本。
- V1训练高成本macro NMAE10.377%、V2-n25为8.528%，满足“至少一个模型可以拟合训练高成本”的P2入口；现有heldout高成本独立组0，近距低成本误差仍超门。入口不是正式G1/G2通过。
- A0仍缺Q_pi_mean匹配后继资格，默认不执行。P2与A0未触发额度不能挪去增Q更新。
