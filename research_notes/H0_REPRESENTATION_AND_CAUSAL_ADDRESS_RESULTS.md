# H0结项摘要：公开Goal可寻址；B_I增量未证

2026-11-15，Issue66批准的一次性H0已按锁完成，未启动正式E/I、PPO或Student。

- **已证实：**六fit源父组×四持续命令有真实后果差异；四独立评价父组/16真实分支，P-E 4/4、P-I 3/4满足≥2cm且质量合格的局部Goal效应。P-E的FOLLOW−HOLD为5.09/6.08/5.07/5.93cm；I在stop仅1.69cm，保留未过门。关节硬越界/失支撑均0；最大loaded-slip约0.004m/s、yaw excursion约6°，精确值见表。
- **已证实：**同进程及新进程完整物理/历史/RNG/solver恢复与真实2s执行逐值一致；同命令重复的完整控制/物理亦逐值一致。K40与原K80每分支raw/applied公式最大差0，原BFM/PD未更新。
- **B_I独立价值未证：**三组与P-E选相同命令、物理结果相同；stop选更小动作，进展不够2cm但误差/力可能有局部取舍，不能称P-E全指标支配I。不能凭B谱/低训练loss宣称I胜出。
- **相容而未唯一证明：**结果编码会改变寻址取舍；当前候选和Goal范围很窄，16D表示有效秩约3.14。未证明低有效秩就是因果失败原因，也未证明冻结底座缺能力。
- **未证：**P-E四组均为HOLD→standing、FOLLOW→arm15，未证明其选择真正依赖不同接触状态；I有一次状态相关改变但没有额外行为收益。更合适的持续行走、高层每100ms重规划、完整接触释放恢复、随机鲁棒性、generalization及正式Actor适配仍未验证。K40中距FOLLOW仍9.64cm，不能称完整中距Expert达标。HOLD的几何/力成绩不是主动跟随成功。

**决策：H0到此完成，不产生H0.1/新Q通行证。进入匹配可训练Actor的E/I合同设计；raw Goal/outcome是强基线，B_I只能凭下一轮实际增量保留。以下完整数据与协议边界。**

## Detailed evidence: finite outcome prediction and addressing

Issue66 signed H0 v0.1; old G17/QA1/QA2 FAIL remains unchanged. This is a single bounded frozen-BFM probe, not formal E/I behavior training or full FB.

## Actual identity and data

Three F1 exposed source/partner ancestors ×4 native command plans, actual first2s reused; three new train-only K40 ancestors ×4 real commands. Four independent source+partner ancestors reserved for physical evaluation, K80/K40 balanced. Whole ancestors kept apart; source prefixes actually executed; no pose-only restore. Original command/source manifests retained privately. Old F1 historically exposed dev is explicitly FIT, not a new blind claim; current R2 Dev/Sealed never opened.

P-E predicts24 standardized raw future outcomes. P-I uses16D B/F with outcome reconstruction and fixed variance/covariance. Same current1639D restricted input, full proposed20×256 legal plan, hidden64/32,1024 groups×24, same parent-balanced sampler. Parameters E435512/I440024; additional I capacity is disclosed, no exact capacity-match claim. T is0-parameter finite selection, not arbitrary continuous latent generalization.

K40/D8/10N is an approved new domain; no cross-K comparison is attributed to method. K80 unchanged calculation checked256 cases; every new branch has raw/applied force audit. Original PD, BFM and physical reward unchanged. No old high actor loaded; historical F1 visited prefixes can originate from old policies, while fit labels are actual native frozen-BFM commands.

## Goal intervention results

| Parent | K | Method | FOLLOW−HOLD wrist progress cm | error worsening cm | Quality pass | Local effect pass |
|---|---:|---|---:|---:|---|---|
|560570cc402e|80|E|5.0949|-2.0239|True|True|
|560570cc402e|80|I|5.0949|-2.0239|True|True|
|92acbb87e917|40|E|6.0791|-4.7067|True|True|
|92acbb87e917|40|I|6.0791|-4.7067|True|True|
|c375fb9aff4b|80|E|5.0713|-1.3478|True|True|
|c375fb9aff4b|80|I|1.6884|-1.4916|True|False|
|d313f47d9377|40|E|5.9336|-1.1635|True|True|
|d313f47d9377|40|I|5.9336|-1.1635|True|True|

Independent local effects: P-E 4/4, P-I 3/4. Need at least2 independent ancestors, ≥2cm progress, ≤.5cm follow-error worsening and retained physical quality. This threshold is an engineering effect criterion, not statistical significance.

## Future prediction on actually executed heldout commands

| Parent | K | distinct executed commands | E MSE | I MSE | constant MSE |
|---|---:|---:|---:|---:|---:|
|560570cc402e|80|2|0.000982|0.001406|0.028432|
|92acbb87e917|40|2|0.013196|0.012231|0.037174|
|c375fb9aff4b|80|3|0.001000|0.001092|0.024354|
|d313f47d9377|40|2|0.002039|0.002668|0.032354|

Selected-command observations alone do not identify the full unexecuted candidate ranking. Duplicate method/Goal commands are deduplicated for prediction error. Full2s commands are real interventions; no predictive branch is called Student on-policy data.

## Representation diagnostics

Fit-only B variance, spectrum/effective rank, same-phase cross-parent raw/random retrieval agreement, phase-only predictor and per-parent actual-command spread are in the machine summary. Raw self-nearest agreement is definitional, not evidence for a learned representation. Small fit-only retrieval or loss cannot establish causal control. Goal-shuffle reported by reusing frozen opposite-Goal choices is an algebraic matched-branch control, not additional physical trials or statistical replication.

## Evidence levels and scope

Established: actual fixed-endpoint models and real Goal branches above; count satisfying local effects P-E=4,P-I=3. Compatible mechanisms and unproven hypotheses must be separated below. No universal contact policy, general intent inference, Actor adaptation benefit or I-over-E significance is claimed. Four heldout parents are prospective for this fit, train-designated, not sealed.

H0 ends once. Next is a matched trainable-Actor contact/Goal E/I design, not H0.1, a new Q ranking exam, candidate sweep or Teacher repair chain. Any future training needs separate approval.

## Physical burden and control quality

| Parent | K | Method/Goal | cmd | error cm | force N | slip peak/p95 m/s | yaw excursion deg | q_PD jump deg | real q jump deg |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|
|560570cc402e|80|E/HOLD|0|4.834|3.993|0.001275/0.001196|0.318|0.191|0.016|
|560570cc402e|80|E/FOLLOW|2|2.810|2.490|0.003718/0.003573|5.597|12.142|1.057|
|560570cc402e|80|I/HOLD|0|4.834|3.993|0.001275/0.001196|0.318|0.191|0.016|
|560570cc402e|80|I/FOLLOW|2|2.810|2.490|0.003718/0.003573|5.597|12.142|1.057|
|92acbb87e917|40|E/HOLD|0|14.342|6.530|0.001919/0.001870|0.869|0.242|0.021|
|92acbb87e917|40|E/FOLLOW|2|9.636|4.630|0.003389/0.003304|5.281|12.160|1.073|
|92acbb87e917|40|I/HOLD|0|14.342|6.530|0.001919/0.001870|0.869|0.242|0.021|
|92acbb87e917|40|I/FOLLOW|2|9.636|4.630|0.003389/0.003304|5.281|12.160|1.073|
|c375fb9aff4b|80|E/HOLD|0|4.782|4.130|0.001416/0.001364|0.334|0.169|0.014|
|c375fb9aff4b|80|E/FOLLOW|2|3.434|3.020|0.003962/0.003891|5.913|12.166|1.065|
|c375fb9aff4b|80|I/HOLD|0|4.782|4.130|0.001416/0.001364|0.334|0.169|0.014|
|c375fb9aff4b|80|I/FOLLOW|1|3.291|2.915|0.001947/0.001787|0.929|1.558|0.205|
|d313f47d9377|40|E/HOLD|0|4.876|2.128|0.001352/0.001305|0.150|0.087|0.008|
|d313f47d9377|40|E/FOLLOW|2|3.713|1.861|0.003731/0.003651|5.945|12.160|1.080|
|d313f47d9377|40|I/HOLD|0|4.876|2.128|0.001352/0.001305|0.150|0.087|0.008|
|d313f47d9377|40|I/FOLLOW|2|3.713|1.861|0.003731/0.003651|5.945|12.160|1.080|

All jumps in this table are internal to the saved window:20ms q_PD target changes differ from5ms actual joint motion. They do not imply the joint executed the target jump instantly. Per-axis force/error, applied cap fraction, raw demand, foot paths, support, root and actual dq are in the JSON.

Successor-feature disable: 8/8 goals have an already executed matching standing branch; remaining physical entries explicitly UNMEASURED. Offline candidate tie selects standing, not a demonstrated new physical outcome. Actual native standing baseline exists in every fit group; no extra independent-evaluation baseline branches were silently added.

The main intervention is a single current-state choice of a sustained2s plan with native50Hz body feedback. It is not a demonstrated high-level controller that replans at every100ms, not full-episode contact/actual-release recovery validation, and not stochastic deployment robustness.

## R1b physical-unit audit and restore evidence

Parent-balanced selected-command standardized MSE: E=0.0043040, I=0.0043495, constant=0.0305786. Both forecast selected outcomes better than this constant on these four parents; no test-time refit or unexecuted-candidate ranking claim. Detailed wrist/root cm and mean-force N errors are in JSON. The same command duplicated across methods/Goals passed full physics/control exactness on 7 comparisons.

## R1a补充：相同phase内当前身体/负担最近父组（只读描述性）

没有重训或改变Goal效应判据；按当前可部署本体与实测力选择最接近的另一父组，再比较该组四个实际未来结果。匹配距离和力差完整保留；没有事前绝对邻近阈值，故只称nearest-available，不能把距离大也说成合格近邻。entry仅一fit父组、无跨父组支持；20条来自其余五个父组，仍是fit-only、非20独立父组。

{"raw": {"n": 20, "fraction": 1.0}, "randomB": {"n": 20, "fraction": 0.9}, "B_I": {"n": 20, "fraction": 0.8}}

## 最终账本与停止

H0累计1802.902486/5400 job秒（含450秒准备和200秒最后归档/发布的保守inclusive记账）；R2累计21137.928257/57600秒。25164 actual/reserved全流程5ms ticks，其中12400新增分支、12764冷启动/reset/native/source前缀；reserved包含actual，不相加。旧F1复用4800tick不计新增。实际新fit12、evaluation16、恢复proof3；两头各1024组，新增RL/Actor/U均0。

末次完整inode去重实扫：H0 237740424 bytes（0.221413GiB）/2GiB；R2 3805158840 bytes（3.543830GiB）/12GiB。此后仅少量报告/源码/账本与Git元数据增长，不冒称重复全盘实扫。分阶段成本与全部source/job/head身份见JSON；科学运行均b77ed34，后续只有只读报告。原生asset manifest与F1身份逐值匹配ea9c3323…；原BFM/PD、旧模型、失败/erratum及旧封存结项全部保持。

全部预定阶段完成，GPU重作业退出，无待启动队列；H0只此一次，未开H0.1、独立Q、正式E/I/Student或新接触。下一合同仅[DRAFT](CONTACT_GOAL_EI_NEXT_CONTRACT_DRAFT.md)。没有因看到结项结果改候选、loss、源父组、Goal效果门或模型终点。
