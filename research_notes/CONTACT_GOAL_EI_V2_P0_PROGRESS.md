# E/I v2 P0实施进展 — 源修复已实测，开始P1冻结资格

2026-11-15。Yi追加“这种你自己的实现问题可以自己修正后继续”后，已按[修复授权](CONTACT_GOAL_EI_V2_SOURCE_FIX_AUTHORIZATION_01.md)推进；[首轮失败](CONTACT_GOAL_EI_V2_P0_SOURCE_RESULTS.md)/[source erratum](CONTACT_GOAL_EI_V2_SOURCE_ERRATUM_01.md)保持，不再把该已修复问题当停工点。

## 已测证据

- 固定12个arm原生准备前缀（train6/dev6）全部完成。66对共同窗口中真实身体/控制历史均不同；无新硬边界。执行祖先字段只使用实际生效参数，原source ID另留来源。没有读取sealed案例或执行dev伙伴/Goal任务。
- 初始原Actor与零FiLM输出逐值相同；E/I公共tail/FiLM/Goal/selector/value逐值相同。policy＋representation参数E5,229,081/I5,227,321（相差0.0337%）；修正后共同value输入1331、参数407,041。没有dummy容量。原BFM/normalizer未更新。
- 隐藏伙伴字段扰动保持deploy/context/Actor输出不变；当前可见force改变FiLM末层梯度，初始零影响但学习通路存在。Actor force字段没有真实active、anchor或future。schema见[字段锁](ei_v2_locks/contact_schema.json)。理想力和理想自身里程计均是仿真假设，非真机结论。
- 一条实际100ms Gaussian命令完整执行与2＋3子步执行，5ms物理、20msaction/q_PD、raw u及实际执行值、reward和采样RNG逐值同。FOLLOW/HOLD同一个原生参考anchor；reference前算后完整handoff恢复。
- 实际categorical＋Gaussian、真实Goal变化与mid-option/mid-command存盘：old joint logprob重算误差**0**（原3e−4断言未改），解析KL0；终止与时间截断的解析GAE fixtures按FP32期望逐值相同。
- 新进程加载修正fixture后，模型、fresh空Adam、全部RNG、完整有效Task/solver/BFM/控制历史，以及后续实际raw动作/物理/评分逐值同。**这仅证明空Adam状态；非空Adam要在P1真实scratch更新后再验。**

## 明示的实现修正与失败

1. 缺stage相对pointer helper：同SHA/头文件/ABI验证后等价复制，旧exit1保存。
2. P0梯度测试处于旧physics的no_grad入口：仅在测试反向局部enable_grad，旧exit1保存。
3. 新状态V曾误用旧action-Q helper，含刚选target-z/slot。未做任何更新即发现，修为合同规定的当前C-P＋公开Goal；现相邻决策的next-V输入和下一段实际V输入逐值同。旧write01仅debug，不是科学checkpoint。
4. restore01误指旧value schema的write01，strict形状加载正确拒绝；修正为write02并在重放前检查schema，restore02全部严格PASS。没有移植旧critic或宽松读取。
5. 原文“1024条最多205段”在跨真实5子步边界时需最多206个片段。仅修正buffer计数上界，保持所有真实draw、插值、1024数据、joint概率与时间GAE；不能删/复制一条来凑205。此口径已在[执行锁](ei_v2_locks/protocol.json)单列。

以上全部无正式学习更新。§9修复子池保守圈记547.307783/800秒，已包含在相应作业/准备账中，不重复加费。已公开源修复是用户明确追加授权，不伪称数学等价路径修复。没有改reward系数、std、KL、物理边界或科学成功门。

## 下一实际工作

P1固定10个train资格父组/12回合，先冻结mean与真实sample完整执行（真实stop、hold、release后4s），同实际release状态另跑原生reference。再按门做同条件reward排序、E/I各4×1024实际scratch更新及非空Adam恢复/成本投影。满足后才从干净共同初始模型进入P2；scratch不冒充32k正式模型。没有旧Q考试、额外Teacher或新Student。

[完整固定参数/顺序/父组](ei_v2_locks/protocol.json)已哈希锁定；34/44 train与16/20 dev身份不改，sealed不读。源修复改变的是实际准备历史，不能只靠换hash声称独立；12条真实结果已补齐，站立分支保留原实现。

截至本进展包：新增3477.172930job秒，P0 2867.172930/4000；真实5ms ticks 29502，正式E/I行0/0、优化器step0。旧R2累计25015.101187/57600；新增6,000额度尚未使用。完整成本含全部失败、source/reset/reference/IO与保守CPU准备；单GPU0串行，100GiB自由盘护栏不松。该报告是**进展，非阶段结项或任务成功**。

原始小摘要：[机器结果](contact_goal_ei_v2_p0_progress.json)。
