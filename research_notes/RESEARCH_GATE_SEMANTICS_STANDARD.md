# Research Gate Semantics Standard v1 — 每道门先对齐现象、测量与反例

**适用：** 后续所有 Human/Humanoid Interaction、BFM、SoftSONIC 科研合同的资格门、主动停机门、晋级门、安全门、信息权限门、资源门。该规范本身不是实验授权。规范源于 R2 E/I 入口 v1.0 把「防大幅偏航」误写为「5ms yaw 总变差≤15°」的实证失配。

## 必填三列（逐门，缺一不得声称合同可执行）

| 防什么 | 怎么算 | 拿已有证据各举一个会通过与会失败的例子 |
|---|---|---|
| 明确实体危险/错误模式，例如「未指令的大幅持续偏航」而非笼统「不稳定」；声明是硬安全、资格、训练保护、统计判据或仅诊断 | 源字段、单位、所用坐标系、精确时间窗和t0、采样间隔、对照身份、聚合算子、阈值、等号含义、缺值处理；禁止将q_PD与实际q、yaw TV与最大偏离、root路程与净位移混同 | 提供已有真实trace/报告的数值与源SHA、PASS和FAIL各至少一例；若只有其他模型的同语义示例、存档推导或故障注入fixture，必须标注证据等级；没有真实正/反例就填UNMEASURED，不谎称新模型已经通过 |

**实例：** R2 v1.0 sampled none 的5ms yaw TV=39.323°、yaw range=3.730°、net=−3.529°；旧A128 none的连续净偏转1081.13°。若想防大幅偏航或自转：用完整5ms unwrapped yaw相对起点最大偏离≤15°，前者PASS（max≤range3.730），后者FAIL（max≥|net|1081.13）；TV仍保留为振荡/平顺性的另一变量。历史按TV的旧失败记录不追改。

## 签署分级：已知判据与未来才会产生结果

- **运行前必过的资格/安全/权限/恢复门**：三列公式必须可执行、历史证据与故障注入测试能证明该门会拒绝实质性坏例。若没有现存真实坏例，需从已有存档构造明确的故障注入fixture并标注synthetic；如果连measurement都无法从原物理字段复算，则SIGNOFF_BLOCKED。所有当前真实训练结果不得先填PASS。
- **训练后才会知道的科研晋级/表示probe门**：签署时允许PENDING_UNMEASURED，但必须有能运行的pass/fail程序逻辑测试、预先固定的数据split/统计分母/有效类条件、未达证据时UNDERPOWERED的处理；只有未来真正通过后才晋级。没有正负类变异不得宣称probe失败或成功。
- **数据/资源门**：必须带量纲与总计规则（physics reset/source/replay/failed、job wall、GPU/IO、replay/checkpoint/临时峰值），同时给实际通道通过案例与已发生的历史越限案例；不能把其他项目的越限当成当前项目已失败。
- **别名/坐标系**：未测相同完整状态的对照不能用同父组不同state冒充；缺同态参照一律UNMEASURED。
- **签署后不得追改**：任何审查时发现门想防和测到的不同，应新建有版本号的修订条款，并保留原严格结项。只有用户重新授权后生效。

## 每个失败出口必须预登记，允许Codex自主做有限诊断

每个门增加以下机器字段，不计为三列主展示：gate_id、failure_class、automatic_next_action、required_original_state、allowed_readonly_inputs、max_new_physics_ticks、max_wall_seconds、max_output_bytes、resource_guard、after_diagnostic_state。

失败后按优先级：**真实物理硬界/信息泄漏/恢复不可信立即停相关job→保留全部原始证据→执行唯一预写的只读归因或在合格同状态上做一次冻结安全对照→报告并停**。所有可预见岔路可在合同签署时统一授权，但不包括自动改变阈值、奖励、模型训练配方、seed/父组、封存、共享环境或扩大预算。

**失败诊断不等于重新放行训练。** 对仅用于观测的warn（如TV大但yaw excursion很小、其余安全与质量门通过），先自动做不修改物理的只读分解，标明WARNING_EXPLAINED后可继续原路径；不能把警告悄悄当成硬失败，也不能把真实硬失败作为警告忽略。

## 审计记录与版本纪律

每次研究合同必须附机器可读gate_semantics_review.json及可读三列表，包含门ID、字段/窗口/阈值/对照、历史source+SHA、PASS/FAIL数据级别、缺格与fallback、诊断资源上限。脚本必须执行拒绝fixture并证明符合规则时放行；新合同签署前先锁所有统计和诊断分支，训练后只能报告事实不能换口径。

首次落地合同：[R2_EI_TRAIN_CONTRACT_V1_1.md](R2_EI_TRAIN_CONTRACT_V1_1.md) 与 [R2_EI_GATE_SEMANTICS_V1_1.md](R2_EI_GATE_SEMANTICS_V1_1.md)。原运行仓库 luoxinyuan/compliance_BFM:gaoe05/research 在当前GitHub集成中只读，需Codex在用户签署、核验后同步。
