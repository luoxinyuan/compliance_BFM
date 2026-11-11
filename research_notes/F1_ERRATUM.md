# F1 L0失败记录（不追溯修改历史）

首runtime a39be585f54b71a213ac65b1171aa6a58f6d7354 在P1-1两个1s重复后，`f1_l0.py:65`完整portable一致性断言失败。job l0-01原sidecar状态failed_no_automatic_retry、原failure.json和两条真实轨迹保留，不覆盖。实际physics/controls/inputs/raws/scores一致，末态25个MjData数组不同，见first-difference数据。不提高容差、不删除字段、不自动重跑。

测速wall记录在完整检查后才统一flush，导致失败时独立候选wall未落盘，这是本L0可观测性缺陷；未用文件时间补数字。后续修复应先逐条原子落盘计时再做严格断言。不存在新reward、模型更新或历史科学checkpoint修改。

其他准备读档时的新worktree尚未checkout完、猜测不存在的旧文件名等只读命令失败，未产生物理或模型产物；GitHub连接器临时传输失败不算实验结果。合同中的Markdown双空格为原文换行，逐字保留，不为了diff-check改变锁定文本。


## 2026-11-11 21:12新授权后的更正与证据

此前“25数组可能为工作区”的假设已补足实际C指针和156条物理干预证据：当前配置这些指针NULL，Python绑定生成未初始化临时数组。额外发现同类iM非零形状字段，其他NULL字段多数shape0。新规则见F1_L0_ARRAY_AUDIT；旧首轮严格失败原判和产物保留，不能声称它当时已经通过。

- `array-audit-01`（bbd207b）：第一个NumPy字段置零后非目标portable hash不同，立即停；只完成3s未干预参考，没有合格干预。
- `array-audit-02`（147a93d）：保存before/after和别名检查；观察到NULL属性反复读取的临时内存变化，包括iM。不是实际有效状态被连带改写。未作合格干预。
- `array-audit-03`（5e5b3a4）：用已安装header的offsetof访问真正C指针，逐一/联合挂入owned零值或存档随机buffer；其他C结构和有效数组/RNG不动。156×3秒全部通过；追加CPU重读字节摘要156/156一致。
- `l0-complete-01`（0b36af7）：存盘map_location=CPU与现场GPU张量被旧torch.equal直接比较，发生设备RuntimeError。不是数值不一致；没有放宽容差。`l0-complete-02`（5f4c82c）用dtype/shape/原始bytes比较，27状态、81次恢复、重复物理和5个共同前缀通过。
- `cold-probe-01`（8bb0de3）：独立进程冷/热测速及3s重复通过。没有清空共享cache。

所有失败计入账本。计时现逐分支写出再断言，避免首轮timings未落盘。只修复状态保存/比较/恢复工具，不改概率、物理、reward、质量门或封存选模。没有新FB update，因此只回填Issue57的portable有效域残余，不改变旧数值更新失败的历史判读。

## L1/L2续执行：完整状态差异与保守恢复（2026-11-11）

用户授权在L1/L2层内自行解决工程问题；科学合同、12h分账、L3独立批准和红线不变。原L0结项、所有失败和状态manifest均保留。

### L1两次停止及分支隔离

- `l1-01`，runtime `f03ed1be5b384ce982fd073b9669b9717742c50b`：完成8状态/128条合格重建后的分支，在`new_tail-sustained-97ff10c4dcd9`完整portable检查失败。高层输入、last-z及归档物理核验已通过；当时没有保存完整diff payload，故**无法从该次存档确定具体首差字段**。不能追溯把后一次发现当作此次已证根因。568.023866762s计入失败成本。
- `l1-02`，runtime `e31b0d2aa9fc3b896887b4a78ee685e8640affcd`：先补全差异payload保存，并复用已完成搜索而仍按原顺序重建前缀。新增2状态/32分支后在`old_tradeoff-sustained-1f7c1c806ff0`停止；保存实际/归档全状态。唯一差异为**已分配的`efc_JT`**，shape2135、666元素不同、max abs差2032.1412678638712。其他数组、metadata、身体/历史/RNG均逐值相同。261.777478793s全部计入。
- 失败状态在通过完整恢复前没有开展搜索。前两次128+32条已验证状态分支保留原runtime和artifact SHA；后续只补未完成的14状态/224分支。合并384条无重复，不隐藏失败、不将部分成功包伪写成原job整体通过。

### 修复与决定性证据

`f42fb177fe3679d0a162ace03b22acb5196279c4`的修复**不是排除`efc_JT`**：先在副本中仅替换该数组，断言其余完整状态均与归档相同；再复制归档全数组到真实MjData并要求全状态逐值一致。任何其他字段差异仍直接失败。

当前MuJoCo3.3.3 dense/Newton配置的源码中，约束乘法dense路径使用`efc_J`；`efc_JT`转置与读取在sparse路径，证据为L0归档源码`engine_core_constraint.c`约390/1998行、`engine_solver.c`约907–916行。字段名或这些源码本身并不替代实际干预。

每次触发该修复，都在同一完整状态用BC500执行3s：归档数组复制前与复制后，physics/control/input/raw/old及v3评分必须逐值一致，且全状态恢复通过。`l1-03`实际触发并通过三个状态：`old_tradeoff-sustained-3077ca809d70`、`new_near-sustained-5f62e065b5ac`、`new_stop-sustained-4ea37e71ee89`。各600tick×两臂，新增3600物理tick，artifact以`*-efcJT-intervention.pt`保留。L2的追加触发数见其checks清单，不能把L1三个次数套到L2。

**比较规则没有新增mask或容差**：原L0已证明NULL字段有效长度0的规则保持；此allocated数组仍全量保存、恢复、比较。干预只支持当前配置与这些状态的“该差异未改变后续执行”，不外推sparse/island配置。旧首轮失败仍为失败；Issue57只补充portable残余证据，没有新FB update或历史更新数值改判。

### 辅助滑移统计修正

首版`semantics`把同脚多个接触点重复累计到持续时长/路径代理，属于**辅助报告口径错误**。原5ms物理、原加载阈值、v3事件成本、reward、候选选择及轨迹均未受影响。e31版改为body/tick去重；完整384条从原始接触行再次CPU复算，并增加真实材料点的下一tick位移。原记录不覆盖，最终语义以`f1_slip_kinematics.json`为准。

旧事件仍按原方式计数；没有借此删除瞬态或提高法向力阈值。材料点路径不是重新物理仿真：`mj_kinematics`重建的全部已存身体位置逐值核验，实际qpos不修改。

### L2进入前的记录修正

runtime `ad93901e9b75791ae918dd37664b7d6f8b8e86bb`在L2启动前固定：按原24状态顺序重建、只在预登记7状态查询；每个query绑定其**当前真实完整状态SHA**，而非重复使用源初态SHA；C-S物理评价时域明确为3s、选择时域0。该修改不改变候选、reward、概率、物理或科学门。无L3入口/自动队列。

所有原始作业和修复记录保留；完整runtime SHA以Git/launch/机器账本为准。报告若引用缩写，不能把发布commit代替实际运行版本。
