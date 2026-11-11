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
