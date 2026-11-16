# E/I v2 — 两臂正式训练完成，统一物理评价待执行

2026-11-17 live verification。两臂状态均为 FORMAL_32K_COMPLETE_EVALUATION_PENDING；每臂32000条真实50Hz、P2 12k＋P3 20k，各127次Actor接受。4/8/12/20/28/32k固定保护均PASS；没有新Dev或sealed行为结论。原工程错误、28k停点和逐值恢复证据保留。

训练runtime `08b9c96a4ea3d0c0007e840029ba6a30f3c858f0` 不变。最终包均为各自formal-06/checkpoints/block-node-32000；E新鲜零学习参考仍是formal-E-03/block-initial，不是续跑initial。完整manifest/core身份见机器报告 contact_goal_formal_training_complete.json。

当前账本29443.143559job秒、603881个reserved物理ticks；runs常规文件5.275GiB（sidecar历史口径包含尚在原子临时发布的峰值，不等于最终目录体积）。原56000秒、10GiB、1.1M ticks总帽和5GiB地板＋1GiB完整停止预留不变。

## 独立评价入口（实现不改变训练）

固定既有100回合枚举：E/I终点、零参数Goal-only、冻结原生零U各24（20主回合＋4不同公开Goal），BC500仅4个FOLLOW历史参考。不读sealed、不改终点、不按Dev反选模型。原生零U=原冻结actor＋constant-standing合法z；Goal-only=原冻结actor＋HOLD-standing/FOLLOW-arm15查表。公开Goal干预逐段FOLLOW/HOLD反转，实际发令时刻不变；此规则在任何新Dev执行前记录。

使用独立runtime/进程/RNG与真正原生前缀；所有方法复用同一父组原生onset参考anchor，不在各自onset重对齐。每个完整回合含真实4s恢复。恢复相对root量从策略实际release的完整状态分支，执行同Goal冻结原生standing参考并严格还原；原生参考自身不合格则记UNMEASURED，不能填PASS。

终点冻结权重读回使用原delta_tree逐层哈希解码core；只免读不用于评价的训练replay，不声称这是训练状态恢复。新CPU资格需比较两层fixture与完整loader、损坏拒绝、100case身份及无优化调用。首次维护目录CPU检查因其非stage路径缺少pointer_layout辅助库而在导入前退出，无物理/更新；在stage独立runtime复验，不修改全局路径或共享依赖。

评价总job子帽12000秒、物理400k或总帽余量较小者，失败/原生恢复参考/源前算/IO全计；任一实际硬界立即保存该分支并停止本次作业，后续缺格明确保留。保护PASS不代替任务收益，学习是否超越查表及I的增量均待统一物理数据。
