# QA2 工程记录（不改科学判决）

本轮已由Yi批准Issue65全文及§9；以下仅处理路径与JSON封装，不改标签、采样、模型数学或门。原失败日志、运行源码与数据包均保留。

## E1 stage-local-f1-pointer-layout-dependency

首次边界：P0只读探查在导入`f1_pointer_layout`时，`ctypes.CDLL`无法找到新stage的`prep/pointer_layout.so`，尚未装载replay或做物理/optimizer。根因是新隔离目录尚未复制既有审计依赖，不是MuJoCo状态合同改变。

仅复制原已资格的so/json；SHA256分别为`8e7d74e1ffb947f1ccfc82b083dbf91b6433e2da847c29dc8e96ef1709dfb439`、`fa3c671ca8162ed1917bf0f7fbdcf2b701aa18f41342b202dbe5a554708df5ff`，与原逐字相同。独立进程重新读取原round14后，完整agent、3500行、manifest digest均与批准起点完全一致；真实物理/历史/RNG源引用不改。没有需要恢复的更新轮。此零物理阶段没有新增物理续跑证据，不将文件读取说成新动力学恢复资格；原完整恢复资格仍为继承的已验证路径。

family修复次数1。证明文件`prep/dependency-repair-proof.json`，实际CPU核验7.739893秒；首失败冷导入/定位另按120秒保守上界计账。

## E2 p1-json-numpy-scalar

runtime984244d的P0/P1作业39.235176秒：全部连续性/时间/source/奖励符号、父组划分、标签与索引构造断言通过，`data.ptz`写完并读回digest一致；末尾`json.dumps`拒绝摘要中的`np.int64`。没有任何optimizer/物理。原日志保留，不能把失败作业说成一次正常完成。

修复仅将JSON中的NumPy标量转为同值Python标量，不重建标签、不重抽索引、不覆盖`data.ptz`。新隔离runtime69f20cc从该原包完成摘要，包SHA和schedule digest前后一致，合成int64/float64 roundtrip通过。其余源码修订是本轮尚未执行的价值头实现预锁，非对失败科学配方的修订。

family修复次数1。证明`prep/p1-serialization-repair.json`；恢复作业29.191413秒。新worktree及修复/回归准备计240秒保守上界。共享工程池包含首失败39.235176秒及上述各项，不因其已记P1而在总账重复加算。

## 范围

两次故障发生前更新总数0、物理ticks0，恢复点仅为不可变真实原始数据。原3500与40条mean分支、旧Q56、所有Adams/targets和物理source都未覆盖。每family最多一次，复发须按§9停止，不以本报告授权第二次修复。没有放宽概率/数值/物理断言，旧QA1/G17 FAIL不改判。

## 发布复核fixture（非科学运行故障）

最终公开JSON复核首次把布尔字段名source_reference_exact误当成私有路径而报错；实际内容不含私有路径或SSH地址。改为检查真正的路径/地址内容后通过，公开数据与科学门均未改，原检查记录保留在publication-review-fixture.json；费用含最终发布300秒上界。没有新实验修复或重跑。
