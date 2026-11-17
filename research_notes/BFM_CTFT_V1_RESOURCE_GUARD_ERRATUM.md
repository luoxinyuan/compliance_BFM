# S0资源交接与跨batch数值负结果：独立记录

2026-11-17，Issue #67。不改旧结果、精度、容差、物理或资源阈值。

## 已确认的资源检查实现缺口

holdout源第一次冷启动前的NVML输出已经显示GPU0有另一项目计算进程；启动器未把该信息设为硬拒绝，仍进入冷启动。这是本执行端的资源guard缺口，不归因其他使用者。

受影响范围：私有stage `runs/s0-holdout-sources-01`，outer92.1148676872秒，SIGTERM退出。仅定位、核验并终止本次拥有的进程组；不接管、不终止其他任务。没有完成的源结果、更新或正式训练。尚未提交的物理计数无法从存档确定，按[0,120000]本作业局部物理硬帽保守隔离；不是“实测0”。完整日志和`resource-interruption.json`保留。

修复：`research/bfm_ctft_v1/resource_guard.py @237cbaf`，启动前明确核对GPU0 UUID、外部compute PID、剩余显存、shared disk≥100GiB，拒绝后不得启动子作业；执行中检查点也监测外部PID。CPU模拟的忙卡/错误UUID/低盘fixture标明fixture，不冒充现场证据。

GPU0真实空闲后，相同预锁候选源通过live guard，`s0-holdout-sources-02 @392aaeb`完整记录28个候选，outer698.3963秒。第一次失败不覆盖，也不混入第二次成功成本。

## 独立数值负结果

`s0-workers-01 @baa60bd`的4 worker跨batch动作一致性断言失败。首个失败输入未在assert前保全，缺失明确保留。后续`batch_shape_diagnose.py @2c131ea`使用同类档案重建输入，并非原失败输入逐值恢复；重复观察到跨shape差异，但没有对TF32根因做干预。原容差、precision=high与TF32配置均未放宽。

处理：4 worker不获资格，不继续扩大worker数；单worker已测路线继续可用。没有把显存低占用当成放宽数值合同的理由，没有将该工程失败转写为行为学习失败。
