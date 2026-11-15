# OR1 P3监控IO修复与完整块续跑（工程记录02）

2026-11-15。Yi直接授权“修复后继续，然后告诉我过多久再来问进展”。原签署/秒帽补充、全部科学门及失败保持。本条不是新算法批准，也不是P3通过结果。

## 已证实故障

原运行源码279b828。E的P3首1024行已完成，Actor接受4次、累计9次；joint KL在原.02内，oldlogprob误差0；原P3 sample-entry6/6通过，首接受后的none/zero已完成，actual-release检查中断。监督器读取physics-counter.json连续ESTALE后停止；随后队列读取缺失exit_code产生次生KeyError。不是已证实学习失败，也不是GPU断联。原作业1031.318503秒、新增1024行、36569实际ticks原账保留。

## 最小修复与证据

1. 仅在计数器atomic rename与监督器读取之间使用同一advisory文件锁，排除read/replace交叠；计数的quantum4、exact-final、单调性、科学/资源阈值及三次有限错误重试均不改。持续失败仍停止，不用陈旧读数伪装新计数。本证据不宣称所有ESTALE均由rename导致。
2. 监督器异常收尾也写exit_code；队列使用缺失安全读取，保留首因而非次生KeyError遮蔽。
3. 增加completeP3模式：只接受该source的完整block1及对应manifest/progress，严格restore模型/完整Adam/RNG/物理/history；核对P3行身份与已接受actor版本。入口6项资格复用，none/zero档案逐哈希读回并复算同一quality/protection，仅补中断actual-release。已完成1024行不再优化、不重采；其后仅E1024、I2048新采样，合计8192不扩。
4. 原通用update、五α、KL/std/GAE/reward、安全/输入/分组/expected恢复比较不改。新增锁不消耗训练RNG，不改动作。原源码与失败输出不覆盖。

共享盘CPU压力回归：1001次atomic写、1002次并发读，54.720333秒；真实监督器read_counter经AST提取执行故障注入：暂时ESTALE恢复，持续ESTALE/已见文件失踪/计数倒退拒绝，exact-final缩回不足4ticks语义保持。无模型更新/物理。Python编译与diff检查通过。此CPU资格不代替后续真实P3恢复。

## 剩余固定队列及投影

E严格恢复block1→补release→第二P3块；I原P3入口6项→2块→3项首接受保护；E/I分别新进程12条＋完整更新恢复。10个未完成完整资格单元，原none/zero/入口不重跑；失败中断部分仍计成本。新采样剩3072，原完整评价24界不扩。形式上的result PASS需与合同每块selector梯度/实际参数/概率证据一并复核，不能仅靠汇总最大变化。

按既有热块272.739秒及冷启动/资格：余下约45–70分钟作业，另留10–20分钟完整报告和IO；这只是当前投影，恢复/硬门失败则停报，不自动32k。分段秒数按已签补充报告，stage47000、项目240h、输出帽、物理计数与样本帽均保留。资源细节仅私有锁记录，OR0按Yi决定接受现状继续。新源码SHA与实际启动身份以sidecar为准；本记录写入时尚未宣称真实恢复通过。
