# OR1恢复对照的无损存盘与读取资格

2026-11-15。保持原输出帽，未删除、覆盖任何旧checkpoint/失败。普通IO实现修订，科学、概率、物理、样本身份与比较门不变。

1. 实际benchmark把study置于外层result中；读取器支持裸study和真实封装，强制内外status一致。真实完成结果CPU测试PASS，不把未运行过的旧恢复路径KeyError写成已发生GPU失败。
2. 原expected签名重复包含完整Actor/Adam/物理，现改为引用已提交不可变完整checkpoint＋原metrics。写后立即读回重建，并与原内存完整签名逐字段逐值比较；有差异则保存原完整签名并STOP，不以平均hash忽略关键字段。老ptz格式继续支持。
3. 实际恢复对照用已有lossless delta_tree相对expected保存完整actual，含所有失败差异；保存后解码与actual逐值核验。源reference、checkpoint manifest、metrics和actual delta各有hash，不能只有一个首动作或只存模型权重。验证模式不再额外落一份无关的fresh未训练step0副本；训练每块完整逻辑checkpoint不减。

CPU在真实已接受Actor的完整签名测试：全字段相同可逐值重建；故意改一位也原样保留；坏hash拒绝。原签名38596081字节，相同状态delta86151字节，一位不同delta86306字节。实际checkpoint→原签名的模型/Adam/RNG/物理/metrics/末尾记录全部逐值PASS；reference506字节＋metrics12388字节，复用原不可变checkpoint，不另存重复权重。两测试实际墙钟19.802968/43.362138秒，另列准备/IO/归档60秒上界，不藏在GPU作业中。

临时原子余量依据已有输出批次实测重估，细节私有留档；原P1 2.2GiB、stage8.4GiB硬帽保持。旧stat计数仍完整计全部输出，不通过hardlink/删旧文件改变口径。新增stage全输出护栏与原P1护栏共同执行。

这些都是IO/CPU资格，**不是**已接受Actor/selector新进程继续12条＋完整真实更新的恢复PASS。后者仍须原GPU/物理完整对照。修复/IO子账超原秒帽按已签补充报告，不能据此放宽科学容差。
