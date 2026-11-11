# Issue58执行错误与未关闭项（2026-11-11）

原始输出、源码、模型与旧erratum全部保留。本页不追溯修改旧D2严格门，也不放宽容差。

## E58-1：compiled连续更新失败

运行d757bee，p0a-compiled-control-01。前段工程状态保存至rolling-1064；进入compiled固定16更新时，
fb_ordered_grad.py的多分量torch.autograd.grad遇AOT second-backward RuntimeError。
原jobs/p0a-compiled-control-01.log、runs同目录failure.json、固定输入和已发布停止包保留。
该compiled资格未通过，compiled恢复对照未启动，不将eager通过推广为compiled成功。
影响范围为本compiled工程作业，不涉及正式A/B19b3117或冻结共享模型。
本轮未以删除GP/梯度分支或放宽断言解决。尚待单独修复与回归。

## E58-2：失败子进程内部result.json仍为running

同一compiled作业的通用m10_physics入口在异常时未把初始result.json更新为failed。
实际job sidecar已returncode=1/status=failed，failure.json和stderr完整，队列正确停止该包。
因此引用内部running不能当当前进程仍在运行或工程成功；本轮CPU/GPU状态已核全部停止。
没有覆盖原result.json伪造终态；未来入口修复须单独测试，本页只隔离解释口径。

## 未关闭的状态资格（不是已证明的动力学错误）

p0a-physical-reference/resumed两分支的模型、同状态action、全部physics/control及RNG逐值一致，
但portable最终状态hash不同；只读比较差异集中23个MuJoCo island/稀疏辅助数组。
是否全是未激活workspace尚未证明，当前不能直接删字段/降容差改判。
因此严格门仍未关闭，未启动任何正式FB长跑。

P0-B安全映射减少真实边界但损害滑移/偏航，是预登记物理负结果，不作为代码错误掩盖。
正式A/B没有发现本页错误的影响证据，所有预登记选模拒绝保留。
