# G17-QA1 独立工程勘误（不覆盖旧G17）

2026-11-15；状态：已定位，未关闭。首科学正确性阻断：切换C分支前的完整起点hash不同，C更新0。保留原运行与traceback；不回写旧模型、old G17结论或成功门。

## 证据

- `research/interaction_fb/qa1_trial.py:121–122` 对同一个内存source再次`agent.load(s['agent'])`，随即全hash断言失败。
- `approved_learning.py::load` 直接将传入各Adam state交给`optimizer.load_state_dict`；PyTorch `_process_value_according_to_param_policy` 对非capturable/fused的`step`返回原tensor。
- 同真实Q56第一参数状态、CPU read-only load fixture：Q1/Q2 step storage pointer均相同；未backward、未optimizer.step、未初始化CUDA。Q56磁盘step56，U512磁盘step512。
- 原source manifest以及完整agent SHA/digest与签前一致；U56完整agent等原source；U512所有非Q可变部分digest相同。

直接证据见私有stage `prep/failure-readonly-01/diagnostic.json`；公开摘要不携带原始轨迹/路径。影响仅本新入口的多分支共享内存初态。不能据此声称历史其它恢复路径出错；历史单进程/新进程恢复已有独立验证。

## 最小修正建议（未施作）

`agent.load(copy.deepcopy(pristine_initial_agent))`，每个分支独立对象与完整校验。回归必须包括一次真实隔离Q更新前后源缓存不变、下一分支model/target/Adams/RNG初态一致，以及loss/sampler合同不变。没有通过这些检查前不得重跑C。当前QA1§6要求一次只读诊断后STOP_AWAIT_DECISION，因此没有自动修复或继续。

## 另一个启动错误（已工程关闭，未改变学习）

首作业49dc30a在模型加载前因CPFS Unix socket Errno95退出，物理/优化0。新入口顶层补入已有`m11_ipc`，使forkserver子进程也复用Linux abstract socket；独立Manager/Queue读写fixture通过。修复源码9d48d4d，科学参数不变；原log及11.242017s保留。它不涉及Q、物理状态恢复、概率或采样定义。
