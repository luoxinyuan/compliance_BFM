# Issue62 Adapter Envelope V1：逆映射前置核查触发明确停止条款

2026-11-12。已读取并按[最新批准评论6827917070](https://github.com/GaoE05/robot-research-bridge/issues/62#issuecomment-6827917070)启动独立准备。**没有启动新的资格物理、B/C训练、dev或B_I探针。** 停止原因是§3的“原映射逆变换无法逐值验证则暂停”及§9(1)/(3)，不是预算、GPU不空闲或Envelope物理失败。

## 结果与最小可复现证据

原动作合同`action-contract.json` SHA `705fa1d4072b31523ef943c1121c8652f2ab09ba4d1bd4066b57adf8f30b36dd`，BFM仍`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。原动作映射：`offset + clamp(action*5,-5,5)*scale/5`，全部运算保持原FP32顺序；clip=5，合法normalized action∈[-1,1]。全部29D scale正且有限、margin后lo≤hi。实数解析逆不存在结构性歧义，但浮点目标的可表示性另需核验。

存档B19 sample第0个控制状态，t=11.800000000000s，关节`left_hip_roll_joint`：

|量|FP32实际值|
|---|---:|
|原BFM action|0.16202348470687866|
|加adapter和原噪声的action|0.33060264587402344|
|q0|0.284076988697052|
|按批准公式生成q_requested|0.40407699346542358|
|下邻执行action|0.23046556115150452|
|下邻原g输出|0.4040769636631012|
|上邻执行action|0.23046557605266571|
|上邻原g输出|0.40407702326774597|

两动作的有序FP32 bit key分别[3194748714, 3194748715]，**相邻、之间没有第三个FP32 action**；q_requested严格介于两输出之间。原g正scale、逐操作单调，覆盖[-1,1]全部float32的二分查找证明这不是简单解析逆公式写错：此目标在当前数值合同下没有精确preimage。最小不可避免差 **2.9802322387695312e-08rad**。不能通过换一个近邻action使其逐值相同。

26存档帧×29关节中，566项被Envelope真实约束；GPU解析逆在其中247项不逐值往返。该计数不等于全部都无精确逆：严格无preimage证明针对上表最小反例，不泛化为247个都不可能。CPU独立执行也找到无preimage反例（不同关节，误差1.4901161193847656e-08rad）；CPU/CUDA逐操作舍入可不同，不要求二者逐位等价，正式阻断依据为NVIDIA后端反例。

检查只针对**确实触发投影**的项；没有用未触发投影时可以直接保留原action的冗余往返误差制造失败。没有修改adapter常数、原BFM、物理、reward或任何科学/概率门。

## 含义与下一项必要决定

- **已证实**：当前FP32执行接口不能逐值表达批准公式产生的所有q_requested；最新评论明确禁止猜测逆映射或放宽后继续，因此在新物理前停止。
- **未证**：Envelope物理无效、adapter被清零、训练不可行。约3e-8rad的数值间隙本身不构成这些结论。
- **可审查的最小修订建议，尚未实施/授权**：将公式输出区分为理想`q_requested`与实际`q_exec=g(a_exec)`；采用确定性的、朝许可区间内部的可表示action舍入，零修正/未触发投影直接保留原action；以实际g(a_exec)为唯一物理/critic/replay重建真值，所有调用路径逐值相同。单独记录理想与可表示目标的ULP/绝对误差，并证明实际修正仍在原Envelope范围内。这可以避免放宽实际关节边界或物理门，但会修改当前“q_exec必须逐值等于理想q_requested”的数值合同，不能自行作为已通过。

不建议为了这个微小数值间隙改共享PD或全链float64，也不通过无声allclose宣称精确逆成立。若Yi确认可表示执行定义，再完成完整重投影、预锁状态、五路径/恢复测试和物理资格；仍须按原资格门决定是否B/C重训。

## 实际完成与未完成

已完成：最新授权全文读取与归档、个人隔离worktree、GPU0/CPU210核配额/RAM615GiB/shared389GiB/I/O检查；29关节机械表；原映射/可逆性检查；CPU与NVIDIA反例；无物理算子前后向microbenchmark。

GPU算子前后向100次热测：batch1 p50/p90/max=0.6698/0.7093/1.3176ms；batch32=0.6817/0.6988/0.7324ms。只对输入求梯度、无模型/Adam更新；不代表完整agent吞吐或路径梯度资格。

**未完成/未启动**：由于逆映射前提失败，没有把Envelope接入Agent，没有预注册新资格状态、完成完整B/C/dev/IO预算投影，也没有启动任何新物理分支。机器锁明确BLOCKED而非假称已锁定可发车。两份完整预算投影仍待数值合同解决后形成，不用算子热测替代。既有A128/B19和原on/off数据不变。

## 身份、资源与记录

读取仓库基准`e60fff41446e4246faa4ecc471fb399c4eb4c95a`；实际执行独立脚本SHA `1ab0bad8338ca61c813cfefc5554fa37b253af97a3174df9946c659b64275cd4`，发布同内容源码`research/interaction_fb/adapter_envelope_inverse_preflight.py`。输入只用已暴露B19存档，没有读取sealed。CPU/GPU输出`prep/inverse-{cpu,cuda}.json`、完整日志/sidecar `jobs/`，独立stage `interaction-envelope-v1-20261112`，机械表/证据机器文件[链接](adapter_envelope_lock_v1.json)。

GPU job PID 2542038，exit0，含启动/IO墙钟10.709575s，结束后GPU0回79MiB/0%；CPU进程未完整计时故保守计120s，准备/分析/checkout/发布保守计1200s。新增计费上界1330.709575s，累计 **34028.588967s＝9.452386/15h**；输出3.052665/8GiB。没有动用条件18h/16GiB扩额，项目240h总账继承。新增physics=0、训练更新=0；未放宽断言，不操作其他GPU/进程，不改main/共享资产。
