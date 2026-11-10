# 包一：FB 恢复更新的首个数值分叉

事实起点 `55cfe85928365ed4933c40a4b62ebc18e6324e12`。
本包完成定位，**没有修复、没有通过严格恢复门、没有发放长跑资格**。
旧 D2 结项及两个 erratum 原样保留。

## 匹配合同与实际验证

在原有 eager、deterministic-algorithms、CUBLAS `:4096:8` 配置下，重新执行
1024 次真实采样和第一次完整工程更新，继续采样 13+27 次；同旧停止包新进程
恢复后继续的 27 次实际采样比较。更新前模型、六 Adam、replay、z-buffer、
物理状态、RNG 指纹精确匹配旧归档。真实 expert/train mini-batch 各 1024 条固定保存，
另在恢复路径核对实际重新抽样的值及抽样后 RNG，随后用相同缓存 batch 完整更新。
不是只声称 seed 相同。

三次被追踪更新各记录 **5443 个算子**、六 optimizer 的前向/反向/pre-step/post-step、
四个 target soft-update 边界。不中断与两次恢复分别精确复现旧两条最终 model/Adam
tree digest，因此追踪未将原失败改造成另一种结果。未初始化 allocator 字节没有定义数值，
仅其 shape/layout 参与比较；这不放宽任何数学结果、概率、物理或恢复断言。

原始证据：私有 stage `fb-native-audit-20261110/runs/trace-{control,resume}-*`；
公开 [完整定位摘要](fb_native_bias_artifacts/update-trace-summary.json)、
[身份](fb_native_bias_artifacts/identities.json)。实现：[trace_update.py](trace_update.py)。

## 首个差异：先区分调度，再区分数值

- 第 0–269 个记录（含输入准备、discriminator 前向及损失）一致。
- 索引 270：恢复路径 `aten.slice_backward`，不中断路径 `aten.detach`。
  这是反向执行顺序差异，**不能把它本身称为同一张量的数值误差**。
- 对反向结果按算子/shape/hash 交叉匹配，恢复索引 373 的 `aten.add.Tensor`
  出现不同的 1024D 中间累加值；中间和不同不自动代表最终梯度不同。
- discriminator pre-step 最终四个梯度不同：`trunk.0.weight/bias`、`trunk.1.weight/bias`。
  首层权重梯度 shape `1024×783`，后续 Adam 尚未执行。

### 首层最终梯度的精确算术复现

从第三次追踪保存实际 CUDA backward 加法的输入；只在 CPU 做 IEEE FP32 加法，
不进行任何优化更新。不使用近似容差：

```text
A、B、C：同一真实反向图的三个实际矩阵梯度分量
不中断：((A + C) + B) → c5ae4cde…da9845
恢复：  ((A + B) + C) → 3ea577e9…80460
```

两种括号顺序分别**精确匹配**两条最终梯度的完整 SHA256；
最大绝对差 `4.76837158203125e-7`，`182830 / 801792` 元素不同。
完整分量 hash、表达式与结果见公开 JSON。
这已定位到 **discriminator 首层反向梯度的 FP32 累加顺序**。
没有仅凭“GPU 有随机误差”解释，也没有把差异归到 Adam 读错、replay 不同或 forward 输入错配。

为何恢复后 Autograd 对这些分支采用不同调度顺序，是否涉及叶节点执行缓存/历史，
本包尚未继续定位；不声称已经证明某个未保存缓存是唯一状态来源。
该路径包含 WGAN 高阶梯度，但没有关闭 GP 的物理/训练对照，不能据此宣称 GP 目标本身有误。

## 六 optimizer 与传播边界

|optimizer|pre-step 自身最终梯度不同的 tensor 数|
|---|---:|
|discriminator|4|
|forward|0|
|backward|0|
|critic|77|
|auxiliary critic|0|
|actor|46|

discriminator step 后首先出现模型/Adam 差异；F/B 与辅助更新保持一致，
随后 critic、actor 及 critic target 出现差异，符合它们使用已更新 discriminator 的控制流。
不能以全模型 hash 已不同来误报每个 optimizer 均有自身更新问题。
恢复进程没有旧 `.grad` 缓冲而不中断进程有，这是记录到的执行状态差别；
各 optimizer 都 `zero_grad(set_to_none=True)`，本报告只对其真正参与 step 的梯度比较，
不把未使用的残留 `.grad` 当成参数差异。

旧最大 model 差 `2.3841858e-7` 的终态由本轮相同 tree digest 重现。
这个量级本身不证明严重行为退化，但严格恢复失败仍然是失败。

## 成本与下一项单一工程干预

本包3个受监督job墙钟合计 **907.82s**（共同准备/CPU分析预留另计）。
本包完整工程 `agent.update` 共 **4 次**：参考的预备更新 1 次、三次被追踪更新；
新增实际 native transition **1118**，物理重建另计、不进入新训练 replay。
没有正式 FB/PPO/Student 训练。
实际 runtime：参考 `a1834af33ac136bbac046dfed26448b43656c670`；
首次恢复 `570cbf002c770ccb3db61453974a1a3b4972f429`；
操作数捕获 `aa62fe541037bb6a59f453fe4b9c366e10977523`。

下一项最可判别的工程修复建议：仅固定 discriminator 梯度各分量的累加顺序，
保持数学损失/所有数值合同，重做同 batch 的不中断/新进程精确回归。
**未执行该修复，本轮到此停止。** 不重启旧 1M。
