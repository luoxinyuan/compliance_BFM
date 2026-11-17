# BFM-CTFT v1：A2单次工程尝试结果

2026-11-17，执行runtime `d3a3e582c3172c73682e60d9830b7b7836aebd82`。
Yi本轮明确回复“批准”，对应A2请求；[签署](BFM_CTFT_V1_A2_SIGNED.md)和原失败保留。

**判读：编译完整更新通过；新进程下一更新逐值恢复失败。A2整体未通过，按单次尝试停止，不发车U/T。**

## 实际改变及样本身份

仅设置`torch._functorch.config.donated_buffer=False`，保持compile=True、原loss/GP、FP32有序梯度、五Adam、1024 logical batch及全部原门槛。没有切换eager、删除梯度项、改变概率/物理或容差。

重放本stage两个原生祖先的1200条实际无外力g0后继；没有新物理采样、Teacher查询或正式U/T数据。本轮是一次性接线和learner恢复资格，不是接触行为或完整Task/solver恢复资格。aux尺度1仅为此工程probe占位，正式共同校准仍未完成；全部probe权重/replay不得进入正式初始化。

## 完整更新与存盘

|实际检查|结果|
|---|---|
|原eager与修复eager：梯度、参数、Adam|逐值一致，expert/train/GP全部保留；2次discriminator-only测试后恢复初值|
|2组Actor冻结更新|通过；245.544732s与24.192758s，含各自首次编译/重编译|
|2组Actor联合更新|通过；68.579039s与5.631802s，前者含Actor首次编译|
|B、target-B及原robot normalizer|上述4组均未改变|
|完整learner快照|8,735,914,318bytes（约8.14GiB）；50.997721s；hash manifest、fsync、原子目录发布|
|不中断的下一实际联合更新|完成，6.227795s|
|新进程读取并执行同一下一更新|初始读取逐值通过；实际更新完成但结果不一致；56.429379s含冷编译|

快照包括全部模型/targets/normalizers、五Adam、实际梯度、z-buffer、RNG、实际源replay/抽样元数据、参考连续片段、模块mode与requires_grad。读取路径显式重建新输入并严格加载，不调用会重新初始化或解冻B的通用load。

快照model.pt SHA256：`1d88255887b1908500c21dc41b43a9afce425ad55d65ed95e38617e6bf642bc0`；learner.pt：`b455e2f698ea3a260ccfd1fd6445f5ce6413862f6f2877d87cc2992693e20dd5`。
二者是一次性工程权重，不能称获准科学checkpoint。

## 恢复失败的已证实范围

1. 文件hash读回、模型/Adam/梯度/z-buffer/normalizer/模块mode/已保存RNG的**初始数值状态**逐值一致。
2. 下一实际expert与train mini-batch的完整内容指纹一致；更新后的Python/NumPy/Torch CPU/CUDA全局RNG指纹一致，z-buffer一致。
3. 判别器的参数、梯度、Adam及其本次判别器损失指标仍一致。不能把后面的`Q_discriminator`指标差异误称判别器参数已经分叉；它属于后续Actor价值评价。
4. 下一更新后共有1320条指纹路径不同：模型469、梯度280、Adam状态559、标量指标12。F、CPR critic、aux critic、Actor及相关targets受影响；冻结B/target-B/原normalizer没有变化。
5. 现存可观测的不一致已出现在F更新相关的`F1/M1/fb_*`输出；这是**子系统边界线索**，不是已捕捉到时间上首个不同算子。没有逐算子中间张量存档，不能虚称已找到唯一根因。

**尚未证实：** 不一致的绝对/相对数值量级、是否仅FP32舍入、具体首算子；编译图缓存/张量布局/中间随机值等可能来源。指纹能证明不等，不能证明误差很小。全局RNG首末一致也不能单独证明编译内所有中间随机张量一致。上述假设不写成结论。

额外只读查验输入filter和normalizer源码：多字段拼接按cfg.key显式列表顺序，未发现由set迭代造成输入列跨进程换序的实现。未运行新的GPU前向、优化器或第三个配置试验。

## 身份、成本与停止

私有证据目录为独立stage的`runs/s0-a2-approved-01`：`joint/progress.json`、`ordered-gradient-regression.json`、完整checkpoint与manifest、`uninterrupted-update5.json`、`restore/comparison.json`、各级failure/cost及`jobs.json`。失败不覆盖。

监督进程13169，串行子进程13173/17041均已结束；GPU0无本任务剩余进程。其他GPU/任务未操作。

监督sidecar：1145.083939s，包含开始前准备时间和串行两子进程，子进程486.824374s/120.730194s不重复相加。收尾只读审计、报告和发布另在最终A2计费快照列明。本阶段实际执行6组完整agent.update（4组预定＋2组下一更新对照），另2次discriminator-only等价测试；正式训练更新0、新采集transition0、新physics tick0。峰值CUDA reserved14,415,822,848bytes，未越原显存护栏；新增输出约8.14GiB，未越A2的12GiB。

本次没有认定原FB算法或接触学习失败；证明的是A2解决了运行时报错，但严格learner恢复门尚未通过。正式规模N未锁，接触全链、材料点滑移和来源划分资格仍PENDING。

最有根据的下一项是同一存档输入下对F路径作逐层/逐算子、含实际z/target action/布局与编译随机中间量的定位，先识别数值差从哪里进入。该建议尚未执行，不自动追加更新、切换eager或放宽逐值恢复门。
