# BFM-CTFT v1：AOT生命周期追加工程尝试 A2（待批准，未执行）

2026-11-17。基准执行与证据提交0fee565。A1奖励单位和支撑澄清已经Yi批准；本页不改A1，不改历史失败，也不把A2写成已批准。

## 为什么按合同暂停

正文§9：**“同一根因簇整个新stage仅一次，复发/超过2h总repair池/完整性不过/触科学变量即停报。”** 当前一次修复后，AOT与有序多分量反向的生命周期兼容簇再失败，故没有第三次自行尝试。不是物理能力失败，不是已证明原算法不可用。

## 现有决定性证据

- 01：compiled把多个loss输出共享tape；首分量求梯度释放tape，第二分量失败，0组完整agent.update。
- 02：保留tape至最后分量后，原eager与修复eager的每个梯度、参数、Adam均逐值一致，三个loss分量全保留。
- 02 compiled报错明确要求donated_buffer与retain_graph不能同时启用，建议设置`torch._functorch.config.donated_buffer=False`。报错是实现定位线索，不是该建议已通过的证据。

## 请求的具体局部动作

只追加一次同根因簇兼容验证，在任何lazy编译前显式固定：

```python
import torch._functorch.config
torch._functorch.config.donated_buffer = False
```

继续`compile=True`，保留02的梯度分量顺序及tape保留规则。此设置阻止AOT捐赠仍需复用的反向缓冲，不移除专家/训练/GP分支，不改变数学loss、系数、五Adam、批量、随机采样、动作或物理。它可能增加显存或降低吞吐，必须实测，不能预写成功。

## 固定检查与通过条件

1. 原来1200条本stage新原生g0真实控制后继原样复用，初始checkpoint和RNG固定；这些不是正式U/T数据，不能冒充真实接触T0已经完成。
2. 原eager与修复eager的梯度、参数、Adam继续逐值一致；不只比较loss平均值。
3. 同一1024 logical batch，按既有一次性接线试车执行2组Actor冻结更新及2组完整联合更新，全部有限；B、target-B、原normalizer不得改变；五Adam所有权和实际step明确。
4. 记录首次编译与热更新成本、所有失败、peak allocated/reserved和free；原88%/至少max(6GiB,8%)余量不改。
5. 若完整更新可用，追加模型/五Adam/targets/normalizer/z-buffer及RNG的同配置新进程读回与一次固定batch更新等价验证。此处仍不是带伙伴Task的完整正式恢复，后者保留为后续S0门。
6. AOT修订仍失败则停止该尝试，不再关compile、改loss、合并掉梯度分量或扫描其它flag。

## 资源与后继

该新增尝试暂建议≤1800 job秒、≤12GiB临时输出；均计入原S0≤6h、stage40h/120GiB/16M与项目240h，**不扩任何总帽**，并检查原2h工程repair池真实成本。不得删除旧失败或共享资产。正式U/T规模N仍待全部S0门完成后机械锁定。

通过后只恢复原授权S0，继续真实接触/校准/全链恢复/吞吐门；不是额外批准正式训练提前发车。未通过或预算不容纳则报告，不自行延长。
