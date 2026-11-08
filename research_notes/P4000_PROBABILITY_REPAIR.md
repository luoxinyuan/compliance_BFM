# Issue52：FP32概率复核修复与固定单seed匹配对照

2026-11-08，用户授权继续执行；第二个开发seed不自动启动。

## 真实归因与有限修复

保存的原5202第31批、iteration30 exact checkpoint SHA373c4388a3df444f4cf86c419c0300b8d3ffe850973b93970c73077bf598ae43。在原CUDA环境重放：三worker实际raw用相同权重/obs/std及已登记随机数流逐值复现，batch-one old logprob误差0。768-batch actor均值相对batch-one最大差2.98e-7，一条联合logprob差.0003662109越过原.0003；即使logprob改double，actor前向差仍导致.0003159064。此批证据支持batch-shape FP32累积差异，不支持权重或历史错配；不将它解释成学习性能差或宣布所有概率问题均不存在。

可选P4000_CANONICAL_ROW_ACTOR=1让learner批量评分使用与真实采样相同的FP32 batch-one actor前向，再拼接均值；不改权重、normalizer、256D raw Gaussian、实际batch-one控制、std、物理或BFM。默认旧路径不改，旧runtime均完整保留。不是PPU专用低精度/阈值放宽。

真实CUDA检查覆盖初始5201及失败5202：chunk1/16/128/256/768 actor均值逐值相同；联合logprob误差均.0001220703<原.0003；故意改错一维raw动作仍触发原断言。使用相同固定batch/全局RNG时，joint与独立critic的actor参数/Adam逐值一致，独立critic完成18步。该检查新物理/RL均0，总约14.5job秒（两job）。没有把断言删掉或重算/覆盖采样old logprob。

## 为何同一个seed要有两种方法

统一FP32评分改变了learner数值路径与梯度累加方式。不能将旧batched梯度的原5201当作严格唯一变量对照。因此在同一固定seed5201下，从同P4000 actor/normalizer、新critic(seed+17)/Adam，配对两组各32×768：

- joint：仅修正概率数值路径，原critic仍随actor KL停止/回滚。
- independent：相同数值修复，critic在actor后独立完成6轮18步。

它们是同seed的方法对照，不是第二个随机种子，也不是恢复已失败的旧5202。旧5201/5202仍是历史参照。reward/std/KL/任务比例/网络/原BFM/PD/10N/K80D8/10-50-200Hz全部不混改。优先运行independent，再按预登记同预算运行joint；不按终检决定单臂额外训练。

## 资源、评价与停止

沿用24k job秒/10GiB/98304新增RL上限。原实际采样48384，新增两方法49152，合计97536；失败批也计成本。两job各5500s上限，预留900s分析/视频；不能容纳则保存边界。采样提效probe占原≤600s，四类完整物理逐值一致且实测≥1.05倍才两方法共同采用，否则保留完整诊断。

第一真实768样本必须与原同seed初批obs/next_obs/raw/reward/discount/done逐值一致；新critic学习后不要求轨迹一样。开发仍7条件和0/8/16/24/32节点，沿用均值/采样、严重退化回滚规则。新封存不在开发迭代中执行；没有自动第二seed或Student。检查点明确记录critic模式、canonical数值路径、采样诊断模式、实际已采样与已更新决策。

检查通过不等于真实任务改善；需看到确定性力/误差、起步/停止、各任务尾部、旧能力与支撑的物理曲线。结果与失败参见实时single-seed-status及ledger。代码/数值检查不是完整里程碑结项。


发车记录：单seed有限队列runtime e1d6929d297442ad2ab4ebb7912306f292750354，PID164620/start_ticks7305189，boot be0c7b44-26b0-4699-8716-9903210007ab。independent优先，joint随后同预算；第二seed/封存不自动发车。probe已证实逐值物理一致但无加速（0.99346倍），因此两方法均保留完整诊断。运行源码固定，之后文档HEAD不能当训练runtime；学习结果仍待真实开发曲线。
