# CURRENT_STATE — Issue56：双轨正式队列已启动

最后实时核对：2026-11-10 09:33 CST（01:33 UTC）。运行身份不变。
[Issue56](https://github.com/GaoE05/robot-research-bridge/issues/56) ·
[正式合同](DUAL_TRACK_CONTRACT_V1.md) ·
[已获批P1–P9](DUAL_TRACK_SCIENTIFIC_DECISIONS.md) ·
[工程实测与执行进度](DUAL_TRACK_FORMAL_PROGRESS.md) ·
[新入口勘误](DUAL_TRACK_FORMAL_ERRATUM.md) ·
[Master批准](https://github.com/GaoE05/robot-research-bridge/issues/18#issuecomment-6739473031)。

## 正在做什么

用户已明确批准全部P1–P9。首臂`expert-kl002`已完成128批/98304真实决策，
开发选模仍iteration0；该选中模型独立封存评价已完成，不计新增学习成功。
第二臂`expert-kl008`核对时88/128批、67584决策，选模仍iteration0。
正式FB1M尚未启动，Student未启动。局部开发成本改善伴有滑移/偏航或任务退化，
没有新策略替换BC500。**接受PPO更新、数值成本降低不等于可靠学习收益。**

新增结项必答项：将首臂第48批长距退化按起步/持续/保持拆开，比较与小请求
改善状态的重叠、实际训练覆盖和可复核的更新/梯度证据，区分任务干扰与关键
状态覆盖不足，允许并存或证据不足；详见[batch48归因要求](DUAL_TRACK_BATCH48_ATTRIBUTION.md)。
尚未完成该分析，不把本要求写成已证实病因。不改变运行源码或训练合同。

队列固定：KL.02 Expert → 其选定checkpoint封存评价 → KL.08匹配Expert
及封存评价 → 全新FB1M → 0/250k/1M开发与1M封存/有限reward-prompt评价。
同一RTX PRO5000仅一个重GPU作业；异常暂停并留证，不放宽数值、不自动重试。
两个KL臂各98304决策（合计196608），同初始化、std.008、分布与预算。
FB首次只运行1M真实50Hz transition，保存不可覆盖0/250k/1M；不自动5M。
后续Student98304是条件性授权，本队列不自动启动Student。

## 科学合同与边界

- 原腕部、原BFM/PD、K80D8/10N、Direct合法256D与原控制合同。
- 真实原生standing/arm-ready/moving源等概率；回合近/中/长/停/零/无接触
  20/20/25/20/10/5%。训练停止/继续成父组；宏组采样保持这些回合边际比例。
- 开发30、封存60为父组目标，停止/继续同父组；帧/seed不冒充独立组。
  原生资产仅3类。工程已暴露3个开发父组；首臂选中初始化已做封存评价，
  不按封存反选或追加训练；第二臂与FB后续评价仍按固定队列。
- 高层gamma.99/100ms，FB.99**.2/20ms；原逐点成本按实际时长积分。
  高层内部5个native reward也按物理时间折扣。真实终止、截断与释放区分。
- FB新增97D当前伙伴/接触字段，旧机器人统计冻结、新字段固定SI尺度；
  保留原LayerNorm域并零初始化新增权重。discriminator仍robot-only。
- FB为独立可训练基座副本：fresh6Adam、fresh replay、Identity辅助reward
  归一化、fresh辅助critic；原系数与16updates/1024采集基准。
  原生因果z先验用于在线采样，不在线调用I/C0/T30或手写force-to-motion。
- 当前两线均为特权Teacher；未来受限Student不能继承这些额外信息权限。
  BC500仍为ideal-force基线，不代表真机力传感已经实现。

## 已验证与未证明

已通过：零新增输入权重的原函数逐值保留、3源短程物理一致性、真实终止
后继/延迟reset、训练与物理RNG隔离、高层768决策完整PPO更新与独立评价、
修改后FB真实采集/完整6优化器更新，以及模型/Adam/replay/实际29D动作
物理前缀的精确读回。最新stage-smoke-04还验证了wrapped timeout正确
bootstrap且不加摔倒尾罚；底层默认时限2250控制步=45s，覆盖本轮任务。

工程失败全部保留：reset关键词遗漏、测试CPU/CUDA比较、FB重复Task
回调绑定、原Box无界元数据序列化，以及两次课程测试脚本/测试假设错误。
它们不构成正式学习结果，也未放宽原概率断言。见独立勘误。

尚未证明：任何正式新Teacher收益、两KL配方优劣、1M FB效果、90个父组
全部具备物理资格、新Student。工程FB256步采样58.6native/s不是完整训练FPS；
两次compiled update仍含warmup，稳态吞吐与正式完成时间待真实日志。

## 精确身份与接力

- 固定运行代码：`147b22b684b0c03a613559fa6d5e5881d380ab7f`。
  文档发布HEAD可能更新，运行worktree不跟随修改。
- BC500：`a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`。
- 原BFM：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- FB完整读回工程runtime：`b9862178a6355043a19a809b08c27c5aa6a2fe94`；
  高层完整工程runtime：`6ab8b6e42e641b5bafeec4839b025ed066a43526`。
- 私有stage：`experiments/dual-track-formal-20261109/`；`runtime/`固定，
  `repo/`仅用于后续文档集成。无新研究分支，公开个人入口`gaoe05/research`。
- 队列PID380408；首臂supervisor380634、worker380646（核对时身份，勿当永久PID）。
  先读取`queue-status.json`、`jobs/*.json`、各`runs/*/learning.json`和
  `selection.json`，再核对进程/GPU；不得重复启动或覆盖目录。
- 实时账本`ledger.json`是有时间戳快照；jobs侧车和实际学习文件是实时依据。
  旧启动快照（2026-11-09 23:15）保守计费10107.25s，其中共同10006.28s，冻结100.97s；包含#55
  4076.20s保守carry及3600s准备预留。总240h按112/80/32/16管理。
- 存储护栏100GiB自由空间；输出上限64GiB。共享自由空间非独占配额。
  不修改main、共享BFM/PD/权重、SoftSONIC、他人任务，不做真机。
- 当前没有新行为视频；历史视频与Mac路径不动。取得实质结果后按实际
  qpos生成少量对照，不将参考动画或工程测试冒充新策略执行。

新会话先读本文件、OVERVIEW、合同与Issue56，核对实际进程与模型SHA。
5403b7不晋级、BC500及历史M38等全部保留；历史完整结果见#53–55及原报告。
旧Library2026-10-28仅历史快照。此stage尚未结项，不自动进入下一阶段。
