# CURRENT_STATE — Issue56已结项：无新模型晋级

最后核对：2026-11-10；训练停止与CPU存档归因已完成，运行身份不变。
[Issue56](https://github.com/GaoE05/robot-research-bridge/issues/56) ·
[正式合同](DUAL_TRACK_CONTRACT_V1.md) ·
[已获批P1–P9](DUAL_TRACK_SCIENTIFIC_DECISIONS.md) ·
[工程实测与执行进度](DUAL_TRACK_FORMAL_PROGRESS.md) ·
[新入口勘误](DUAL_TRACK_FORMAL_ERRATUM.md) ·
[Master批准](https://github.com/GaoE05/robot-research-bridge/issues/18#issuecomment-6739473031)。

## 正在做什么

用户已批准P1–P9，后于2026-11-10明确授权无效时提前结束FB。
两组高层KL各完成128批/98304真实决策，均按开发规则选iteration0，
选中初始化封存评价已完成，不计新增学习成功；BC500仍为冻结基线。
FB在最后完整落盘33792真实50Hz transitions/512完整agent updates后提前停止，
下一未完成采集块增量无法精确确定。244记录回合中198触发物理质量早停，
后续多个30回合窗口全部早停、多数在接触前，停止理由见[提前停止记录](DUAL_TRACK_FB_EARLY_STOP.md)。

**保存缺口：只有FB step0完整包，没有512更新后的内存模型/优化器/replay。**
不能把初始化冒称末轮；两组PPO末轮candidate128和全部历史模型完整保留。
独立[存盘勘误](DUAL_TRACK_FB_CHECKPOINT_ERRATUM.md)已登记，未修改运行源码或重训复造。
训练队列与本任务子进程已退出，GPU0核对0MiB；没有自动评价、Student或新阶段启动。

本轮正式[结项报告](DUAL_TRACK_RESULTS.md)与[batch48归因](DUAL_TRACK_BATCH48_RESULTS.md)已完成。
重要更正：batch48六条长距中五条误差改善，一条standing较快引导显著退化，
不能从族中位数推断普遍长距能力丢失。主要额外损失在该条件保持阶段，
起步首秒已出现质量差异。累计48批长距占29.96%决策，非整个族没有训练；
固定点梯度未显示强相反方向，但无法从存档确定特定开发轨迹逐帧重叠或唯一因果。
证据缺失已明确记录，不将训练域近邻冒充开发状态事实。

原队列：两KL臂及选中模型封存 → FB1M及检查点评价。
现因用户授权早停已撤销后续自动执行，未达到250k/1M，不能报告相应结果。
原1M批准是上限，本次不是完成1M，也不证明FB方法无效；不自动续训或5M。
后续Student98304是条件性授权，本队列未启动Student。

## 科学合同与边界

- 原腕部、原BFM/PD、K80D8/10N、Direct合法256D与原控制合同。
- 真实原生standing/arm-ready/moving源等概率；回合近/中/长/停/零/无接触
  20/20/25/20/10/5%。训练停止/继续成父组；宏组采样保持这些回合边际比例。
- 开发30、封存60为父组目标，停止/继续同父组；帧/seed不冒充独立组。
  原生资产仅3类。工程已暴露3个开发父组；首臂选中初始化已做封存评价，
  不按封存反选或追加训练；第二臂选中初始化封存亦已完成；FB后续评价因提前停止而缺失。
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

尚未证明：可晋级的正式新Teacher共同收益、1M FB效果、90个父组
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
- 原队列380408、FB supervisor499277、worker499290及已核实子进程均退出；历史PID不可复用。
  先读取`queue-status.json`、`jobs/*.json`、各`runs/*/learning.json`和
  `selection.json`，再核对进程/GPU；不得重复启动或覆盖目录。
- 实时账本`ledger.json`是有时间戳快照；jobs侧车和实际学习文件是实时依据。
  旧启动快照（2026-11-09 23:15）保守计费10107.25s，其中共同10006.28s，冻结100.97s；包含#55
  4076.20s保守carry及3600s准备预留。总240h按112/80/32/16管理；12:30刷新已登记总16.056h（另计存档CPU复算，详见最新ledger），
  冻结45110.10s、FB2605.98s、共同10006.28s（含carry/准备预留），详见新账本。
- 存储护栏100GiB自由空间；输出上限64GiB。共享自由空间非独占配额。
  不修改main、共享BFM/PD/权重、SoftSONIC、他人任务，不做真机。
- 当前没有新行为视频；历史视频与Mac路径不动。取得实质结果后按实际
  qpos生成少量对照，不将参考动画或工程测试冒充新策略执行。

新会话先读本文件、OVERVIEW、合同与Issue56，核对实际进程与模型SHA。
5403b7不晋级、BC500及历史M38等全部保留；历史完整结果见#53–55及原报告。
旧Library2026-10-28仅历史快照。本轮已完成结项；不自动重启训练、修复后续跑或进入下一阶段。
