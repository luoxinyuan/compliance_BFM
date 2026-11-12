# Issue62 修订锁准备结果（非行为 Pilot）

2026-11-12。T0三门结项保持不变。本工作按用户拒绝3,000/臂后的授权，补齐A/B执行、统一评价、schema/容量审计、事务存盘及规模反推；没有启动正式三臂Pilot，没有执行新dev/sealed清单。完整私有stage为`interaction-fb-pilot-lock-20261112`；所有失败/原始工程物理轨迹均保留。

## 已测执行与信息合同

| 臂 | 可训练模块 | 参数量 | 真实750 transitions/6更新，完整进程s | 含reset/update/IO的warm transitions/s | 冷进程transitions/s |
|---|---|---:|---:|---:|---:|
| A | U,Q | 440065 |212.5685|9.5653|3.5283|
| B | U,adapter,Q,robot-only discriminator |658463|204.5707|9.5491|3.6662|
| C | B的共同模块 + B_I,F_I |947679|222.5448|7.8693|3.3701|

U185856、Q254209、adapter160669、disc57729、B_I31008、F_I258208。共同模板以同顺序初始化后移除无关网络；实测U/Q、B/C adapter/disc初始tensor hash及三臂初始实际action一致。原大型robot Actor/F/B/normalizers全冻结；这不是全基座FB更新吞吐。C增加289216参数；不能把C>B唯一归因分解结构，容量/目标对照仍缺。

共同deploy808D；privileged1538D；B_I80D；U1064D，Q/F_I1855D。当前partner真值仅训练特权路径，隐藏真值扰动不改变deploy输入/动作已逐值测试。A/B只有零32D占位，不含隐式交互B；C的z_I来自固定公开目标原型，不来自当前真实partner B。目标原型的有效性仍为方法假设，不冒充资格化反馈目标。原动作投影、5子步插值、PD、K80D8/10N和10/50/200Hz不变。

每轮250真实50Hz数据、2次完整更新、batch32。A/B/C更新总耗时分别.2346/.2600/.3117s；IO总耗时3.6947/3.8228/4.5103s。GPU allocator peak约.736/.740/.746GiB，RSS约7.66GiB。CPU物理、源前缀和IO占主导；不由低显存推断可增加实验。短测吞吐方差不能归因C架构成本，更不能推断未实现并行环境性能。

runtime：A/B `47d2215718d19ebe88e85cff27a4c449b0fedf6c`；C/恢复/中断 `4f79752c85e550484801bfaf3ff93b6b22db767e`；修正评价汇总 `22651a9fad4f2e7c03bc229be3ab5a6915e0be05`。每个sidecar保存精确运行SHA，不用最终文档SHA代替运行SHA。

## 完整增量checkpoint

旧full-package历史不改。新`delta-v1`在每轮保存全部mutable核心＋新增数据delta，SHA/计数/readback/fsync/原子提交。core含完整网络/targets/所有有效Adam、normalizers、目标/z-buffer、Python/NumPy/CPU/CUDA/Task/physics RNG、Task spec/callback可重建绑定、solver/物理/控制历史。delta包含replay、原生保持片段、全5ms轨迹。所有旧段被manifest显式引用，不只保留latest权重。

C不间断 vs 从round1新进程恢复：round2、round3的agent（含targets/Adam）、rows、native、physical_stream、各RNG、cursor/spec、commands/actions/episodes/task_specs **逐值一致**。恢复job234.6072s。轮中注入15条未提交transitions、0更新，latest仍round1，全部core/segment依赖与原round1逐值一致；注入job160.4607s。原严格容差未改。具体布尔证据见 `interaction_pilot_preparation/delta-recovery-audit.json`。

末轮core A/B/C=6,828,705/10,063,260/14,410,142bytes；每250数据delta约3.3MB/臂。持久化仍随轮数线性增长，不能声称零成本恢复。原大snapshot方案在数万规模内不适用，新方案经实测后才用于投影。

## 独立评价与探针入口

全新进程、Task/RNG/output；只加载冻结模型，不更新optimizer。完整stop/continue/none工程回合；分叉前BFM输入、实际z、action、qPD、q/qdot逐值核验。当前active窗、分段力/误差、真实q/qPD、末yaw、loaded滑移、接触法向力/部位、瞬态/持续语义及材料点transport均保留。材料点用保存qpos在独立MjData做kinematics，不改变真实轨迹。

硬边界保留失败回合和真实末状态；未知RuntimeError继续报错，不能吞掉正确性错误。新manifest加载先校验文件hash；第一读入口只接受train/dev，不接受sealed。

C真实轨迹另存当前B_R/B_I及因果contact特征，未来0.5/1s质量/active标签独立；三折按父组，stop/continue不得拆开。固定ridge、train-only scaling、当前active分层。空/不足样本测试返回underpowered。仅首读，未宣称§5.2通过，不借未来标签作为Actor输入，也不靠离线probe挑policy。

## 父组与规模结论

生成并哈希锁定34train/16dev/28sealed父组，44/20/36回合。source/partner/common-prefix结构审计无跨split；stop含16–45cm，不继承旧long。真实源配方唯一是必要条件，不足以证明执行状态的统计独立；未来执行时须核实际源SHA、边界与有效组数。封存未执行。

训练第一遍预计21,650 online transitions＋9,730 source transitions。**建议32,000/臂**，约1.48遍。固定相同轮/更新数/训练清单，而不是A只采几百步。上限按实测min throughput×.8、每轮完整存盘×1.25与评价/准备/失败预留反推；详见 [修订锁](PILOT_BUDGET_AND_SPLIT_LOCK.md)。新入口遇`awaiting_Yi_confirmation`在加载模型前拒绝开训。

## 失败和修订（不追溯更改T0）

1. 第一私有队列在B完成前被有意停止排程（已核owned PID），B本身正常结束；静态审查发现材料点logger缺字段，在下一runtime补齐后才评价。
2. CPU恢复摘要脚本被同目录`queue.py`遮蔽stdlib queue；已有任务名防重断言阻止重复启动。修正CPU审计sys.path后成功，保留首次错误。
3. `eval-C-01`实际三回合和probe输出完成，但旧外壳要求返回`status`，汇总KeyError退出1，306.7494s全计账；新增status后独立重跑，不把exit1计通过。
4. 重试排队命令使用不存在的`python`别名，未启动GPU子任务；改用已验证项目venv绝对路径。错误记录不删。
5. 时间网格量化后零需求伙伴运动描述可能重复，生成器过强的全局partner-descriptor唯一断言在写清单前失败。修订为“真实源原点配方＋请求”的伙伴身份，所有共同前缀仍整组。明确相同零运动本身不增加独立性；不是看结果后换条件。

所有原科学阈值/信息权限/训练数学与已有物理结论不变。最新代码中的训练父组metadata只记录来源，不作为模型输入。Pilot仍须Yi再次明确确认。

6. `eval-BC500-02`参考入口在子步重算BC500，读取已更新last-z，与100ms入口命令不一致；严格插值断言使其在首动作前停止（138.5604s）。仅参考入口改为每100ms查询BC500一次，原冻结BFM执行Task给定的五子步z；A/B/C的动作/训练断言不变。参照入口修复不是BC500权重变化。
7. 最后静态审查发现新增父组metadata若用字符串字典会不兼容现有tensor replay stack；修为既有provenance中的int64 spec_index/online_episode，父组SHA沿用原uint8字段。此前新metadata仅出现在评价（无更新），未启动Pilot。最终通过完整A工程重跑检查无训练行为变化。

修订锁拒绝测试：将`awaiting_Yi_confirmation`锁交给Pilot入口，在模型加载和输出目录创建前明确AssertionError；`must-not-start`目录不存在。此为预期防误启动测试，不是失败训练。

## 最终复验、身份与成本

最终执行runtime `6ee2dd283ca532862ca190c3df7b543895470680`。A的完整750样本/6更新重跑202.6360s，三个轮次的agent（模型/Adam/targets）、RNG、5ms物理流、q/qdot与history均与原A逐值一致；replay唯一新增为显式tensor provenance。C评价修复前后完整三回合物理/controls及评价摘要逐值一致。独立C与BC500评价最终均exit0、三回合、一个共同前缀配对检查、无硬边界、无接触外力0；各耗286.8697/290.8578s。这是评价驱动资格，不宣称工程C行为优于BC500。

本次10个子job实际累计2260.4251s；另保守计CPU准备/审计/报告/失败包装300s（非伪称kernel实测），合计**2560.4251s = 0.7112h**。所有失败均包含；等待子进程的父队列不重复计费。新增独占资产339,689,135bytes（约0.316GiB，最终小型源码/报告另不足几MiB），未删除任何历史。全流程实际physics tick **60,170**（含source/reset/恢复/评价），工程采集3515 online transitions/28完整update组；Pilot transitions=0。计入原项目总账后175904.1199s =48.8623h，240h不重置；此次成本亦占拟议8h框架。

每个job的精确SHA/失败码/成本、A/B/C checkpoint core与manifest SHA、物理计数及最终评价资格见 `interaction_pilot_preparation/final-preparation-audit.json`。原BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`、BC500 `a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e` 不变。完整模型/replay/轨迹仅在私有stage，公开提交只有代码、清单和小型汇总。

结论：工程准备完成，修订锁待确认。建议32k/臂在8h/8GiB内做单seed方向性首读；没有新行为Pilot结果，§5.1正式判据尚不可满足，不执行封存、不自动晋级。
