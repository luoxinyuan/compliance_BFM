# Issue56 FB接触扩展：用户授权提前停止记录

记录时间：2026-11-10 12:30 CST。状态：**训练已停止，完整阶段归因尚未结项**。
本记录说明为何停止及保存边界，不是完整1M终检，不否定整个FB路线。

## 已完成与已停止

- 冻结BFM高层KL .02/.08均完成128批，各98304真实高层决策。
  两臂17个开发节点中只有初始化满足预声明保护，均选iteration0，
  且各自选中初始化的独立封存评价已完成。不能计为新增学习成功。
- 接触扩展FB正式启动，最后完成并落盘的block为33792真实50Hz transitions、
  512次完整agent update。下一采样block在停止时未完成，其增量无法从存档
  精确确定，不伪称总采样数恰好33792；实际job-wall包含该部分。
- 未达到250k或1M，未产生相应模型/评价。后续FB自动评价已撤下，
  没有Student训练或下一阶段自动启动。
- 用户明确授权：若持续无效，可提前终止并完整记录。此授权取代原先
  “不因早期任务质量不达标自动停掉1M”的执行要求；物理/概率/质量合同未改。

## 停止依据：有效采样窗口持续萎缩

归档244个已记录回合，198个触发physics_review，46个正常结束。
这是物理质量检查提前终止，不等同198次摔倒或数值错误。

|回合索引（从0计）|回合数|物理质量提前终止|其中接触前终止|实际50Hz控制步数中位数|
|---|---:|---:|---:|---:|
|0–29|30|6|1|483|
|30–59|30|12|6|459.5|
|60–89|30|27|8|127|
|90–119|30|29|16|35|
|120–149|30|30|25|6|
|150–179|30|30|29|4|
|180–209|30|30|30|3|
|210–239|30|30|24|4|
|240–243|4|4|4|4.5|

前段尚能采集较长真实任务，后段大量回合在原生源接管后、
实际接触前几个控制步就终止，已难以继续学习持续交互。
最近示例含joint_limit_excess约.020–.035rad，但height/tilt尚正常、
failed=false；不能直接称作失去平衡。质量边界触发机制与真实动作
风险的对应关系尚需只读归因，不能在本轮为通过而放宽阈值。

已观察到完整FB agent update实际运行，编译热身后16更新约5.1s。
这只证明执行/更新回路工作，不是物理能力收益。累计样本太少且已中止，
不据此宣称FB方法不能学会或1M必然失败。

## 保存缺口：最新FB训练内存没有形成checkpoint

固定入口仅在step0/250k/1M调用save_package，缺少早停/异常落盘。
用户授权停止后，先终止本任务队列，再对已核实worker尝试SIGINT；
其SIGINT被忽略，随后SIGTERM退出，未注入调试器、修改运行源码或重训复造。
当前FB仅有完整step0包；512更新后的模型、六优化器和replay内存未保存，
不能恢复这一个中途训练状态。对此按独立[存盘勘误](DUAL_TRACK_FB_CHECKPOINT_ERRATUM.md)
记录，不能把step0冒称末轮。已保存全部现有日志、失败物理轨迹和更新指标；
更新日志不是可恢复replay。

## 最新可用模型与运行身份

|产物|SHA256|身份|
|---|---|---|
|KL .02 candidate128|e4cd0f4e361f4cfc7dfffe74c6205145ee6fd20ee0492815e74ab17c23bbbe5e|完整末轮研究模型，未晋级|
|KL .08 candidate128|edb41ba25c078910d9150c731ccdda0e949a85ca29636f0986dc1a2035ad4afd|完整末轮研究模型，未晋级|
|FB step0 model|ab4d9b712d6ffef1e9415823e8ab54c12b0e8882cc1aa55e687c6d52deb97466|独立扩展初始化，非训练末轮|
|历史BC500|a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e|冻结基线不变|
|原BFM|33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361|共享原权重不变|

运行源码始终为147b22b684b0c03a613559fa6d5e5881d380ab7f。
公开文档commit与运行commit分开。所有旧检查点/失败/历史报告保留，
未复制大权重到Git，不覆盖main或共享资产。

## 私有证据与成本

stage：`experiments/dual-track-formal-20261109/`。
- `runs/fb-1m/{learning.json,episodes.json,source-episodes.json,stdout.log,failures/}`
- `runs/fb-1m/step0/`，以及两臂`candidate-128.pt`及其原有历史检查点
- `early-stop-20261110/{request.json,assessment.json,signal-followup.json}`
- `raw-supervisor-job.json`、`raw-wrapper-ERRATUM.json`、
  `raw-result-before-stop-finalization.json`、停止前日志与账本快照

原wrapper将SIGTERM记为failed/returncode−15并生成通用ERRATUM。
原字节已保留，最终侧车补充stopped_by_user与原因，不能混称实现报错。
supervisor实际FB job2605.98s（约43.43分钟），包括初始化、采集/更新、
失败/reset和等待。全部已登记约16.034累计job小时：
冻结45110.10s、FB2605.98s、共同10006.28s（含原carry/准备预留）。
训练队列380408、FB worker499290、supervisor499277及其已核实子进程
均已退出；核对GPU0显存0MiB、使用率0%。只操作本任务进程。

## 尚需完成的结项分析

用户指定的[batch48分阶段/状态覆盖归因](DUAL_TRACK_BATCH48_ATTRIBUTION.md)
尚未完成。已有摘要支持阶段指标复算；需要特别注明开发评价默认只保存
汇总，不保存全部逐帧输入，因此状态重叠若无法从既有资产精确确定，
必须标为缺证，不能从训练rollout替代推断成开发状态事实。
本记录不宣布整个Issue56结项，不自动重启FB/PPO、补训或扩展Student。
