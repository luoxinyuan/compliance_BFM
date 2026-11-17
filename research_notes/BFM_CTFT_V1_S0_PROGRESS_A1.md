# BFM-CTFT v1 S0：A1执行与中间资格记录

2026-11-17。这是中间实测记录，不是S0通过、U/T发车或最终科学结论。Yi本轮追加批准已记录于Issue67评论6940383957。原合同、原单位阻断和旧实验结论保留。

## 已批准并落地的规则

在线支撑和离线滑移分开：每5ms单足实际地面法向载荷合计达到0.05mg即刻有效支撑，不等待0.10s；双足都不足才累计无支撑计时，任一足承重立即清零。严格超过0.05s停止，因此离散实现首次为第11个tick、55ms。计时器随真实状态恢复。

离线每足连续承重至少20个tick、0.10s才成为主滑移评分段；合格段包括首帧，不删前0.10s。材料点追踪、p95和累计滑移门均未放宽。当前完成的是支撑与区间划分资格，不代表材料点滑移全套度量已经通过。

奖励修订按A1：原生aux维持逐步raw单位，新接触项送critic前采用r_contact_integral/dt；物理报告保留时间积分。20ms时相对原字面相加增强50倍，明确是用户批准的单位接口修订，不宣称数学等价。总aux共同冻结尺度仍待真实train-only校准。

## 实证一：在线不误停，明确失支撑会停

机器人质量35.112342kg，实际重力下总重344.45207502N；单足载荷阈值17.222603751N。

|真实存档轨迹|最长连续无有效支撑|新落脚/重新承重次数|因规则误停|
|---|---:|---:|---|
|原生standing|0ms|—|无|
|原生forward_motion|5ms|12，均即刻恢复支撑|无|
|原生left_motion|0ms|2|无|
|原生right_motion|0ms|9|无|
|历史B19均值与采样，两条失败轨迹|各85ms|—|两条均应在55ms停止|
|历史adapter-off对照|0ms|—|无|

这是已有真实5ms轨迹的只读重算，不新增旧B19训练，不重判旧失败。证据为s0-support-audit-01/result.json、quality.py及support_audit.py。另有10/11tick边界与20tick离线区间fixture，fixture不代替真实证据。

## 实证二：原生源与完整恢复

有限锁定所有12个walk整段祖先的首clip0，沿用原checkpoint的tracking_inference及原Actor/PD真实执行，未加载旧高层学习权重。每源执行3s真实站立前缀、1.5s进入、6s原生片段、1.5s退出和4s恢复，失败全保留。

- 完整13s后缀10/12未触硬停止；6/12通过脚本额外的“完整13s且持续移动及支撑交换”筛查。该筛查不是合同要求每个祖先必须完美走停的科学门，不能据此宣称只存在6个合法来源。
- 一个源在退出时出现真实55ms无有效支撑，保留为失败，未放宽安全规则；另一个源的完整退出失败同样保留。
- 每个真实祖先预固定3个较早状态，共36个状态：完整恢复后BFM观测、历史、动作、有效MuJoCo数组、所有全局与自有Generator RNG逐值一致；随后各12次20ms控制、48个5ms物理tick逐值一致。
- 该恢复证据只覆盖上述36个窗口，不证明全部祖先后继可恢复，也不证明尚未实现的训练checkpoint已合格。合法时段、转变标注、train/qualification/eval血缘隔离仍未锁定。

主证据：s0-native-sources-03/progress.json及source-NN.pkl.gz；s0-source-restore-01/result-restore.json。完整私有状态不进入公开Issue。

## 实证三：原始模型迁移和权限

runtime提交8a820c6，s0-model-qualify-01/result.json：

- 原始checkpoint SHA256为33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
- 接触新增输入列零初始化，保留原输入LayerNorm域；batch1及batch4的Actor/F输出与同精度原模型逐值一致，最大差0。
- 修改隐藏伙伴字段不改变Actor输出。新aux与target_aux fresh且逐值相同；B/target-B及原robot normalizer冻结。
- 五个fresh Adam覆盖Actor全部网络、F、CPR critic、aux critic、discriminator；无空B optimizer，无旧selector/U/FiLM/adapter权重。
- 测得CUDA peak reserved 10,248,781,824bytes。该job主程序核心计时23.925453s，不含顶层Torch导入；未逐项测量的导入记入保守准备reserve。没有物理采样或agent.update，不能充当训练吞吐。

## 工程失败与等价回归

1. clip0名称与实际MotionLib整段键不同：失败原样保留；对12源的fps及原始字段逐值核实clip0等于整段首片段，再做键映射。没有换候选。
2. 编码原生参考时改变MotionLib当前选择，导致下一reset元数据不匹配：修复为编码后恢复原库选择，并保存自有Generator RNG；修复前后首源1655个真实tick的身体、力和支撑逐值一致，随后36状态恢复通过。
3. 一个脚本将全13s走停成功筛查写成DATA_SUPPORT_BLOCKED：原结果不覆盖；本报告区分该脚本统计与合同的合法源要求。不据此直接放行源池。

## 成本、当前阻断与下一步

前述完成job核心计时共817.720394s，含失败、只读重算、恢复和模型资格；新增实际5ms ticks共44,533。加此前ledger976.906147s为1794.626541s。另预留3600s未逐项测量的CPU准备/工程/IO成本，保守阶段快照5394.626541s已写回远端；这是保守计费，不是声称GPU训练运行了该时长。后续更新试车成本另计，不能将此快照当最终总账。正式U/T训练样本和联合agent更新均为0，N尚未锁定。

模型资格进程已正常结束后，授权服务器1017端口出现connection refused；另起无复用连接再次被拒绝，后一次连接超时。随后原服务器恢复，GPU0 UUID不变且空闲；未擅自换服务器、GPU或后端。

## 后续完整更新试车（仍是S0，非正式训练）

先用本stage新采集的两个原生g0来源、1200条真实连续控制后继检查完整1024 batch更新接线。它不满足含真实接触/release的全链资格，不用于锁N，也不向正式U/T传递权重/replay。aux尺度1明确为一次性占位，不是正式校准结果。

首次试车s0-native-joint-01在首个完整更新内失败，0组agent.update完成：AOT编译将判别器expert/train多个输出共享反向tape，先求一个分量梯度后释放tape，下一分量反向报second-backward。原损失三分量均保留。核心失败计时133.092202s另计，原日志/failure/cost均不覆盖。

独立工程修复runtime460aba9只让tape存活到最后一个有序梯度分量，不改变loss/GP/系数、优化器、累加顺序或compile模式。s0-native-joint-02的原eager与修复eager对照：逐梯度、参数和Adam全部逐值一致，expert/train/GP三项保留。用于等价对照的两次discriminator-only step已记录并恢复初值，不冒充完整agent.update或正式训练。

但编译重试失败：PyTorch AOT donated buffers要求retain_graph=False，而当前多分量有序反向需保留共享tape。异常明确指出可用donated_buffer=False禁止该缓冲优化；本轮没有擅自改变此编译设置。第二次试车核心53.179073s；独立外层时间记录96s，包含顶层导入。两次均0组完整agent.update完成。

状态为BLOCKED_COMPILED_ORDERED_GRADIENT_LIFECYCLE，GPU作业已结束。依据正文§9同根因簇一次修复、复发停报的约束暂停，未进行第三次同类修补，没有把eager等价通过写成编译或全链通过。A1支撑与单位资格不受此实现失败否定，正式U/T不发车。

建议下一项最小修订仅为：授权一次AOT donated-buffer生命周期兼容验证，保持compile=True、原数学与原FP32顺序，显式禁用该内存捐赠优化；用同一实际mini-batch重新检查eager等价、编译完整更新、显存护栏和完整恢复。若仍失败，报告，不扩成新算法/配方/扫描。此建议尚未执行，不把报错建议视为验证成功，也不自动选择eager代替原编译设置。

恢复连接后继续原S0：合法源/血缘与转变标注、材料点滑移量尺、真实接触字段和固定aux校准、1024真实行与完整联合更新、完整训练状态新进程恢复和故障注入、多worker端到端实测，最后按合同机械锁共同N。任何未完成项均不记PASS。
