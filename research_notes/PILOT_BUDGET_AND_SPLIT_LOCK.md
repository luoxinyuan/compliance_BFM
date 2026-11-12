# Issue62 Pilot lock v2 — 等待 Yi 确认，未开训

2026-11-12。用户否决每臂3,000规模。原草案保存在 `PILOT_BUDGET_AND_SPLIT_LOCK_V1_REJECTED.md`，T0三门结论不变。本版本完成 A/B/C 实际路径、独立评价、增量完整存盘恢复资格及父组结构审计后提出。工程试车不计作Pilot学习成果。机器为单RTX PRO5000 GPU0，可靠num_envs=1。

## 请求批准的规模与证据范围

**建议每臂32,000真实50Hz transitions，A/B/C合计96,000；单seed6201。** 每轮250 transitions＋2次完整agent update、batch32；128个不可覆盖完整轮/臂，256次完整更新组/臂。不改T0优化配方来制造速度。低层实际探索仍为std=.05 Gaussian、noise clip=.3、执行[-1,1]；不是高层PPO/std=.008。网络与loss沿已审T0实现，容量差异在准备报告单列。

首读仅为 **dev方向性证据＋B_I探针首读**。不执行sealed、不以单seed宣称显著性或收敛；§5.1正式判据尚不可满足（组合消融、匹配容量、可靠探针不确定度等仍缺），不自动晋级或扩训。物理绝对线、原相对门及尾部照旧报告，不能被平均J抵消。固定末轮与初始化比较，不依开发成绩反复挑选模型。

## 父组与实际覆盖

train=30主父组＋4控制=34父组/44回合；dev=12＋4=16父组/20回合；sealed=24＋4=28父组/36回合。近/中/停止主父组均分；每split零需求/无接触各2组。near4–12cm，mid/stop16–45cm；stop/continue共用真实源和伙伴前缀。所有接触物理不变。

源由真实原生standing过程中的小幅native arm5脉冲产生（幅值.02–.08，持续.3–.7s），接管4.1–4.8s；不是qpos摆姿或仅换seed。零需求运动脚本天然可重复，伙伴身份包括真实源原点配方；**不把运动描述哈希不同当物理独立性**。生成器审计source/partner/common-prefix整组不跨split。实际执行后另记录源状态SHA、实际完整回合和各阶段覆盖，若发生相同真实源/边界失败，不虚报独立组或用新组替换。

所有回合时刻在生成前按100ms网格量化，保证完整50Hz/五子步合同；不根据行为结果改动。训练固定清单顺序循环，三臂相同；一次完整44回合需约21,650 online transitions，另有9,730 source transitions（后者计成本，不当训练样本）。32k约1.48遍，保证无硬边界时每个训练父组至少完整经历一次。重复遍历不增加独立父组数。这仍是稀疏首读，不承诺已充分学习。

具体范围/清单见版本化生成器和 `interaction_pilot_manifest_v2/audit.json`；文件SHA见下表。封存不做物理执行、模型选择或探针训练。

| split | parents / episodes | file SHA256 |
|---|---:|---|
| train | 34 / 44 | `ef9a6e9be602f613fecb30bd0941543aed0b4e4d366e2498f7579e55e1eebfb4` |
| dev | 16 / 20 | `d418de94370846562179c4bed5f6c39a8e6a0e7195e0e040c35f8bf9dec224e5` |
| sealed | 28 / 36 | `3bc707488465a639fed3c8568e7f440c09909c361925a45156e2dfef6a41c69f` |

## 吞吐与反推预算

实测每臂750 transitions/6完整updates，包含reset、采样、更新和事务IO的warm throughput：A9.565、B9.549、C7.869 transitions/s；完整冷进程分别3.528/3.666/3.370。短试车的cold率不能外推长程，也不能只用warm不计启动。工程报告保留每轮与失败成本。

保守训练投影用最慢实测warm×0.8 = **6.295 transitions/s**，三臂96k约15,249s，另预留三次cold各140s。实际共享IO可能更慢，不能保证用满32k。

预算分账（均为累计job，不按并行折半）：准备/失败/报告预留3,000s；正式采样更新存盘15,249s；cold420s；100个完整开发回合（A/B/C末轮60＋共享相同初始化20＋BC50020）预留7,000s；CPU分组probe/汇总300s；余额2,831s用于已计账失败/恢复/超预期IO。总≤28,800s，全部准备成本也扣除，不自动借240h余额。三臂重GPU串行。

存盘投影：末轮实测core合计31,302,107bytes，三臂每250新增delta合计约9,904,583bytes；取**41.6MB/三臂轮**再加25%增长，128轮约6.656GB。预留1.5GiB给准备、初始core、全部独立评价、probe、错误/中断包及增长，合计约7.70GiB。8GiB硬顶及共享free≥100GiB均保留。

在上述保守预留下，另保留至少1,200s故障余量时，时间允许约35,250/臂，存储允许约**33,500/臂**（250整数轮向下）；这是当前实测投影的最大值，不是保证。选择32,000留出余量，不把理论最大值当必须耗尽目标。规模不因首读好坏追加。预算不足以完成全部时保留已完成轮及差异，报告不等额/不足证据，不删旧包找补。

## IO与恢复

每完成轮保存完整mutable core（模型/targets/Adam/normalizers/prompt/全部RNG/真实Task与solver/控制历史）＋本轮新增replay/native/5ms轨迹delta。lossless zlib；每段不可变、有SHA/长度/计数；manifest引用全部前段，读回验证后fsync与原子latest提交。每轮均可完整恢复，不只存权重，不删除旧轮。恢复到新目录使用不可变依赖hardlink；不能仅搬走latest而遗漏delta。两轮新进程继续的agent/replay/RNG/物理等逐值一致已验证。轮中15条未提交数据单独保留，latest仍为round1，不冒充已提交。

按完整core+新增delta增长实时监测；下一轮前需保证最坏一轮和停止包空间，以及评价保留额度。预算或100GiB护栏将触顶时正常保存并停止；禁止降低护栏、压缩掉字段或覆盖历史。

## 评价与B_I首读

独立进程、独立Task/RNG；没有训练中同步评价消费训练随机流。评价原始checkpointSHA、真实controls/physics、各阶段XYZ力/误差/支撑/q/qPD/yaw、滑移瞬态/持续/材料点路径和失败。stop/continue分叉前输入/action/qPD/身体状态要求逐值一致。

C只从真实开发回合保存当前B_R、B_I与因果contact特征，未来.5/1s标签单独保存。固定按父组hash三折，整个stop/continue父组同折；trainfold-only标准化，固定ridge .01。R/I/RI/current-contact等长288槽位但有效秩不同，明确不是容量因果证明；有效训练/测试父组或标签类不足则probe_underpowered。当前active分层，无taskID/clock/未来输入。此首读不完成§5.2可靠性及控制相关性资格。

## 启动闸

`interaction_pilot_lock_v2.json` 状态仍为 `awaiting_Yi_confirmation`；入口在加载模型前拒绝此状态。只有Yi明确批准修订规模后，另存带批准引用的不可变执行锁，匹配train哈希再启动。此次提交不是开训记录。
