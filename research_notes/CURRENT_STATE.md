# CURRENT_STATE — Issue51 Teacher覆盖与可蒸馏性验证

最后核对2026-11-07。[阶段Issue51](https://github.com/GaoE05/robot-research-bridge/issues/51) · [预登记协议](native_transfer_protocol.json)。Issue50已结项；旧完整结果见NATIVE_ROLLING_RESULTS/METHODS，不恢复PPO或C0。

## 授权、状态与预算

用户新授权：固定7提议/2.4s因果滚动Teacher，验证4条新完整回合，分离真实首动作数据，比较fresh特权P与受限ideal-force S的BC及真实闭环。Teacher覆盖已启动，尚无新条件成功结论；BC重放/训练/评价代码准备中。仅预定任务在本阶段内顺序执行，不将计划当结果。

独立stage experiments/native-transfer-20261107；运行repo固定提交后启动，Issue50目录只读。H100级NVIDIA当前UUID GPU-3654846e-2666-fe93-20e9-cd7d08a217db，NVML L20Z名称冲突保留；14:44 UTC核对无GPU任务、20CPU、可用RAM199GiB，沿用Python3.10/Torch2.7.1cu128/MuJoCo3.3.3。发车前再次实时核查。

硬上限18000累计job秒（含CPU300预留）、10GiB、480搜索查询/1612800候选物理步、9600实际Teacher物理步；P/S各4000监督更新，必要单轨诊断各1000，总不超10000。无PPO、无Student访问状态无限聚合、无未来oracle、无新reward扫描/资产下载。

固定Teacher新条件：8cm/ramp1.6/onset5.2；72cm/平均.22/onset5.2；66cm/ramp4.4/onset5.4的继续/8.1刹停配对，减速.7、保持2、释放恢复4秒。冻结BFM/PD、K80D8/10N、256D/10Hz/50Hz/200Hz、原reward与候选不变。

## 数据与选择合同

train为Issue50停止/继续整组＋新短请求；dev为新.22速度整回合；final为新66cm共同前缀两支整体。相邻帧不随机拆分，不将共同前缀跨train/final，不按final调参或选模。仅真正执行的首100ms命令作标签；预测/未执行节点/搜索评分单独保留。非法/严重失配全cap段标无效，保留原轨迹，不能假装正确恢复。

P1088D/S1639D：同小128-128 Tanh Direct actor、fresh初始化与相同随机批次，train-only归一化，4000步。0/1000/2000/4000 checkpoint按预定dev物理分数选择；参数量和成本差别独立记录。当前Teacher不是新神经网络；当前P/S也未训练。

S不含anchor/伙伴真值速度/目标朝向/未来/任务ID/active真值/Teacher内部phase或kind；保留原本体、实际执行z、已知安装和理想力历史。理想仿真力不等于G1传感已解决。未来可见Teacher只为另行上界方向，本阶段不做。

## 不可丢失的事实与基线

Issue50真实MPC .15牵引7.417cm/5.465N（I10.265/6.682），同前缀停止7.605/5.457，未训练新PPO。6个reward-z未成可靠低力walking，未在223完整滚动决策选中。停止偏航约20度、q_PD峰56.91度、严重超前仍10N、约23秒计算/100ms控制仍是边界，不因新阶段抹去。旧有限域Teacher→ideal-force Student成功、I/C0/T30/S0/D2与失败全部保留。

BFM SHA33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361；I72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a；C0ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c；旧Issue50物理runtime32b3502262399409dd9091b067c89ba619d20873；文档维护9c2f383不是新物理运行。

长期gaoe05/research；Teacher实际runtime443f02c4fc22fd84e59f2cee4e25750d838fe527；监督进程PID23033/child23035、boot be0c7b44-26b0-4699-8716-9903210007ab，实际身份以ledger/start_ticks为准。BC另在bc-repo固定提交后运行；模型SHA/成本待产出，不用文件名代替。不改main、共享BFM/PD/权重/SoftSONIC/他人任务，不做真机。新会话先读本页/OVERVIEW/协议/Issue51和实时ledger，不以旧PID/空闲快照重启。
