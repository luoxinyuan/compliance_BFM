# Issue51 — 方法、权限与复现入口

[结果](NATIVE_TRANSFER_RESULTS.md) · [预登记](native_transfer_protocol.json) · [Issue51](https://github.com/GaoE05/robot-research-bridge/issues/51)。本阶段是冻结物理搜索Teacher的有限覆盖/摊销实验，不是PPO，不声称完整FIBER复现。

## 实际Teacher和标签

同权重原生standing、goal伸臂和motion来源，固定7个提议、2.4s真实MuJoCo预测；每100ms仅实际执行第一个合法latent命令，再读当前物理状态重新搜索。只使用当前伙伴位置/速度和明确恒速短期预测，不读取真实未来停止/最终距离。候选kind/phase和搜索计划作为Teacher自身控制信息保存，不作为部署输入。

冻结BFM SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`、原PD、K80D8/总cap10N、10Hz高层/50Hz BFM/200Hz物理不变。原生reward-z没有成为主要有效候选，不能宣称reward prompting已学会接触任务。I/C0/C32仅冻结评估参考，没有继承其actor/隐特征。

实际执行输入、target z、上一真实z、本体/BFM历史、solver状态与force history精确保存；预测分支/未来未执行节点与实际回合分开。642查询，只有641个完整100ms命令作有效样本；新.22回合最后80ms完整保留并重放，但不当100ms标签。6回合重放P/S输入、qpos/qvel、上一z和q_PD最大误差均0，norm16重构舍入最大2.384e−7；概率/物理阈值未放宽。

## 权限与训练

P1088D为当前特权反馈，不含未来。S1639D = 465原BFM actor本体/动作历史 +256真实上一z +900力历史 +18已知接触安装。原始BFM观测dict虽保存time/privileged_state metadata，S只拼state、last_action、history_actor，不拼time或privileged_state。S不含anchor/error、伙伴真值速度、目标yaw、active真值、taskID、未来、搜索kind/phase。真实仿真外力输入应称ideal-force，G1力传感配置未知。

P和S均fresh128–128 Tanh Direct actor，输出原256D合法norm16 z；相同seed5101、train-only归一化std floor.05、Adam3e−4、batch128、episode-uniform同批次顺序、4000更新。P188,928与S259,456训练参数，输入维不同，非严格同计算成本。标签同权重合法z，采用简单z MSE，无近恒等逐状态投影优化。

train=旧Issue50停止/继续整组223＋新8cm88=311；dev=.22m/s104完整；final=新66cm继续118/停止108整组。共641，不随机切相邻帧/共同前缀。按0/1000/2000/4000的开发物理分数 `force_RMS + .03*error_cm + cap_fraction + 20*boundary` 选模，P4000、S0（新初始化）；终检不用于重选。另固定S4000已见诊断及预登记单轨P/S各1000更新，仅训练条件检验，未推广为正式模型。

同状态5子步动作检查保持真实上一z、真实BFM观测、slerp时序及q_PD映射。固定观测重放不是5步可微物理rollout。诊断JSON字段qpd_max_deg是5个子步中最大的关节RMSE，不能解释为单关节最大误差；正式实际轨迹的qpd_peak另有joint/time/20ms定义。

## 源码/runtime与私有产物

服务器项目根 `/mnt/data/temp_user/yi.gao/projects/compliance_BFM`；独立stage `experiments/native-transfer-20261107`。运行目录均保留，不修改旧Issue50。

|环节|实际runtime commit|stage内位置|
|---|---|---|
|Teacher覆盖|443f02c4fc22fd84e59f2cee4e25750d838fe527|repo；runs/teacher-coverage-01|
|重放/主BC/评价|0b5135574719f41d888f9f77fcf2d32e951868ca|bc3-repo；runs/data-prepare-01、bc-fit-01、bc-eval-01|
|初版分析/视频|871df51e37653bfb6bd161d3e449efbf471e6ed8|review-repo；runs/review-01、video-01|
|冻结主S与单轨诊断|d800eb0b90d00526053cc983b33abb39ba6e5d26|diagnostic-repo；runs/diagnostic-01|
|最终分析/实际视频|73c437e81f2aca617bc432ddf859dddd2a959b4e|notes-repo；runs/final-analysis-01、video-final-01|

P/S checkpoint：stage/runs/bc-fit-01/P-4000.pt、S-4000.pt、S-0.pt；SHA见结果/evidence。单轨P `3819d9b463a724e51d99ca1586d5673809906b0cb28d0b64f9cba5afe2a59ef0`、S `71928776678f6dc01bee3a6dc657863407630e344e5105d7093a48e82e2b45f5`，仅诊断。数据dataset.pt的SHA、6个源回合SHA和模型各候选身份均在私有manifest/selection/checks和公开精简evidence保留。权重/原始qpos/数据/视频不进Git。

## 可复现说明（勿无意重启已完成任务）

沿用项目`.venv/bin/python`，Python3.10.21/Torch2.7.1cu128/MuJoCo3.3.3，GPU UUID `GPU-3654846e-2666-fe93-20e9-cd7d08a217db`。CUDA9.0/132SM/~80GiB验证，NVML名称L20Z冲突保留，不以名称猜型号。启动前重新检查空闲权限/CPU/RAM/I/O，本阶段单GPU顺序调度；CPU仅用于保存轨迹统计，不替代BFM推理或物理实验。

`native_transfer.py --mode prepare|fit|eval --output <new-isolated-dir>` 经既有m10_physics包装执行；真实运行由 `rolling_job.py <name> <script> <max_seconds> <script-args>` 约束，不能直接覆盖旧output。源码固定后再启动，运行前读取协议/案例文件及对应ledger。

正式来源/评估在 `native_transfer_cases.json`、`native_transfer_protocol.json`；过程ledger含PID/start_ticks/boot/runtime/elapsed和失败；新最终结果见native_transfer_evidence.json。初版排队脚本IPC及80ms尾命令修正发生在BC启动前，未打断Teacher、未改物理、未重复采集，证据文件单独保留。

视频 `native_transfer_final_render.py` 只读已保存真实qpos，不重跑物理、不挑seed；共3条25fps1×，不同长度格子结束后明确保持末帧。3–4s真实前缀/全部接触和恢复保留。新S-init正式选中但视频用S4000标明非选中失败；历史S0身份完全独立。
