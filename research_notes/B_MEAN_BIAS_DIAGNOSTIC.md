# 包三：原 B 表征公共均值方向的离线诊断

使用原冻结 checkpoint `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`，
不是 FB 接触扩展后的模型。原 normalizer、B、project_z 和 reward_inference 均未修改。

## 数据与定义

复用现有 26.7GB replay，完整 SHA
`46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b` 核对通过；
无下载、无重写。原 sampler、seed5803 抽 30000 successor 行（29898 个不同 replay 索引），
保留原连续采样结构，不称为 30000 独立回合。
已审计的历史 qpos/qvel successor 合同沿用 `reward_stage_contract.py`。

七个固定、仅机器人状态的非负 reward：正/反向 .15m/s、静止、root 高度 .65m、
右肩抬升、腰 yaw ±.26rad。velocity sigma .12、侧速 .10、yaw-rate .25；
其他公式见 [b_mean_audit.py](b_mean_audit.py)，不是本项目接触任务 reward 的改动。
五个 goal：真实原生 standing 状态及其私人目标观测中的腰±.26rad、肩 pitch−.6rad、
降低高度/屈膝。goal 只推断，不在包三执行，不把目标姿态当实际动作。

七个原始 reward z 均实际调用未改动的 `model.reward_inference`。
另保存 `e=ΣrB` 双精度聚合用于分析，与原 API 投影后的最大差 `4.768e-7`，不修改任何容差。
Goal 用原 B(goal) 和原 norm16 投影。

```text
μ = mean B(s)                  （同一 replay successor 分布）
原 reward: z = project(Σ r B)
中心化诊断: zc = project(Σ r (B−μ))
原 goal: project(B(goal))
中心化 goal: project(B(goal)−μ)
```

**中心化 z 从未用于物理控制、训练、正式候选或生产推断。**
包二的 reward 额外探针只用原始未中心化 API 的 z。

## 量化结果

`||μ||=1.00134`，单个 B 输出平均 norm `16.00000`，比值 `6.258%`。
两半样本各自估计的均值方向夹角 `9.914°`，说明该估计仍依赖有限分布。

|prompt 类别|原两两角度 min / median / max|中心化后 min / median / max|
|---|---|---|
|7个 reward，21对|29.89 / 74.90 / 88.55°|35.11 / 97.02 / 120.39°|
|5个 goal，10对|17.88 / 40.27 / 74.40°|17.91 / 40.33 / 74.51°|

在 reward z 中，μ 方向的平方投影为 **14.84%–61.68%**；
`||Σr μ|| / ||ΣrB||` 为 **37.10%–77.75%**，中心化令单个 z 转动 **21.78°–51.03°**。
Goal 对应平方投影只有 **0.191%–0.525%**，单个方向转动约 **3.59°**，
两两分化程度基本不变。逐 prompt 数字及完整角度矩阵：
[b-mean-summary.json](fb_native_bias_artifacts/b-mean-summary.json)。

平方投影是几何量，不是“性能失败由均值导致的百分比”；公共项范数比也不是可相加的因果贡献。
小的 B 均值可以在大量加权平均后变得显著，不等于所有单个 B 都贴在同一方向。

## 有限判读

**有可测量的公共均值方向，且这些 reward 的聚合受其影响明显；没有发现所有提示都坍缩为同一 z。**
本组 reward 原夹角中位已约75°，不能再说“不同 reward 几乎生成同一个候选”。
中心化增加了角度分化，不等于提升运动/接触质量；原 B 中公共成分也可能承载有用信息。
本轮未做中心化物理 A/B，未解释 replay 时序、多模态平均、进入状态或反馈学习的全部失败原因。

实际 runtime `aa62fe541037bb6a59f453fe4b9c366e10977523`；
受监督job墙钟 **147.97s**，其中study计算27.47s，导入/初始化/IO同样计入。
7次官方 reward 调用及缓存 B/goal 共240005个 B-state evaluation（含重复计算）。
完整 tensors、索引、目标与 z 保留私有目录；公开仅汇总。
所有模型与 normalizer 冻结核查通过，正式训练0。
