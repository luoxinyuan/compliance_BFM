# Issue62 B释放后越界：均值控制真实失效，按授权边界停止

2026-11-12。Yi授权在共同合同内自行修复并续B/C/dev；若均值动作本身越界且需改配方则停止。**本次命中该停止条件**，没有改std/reward/硬界/划分，没有跳失败父组、重抽RNG、续训或启动C/dev。A128及B19完整模型保留；这不是预算耗尽。

## 同状态真实干预

B19=4,750真实50Hz transitions/38更新；原第20轮另26条未提交数据保留。父组`8dde57eac4325d7701fafe97d48f57cb962cd8ff121a9e4770a1bbdffa10db4e`，standing真实源4.4s、25.984cm mid、onset5.1s、ramp4.2s、release10.8s、end14.8s。恢复点11.8s，在释放后1s；两臂保留同一完整physical/solver/历史/last-z/force/RNG及冻结模型。

|分支|实际停止时间|左踝pitch实际q rad|越界 rad|实际力|控制/物理步|
|---|---:|---:|---:|---:|---:|
|原采样std .01/clip .06精确复现|12.315s|−0.89405361|0.02138361|0N|26/103|
|只去掉采样噪声，均值真实执行|12.310s|−0.90022882|0.02755882|0N|26/102|

当前实际模型`left_ankle_pitch_joint`范围[−0.87267,0.5236]，qpos地址11、qvel地址10；原真实q额外0.02rad保护线不变。两分支各在约0.51s后撞线，没有等3s强行越界继续。目标q_PD和实际q严格分开。

sample分支全部高层commands、实际actions、controls及103个5ms physics记录与原失败逐值一致。额外诊断前向未改变原采样轨迹。两分支冻结agent/Adam/targets摘要与恢复RNG摘要相同，更新数始终38。失败不是restore错配或本次日志前向改了控制。

## 同一真实状态的动作分解

原采样访问状态12.300s，左踝pitch：

|层|normalized action|q_PD rad|
|---|---:|---:|
|冻结BFM，在该状态/实际z上|−0.27843824|−0.81058341|
|adapter增量|−0.15926778|约−0.34925622增量|
|完整策略均值|−0.43770602|−1.15983963|
|实际采样|−0.43881938|−1.16228104|

此帧BFM自身目标在范围内，adapter将均值推到下限之外约0.28717rad；随机噪声只额外约−0.002441rad。整个sample续跑26个控制状态中14个均值目标在机械范围外，最小−1.586633rad。均值真实分支26个状态中13个目标范围外，最小−1.452688rad。

但**不能据此把整个失败唯一归因于adapter某一帧**：均值分支随后访问不同状态，12.300s的BFM单独目标也已达−0.934970rad、完整均值为−1.286558rad。没有做adapter-off物理rollout，不能声称删除adapter就一定恢复。作用层分解是同状态函数证据，物理mean干预才证明去噪声仍失败。

## 合同核查与为何不能当普通工程修复续训

`INTERACTION_FB_METHOD_SPEC.md`明确adapter在normalized29D action空间加一次、最终clip[-1,1]；`agent.py::action`按此执行。原映射scale基于PD effort/P增益，统一[-1,1]不是各关节的机械q_PD限位。本次同状态action→q_PD独立复算与实际控制记录一致（浮点乘法路径最大误差仅检查≤1e-6rad；原物理/恢复逐值比较未放宽）。模型中没有已批准却漏执行的逐关节target限位合同。

**已证实**：B在其真实释放后访问状态上的完整均值控制也触发硬界；原sample可精确重现；均值目标的大偏离远大于该帧采样噪声；当前实现执行了既定normalized-action合同。
**相容**：自由29D执行adapter可突破原BFM动作先验，原生source域上的CPR保持未覆盖该释放恢复域，闭环状态传播进一步放大了风险。
**未证**：具体哪个loss/梯度首先驱动偏离；adapter独自是唯一原因；原初始化完整mean回合也失败；C必然失败；A128具备行为收益。均值干预起于B自己的真实状态，不是从episode初始状态跑完整均值策略。

若从B19按同RNG原合同直接续训，会确定性重现该硬界。裁剪q_PD、约束adapter、加强/扩展保持损失、改样本或RNG均不是恢复文件修复；至少会改变动作或训练配方。因此按Yi边界停止，未自动把这些选项接入，也没有用普通reset掩盖越界。

下一项最小、可判别建议：**先在该释放窗口做同状态adapter-off短程物理对照**，验证是否恢复稳定，再决定是否需要adapter的物理输出约束或保持域修订。这个建议尚未执行，不直接承诺硬clip就有效；既有P0-B动作限幅损害质量的负例仍需保留。当前不继续扫噪声或更换reward。

## 当前交付范围

A已完成128轮32,000 transitions/256更新；B完成19轮4,750/38，另26条未更新失败采样保留。C与统一dev未执行，缺格原因是明确的均值物理失败，不写成已比较。用户要求的三臂完整交付被其预声明停止边界阻断，不以A训练完成宣称研究完成。

## 身份、成本和证据位置

实际诊断runtime `9bceb9956baa0d29d9c6d6d222b9e3484ef3707c`；原三臂runtime `62d55b0d26127440253dba2b5d0baa8bd1e457fa`。
B19 core SHA `0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984`；manifest SHA `150f107489e267fc0c93ef8e99b7a51d798cd5a1011f9c3c3ddba9017855306a`。
A128 core SHA `d97f8c515fb76f34b7ff3d75b7a29208a68987afbf75de5cf727f8227180114b`。完整身份、trace/model/RNG摘要、逐值比较及计数见[小型JSON](interaction_pilot_b_failure_summary.json)。

独立stage `interaction-pilot-b-diagnosis-20261112`，`runs/{sample-B19,mean-B19}/{probe-trace.pt,probe.json,same-state-actions.json,physics-counter.json}`、`sample-exact-reproduction.json`、`b-failure-analysis.json`、`validation-ledger.json`。旧`pilot-B-v3/uncommitted-stop.pt`和所有checkpoint不覆盖。统计脚本`research/interaction_fb/analyze_b_failure.py`，协议[链接](INTERACTION_PILOT_B_DIAGNOSTIC_PROTOCOL.md)。

两个GPUjob共425.747601s，含初始化/完整重建，实际物理步共9567；另保守计准备/CPU/发布1200s。Pilot累计 **31066.150841s＝8.629486/15h**，聚合输出约3.021/8GiB，项目240h不重置。新增正式训练/更新0，GPU0已释放；共享自由约399GiB。未读sealed，未修改main/共享BFM/PD或他人任务。
