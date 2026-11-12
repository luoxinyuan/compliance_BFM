# Issue62：B19 adapter-off 真实物理反事实

2026-11-12。Yi授权“做adapter-off对照，结果出来再定配方”。**两条off分支均完成3秒，实际关节越界量0；on采样/均值约0.51秒均撞原硬界。** 这支持执行adapter对该释放状态越界有因果贡献；不证明adapter是此前状态退化的唯一来源，也不证明关闭adapter可直接替代B或恢复训练。

## 干预与身份

B19 core SHA `0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984`；runtime `33d5277c9c232bd817302aee1cf5d45823510754`。原B19模型、optimizer、targets、arm身份都不变，38个历史更新，无新增更新。真实父组`8dde57eac4325d7701fafe97d48f57cb962cd8ff121a9e4770a1bbdffa10db4e`，standing源4.4s，25.984cm mid，release10.8s，恢复11.8s，结束14.8s。

唯一干预：不把29D execution-adapter增量加到BFM动作；原U/Guidance→合法z保留。offsample只采一次原std.01/clip.06噪声，offmean去噪。不是置arm=A，不改权重，不夹q_PD，不改真实q±0.02rad硬界、reward或物理。原on轨迹复用[已逐值复现的诊断](INTERACTION_PILOT_B_FAILURE_RESULTS.md)，没有重跑训练。

模型/RNG摘要、首qpos/qvel、slot、task、全部370条接管前控制记录、首BFM观测和首实际插值z逐值一致；前75个高层命令（含首干预命令）一致。之后状态、观测和高层命令按真实闭环演化，不要求它们继续相同。完整checkpoint恢复使用原严格验证路径，无新比较容差。off两条各150个诊断50Hz控制/600个5ms物理tick；没有换父组或跨episode重置。

## 实测

全窗口表：on遇硬界立即停，off到14.8s。时长不同，不能直接把全窗p95差异视为同窗因果效应。

|分支|实际续跑|最大实际q越界 rad|左踝pitch最小q rad|加载滑移峰 m/s|加载滑移p95 m/s|末root速度 m/s|无支撑tick|
|---|---:|---:|---:|---:|---:|---:|---:|
|on采样|0.515s|0.021384|-0.894054|2.462716|0.329106|0.668848|16|
|on均值|0.510s|0.027559|-0.900229|2.379839|0.328603|0.609022|16|
|off采样|3.000s|0.000000|-0.622415|0.606027|0.021672|0.004686|0|
|off均值|3.000s|0.000000|-0.622240|0.524111|0.021964|0.001618|0|

所有分支实际腕部外力为0，均属于已释放后的恢复；不再评价旧anchor跟随。实际左踝pitch范围[-0.87267,0.5236]，on的越界真实发生，不能解释成合理收步。off达到3秒时root速度分别约0.00469/0.00162m/s，所有实际关节在机械范围内。

相同11.8→12.31s窗口（102个tick）：

|分支|滑移峰 m/s|滑移p95 m/s|root平面净位移 cm|末root速度 m/s|最低高度 m|pelvis相对恢复点yaw变化|
|---|---:|---:|---:|---:|---:|---:|
|on采样|2.462716|0.332385|9.632|0.633851|0.658716|-3.362°|
|on均值|2.379839|0.328603|9.060|0.609022|0.666648|-8.853°|
|off采样|0.606027|0.220862|10.850|0.052653|0.724671|-14.839°|
|off均值|0.524111|0.137767|10.563|0.052458|0.724673|-13.699°|

滑移沿原`stage_loaded_slip_mps`定义，未提高载荷门槛。off完整窗口仍有0.6060/0.5241m/s峰值，高于0.45参考线；**不是动作质量全过关**。off的共同窗口root净位移也没有变小，不能把“不越界”等同“所有多余运动减少”。世界yaw坐标保存在JSON；此处报告相对11.8s的角度变化，不能把世界朝向直接混作任务yaw误差。

## 动作层解释与边界

首状态左踝pitch：on均值q_PD −0.675103rad，off均值−0.333085rad；原adapter对该关节约−0.342018rad。此前on最后状态的BFM目标−0.8106rad、加adapter后−1.1598rad；本次真正关闭adapter后，物理轨迹避开原越界并恢复支撑，而非只比较离线目标。

但off期间左踝pitch均值q_PD仍分别有3/4个20ms目标超机械范围（上界侧，最大0.612759/0.657220rad），实际q未越界。不能把q_PD超范围与实际关节越界等同，也不能据此宣布任何q_PD硬裁剪一定必要或有效。

- **已证实**：在该B19完整真实释放状态，关闭执行adapter、保留学习U与其余合同，可让采样/均值两分支避开原实际越界；不是仅随机噪声导致。
- **相容**：当前adapter在无接触恢复域缺乏足够的实际动作约束/保持，造成危险控制修正；其是否由训练目标、覆盖或优化导致仍需区分。
- **未证**：adapter在所有任务有害、全回合关闭它优于B、只关左踝即可、某种clip或正则能解决；本次关闭的是全部29D adapter，不能定位为单关节唯一机制。

建议下一配方优先解决adapter的实际动作修正/无接触恢复保持，而非继续降探索噪声或直接同合同续B19。具体机制需要另行确定；**本次不改配方、不续B/C/dev**。旧A128与B19及所有失败保留，三臂研究尚未完成。

## 成本、失败与可复核资产

新GPUjob共431.728551s，另保守计准备/CPU分析/发布1200s；pilot累计32697.879392s＝9.082744/15h，聚合输出3.053/8GiB。完整初始化/重建及实际物理计数见[JSON](interaction_adapter_off_summary.json)。实际实验两分支exit0；一个只读探查脚本曾对Tensor调用keys报AttributeError，未涉及控制、未重跑物理，成本计入准备额，保留在本记录。未读sealed，未修改共享BFM/PD/main/其他进程；GPU0已释放。

独立stage `interaction-pilot-adapter-off-20261112`：`validation-protocol.json`、`validation-ledger.json`、`jobs/`、`runs/{offsample-B19,offmean-B19}/{probe-trace.pt,probe.json,same-state-actions.json,physics-counter.json}`、`adapter-off-analysis.json`。真实trace SHA分别 `267f87983907762b5caac8dd54cacf8add8ce4820257c5a2bdedb285fb88f2a5`、`88a575cddbdcb527cc0b407982f8a4bcf31ffe7cfb572f92c498b1e2a019076a`。协议[链接](INTERACTION_ADAPTER_OFF_PROTOCOL.md)，分析源码`research/interaction_fb/analyze_adapter_off.py`。原on资产仍在`interaction-pilot-b-diagnosis-20261112`，不覆盖。
