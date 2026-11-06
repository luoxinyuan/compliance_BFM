# CURRENT_STATE — 新会话先读

最后核对：2026-11-06 CST。**Issue47完整结项并停止：物理接管、两seed PPO、196回合终检、3条Mac视频已完成。未满足联合延长门槛，没有继续训练/Student/下一阶段队列。有限正结果成立，但不推广替换旧基线。**

[当前完整报告](C0_ENGAGEMENT_RESULTS.md) · [协议](c0_engagement_protocol.json) · [Issue47](https://github.com/GaoE05/robot-research-bridge/issues/47) · [Master18](https://github.com/GaoE05/robot-research-bridge/issues/18) · [长期纲领](RESEARCH_OVERVIEW.md)。ChatGPT Library2026-10-28为历史快照，不作实时状态。

## 目标和可靠基线

Contact-conditioned whole-body behavior selection/modulation/transition：依据实际接触、身体状态及可维护记忆，合理组织局部手臂、躯干/root和支撑。冻结BFM/原PD，不预设先手后脚，也不把root走远等同任务合适。有限直行Teacher→ideal-force Student、原生伸臂和双向小转腰已有真实证据；本轮没有训练Student或验证真机力传感。

|身份|SHA256|
|---|---|
|BFM|33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361|
|I行走基线|72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a|
|C0局部/迟到行走基线|ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c|
|T30|c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33|
|S0 Student|0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12|
|D2|9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5|
|4601末轮参考|8fa4759d3766568dee8fbba0367e772a97977694d19a503868708efa6d8c9228|

C0是I起点混合750条六类示范、2000监督更新的**原冻结checkpoint**。本轮PPO后继另存，没有覆盖。C1恢复walking但丢局部、F/R监督未稳定共存、M41启动历史退化，以及Issue46停止取舍全部保留。旧S0/T30和M8/M9已验证能力不因新研究模型退化而被否定；不按任务拼checkpoint。

## Issue47：实际结论

|模型|新6.5cm|新20.5cm|新68cm|
|---|---:|---:|---:|
|原C0|2.446cm / 2.045N|6.429cm / 5.237N|31.631cm / 8.568N|
|I行走参考|4.189cm / 3.476N|5.378cm / 4.417N|8.527cm / 5.487N|
|4601末轮参考|3.608cm / 2.980N|4.675cm / 3.865N|4.931cm / 4.066N|
|4701/batch8|2.466cm / 2.060N|6.434cm / 5.245N|20.462cm / 7.401N|
|4702/batch8|3.390cm / 2.823N|6.418cm / 5.228N|27.040cm / 8.155N|

1. **有限共同收益**：同4701/batch8保住新6.5cm局部表现，新68cm误差−35.3%/力−13.6%；实际足部/root推进有改进。另一seed局部退化，不能声称稳定共存或general成功。
2. **仍然启动迟到**：新68cm持续root推进13.69→11.65s，仍远晚于I7.545s。移动期28.448cm/8.147N→22.826cm/7.968N，保持期41.492cm/10N→5.738cm/4.683N。主要收益在保持追赶；残余hold XYZ偏差(+1.791,+5.255,−.215)cm，Y负担增加，不能只看X追上。
3. **恢复域有明确证据**：完整真实C0前缀→I@6/7.5s有效，@9/10.3/11s不可靠；I→C0@7.5s可维持行走，@6s不行。近6cm切I反而退化。不把早期有效I标签推广到全部状态。
4. **高层确实参与**：固定6s的实际z、BFM实时反馈＋真实力，root只约3.81cm且不出现正常C0步态；正常C0约49.98cm。固定z且真实撤力约.44cm。不是完全被动机器人，也没有“主动百分比”结论。
5. **边界**：5N长距未改善（4701约42.219cm/4.544N，I6.549cm/3.511N），自然arm-ready20.5cm无实质收益，4702新局部退化。不能把全部问题归为force太小或BFM不会走。旧转腰有限保留，不冒称迈步转身。

两seed各16×768从原C0 actor/normalizer/std精确相同起点开始，fresh critic/Adam，原Direct256D/10Hz→5×50Hz→原PD/200Hz，reward/std=.008/历史/K80D8/10N不变。无在线I/M10/Cyclic、无BC/replay标签、无新Student。仍为1088D**特权Teacher**，含当前anchor误差/伙伴相对速度，不能称force-only或实机部署模型。

开发规则均选batch8，4701总体.9796、4702.9943；末轮.9967/1.1089，后者near1.5100。联合延长门槛不满足，未看终检再调参。[曲线/全部候选](c0_engagement_artifacts/ppu-training16.json)。新终检6标签仅5种唯一物理条件（continue与long68重合）；共同前缀7.61s前physics/命令/观测精确一致，7.7s之后才随真实需求改变。

## 下一步唯一推荐（未授权执行）

**在C0实际早期可恢复状态中，比较现有Gaussian PPO与已有I有效反馈方向引导的训练期探索。** 保持reward/物理/合法z，保留局部与相同前缀停止/继续；最终是一个学习策略，不部署手工force/时钟切换。先利用已证实存在的方向，不原样延长PPO或新建大型reward-prompting/F-SF工程。探索/信用分配/参数干扰尚未隔离为唯一根因。

原生reward prompting、reward composition、F/SF与force-history网格仍在[积压记录](BFM_REWARD_REUSE_ROADMAP.md)，不是自动发车权限。蒸馏初始化→受限观测Student真实闭环RL→可靠专家聚合是后续构想，见[方法讨论](CONTACT_LEARNING_RATIONALE.md)，本轮未执行。

## 源码、模型、资源与交付

- 长期个人分支`gaoe05/research`，最终文档commit见Issue47；**运行runtime**：物理d24622b259f984a42a3f628e983a15e8ea16d547，正式PPO eeeb8979258b774a87dbedf8b17d18e3d15fc47a，终检/渲染c6f4a31c3ad32c62d2ce8be810a85fe0d195fa46。报告commit不混作runtime。
- 4701选定8：`f23a6818dab277b775b41d5926641c359546cd6d5f6b2f2003a33514c446e2f6`；4702选定8：`37f2c7e54375df301c9cc8160b2363ff8b8eb03edf251b5b1b93cdac583ef4ce`。全部初始/末轮/旧模型精确路径与SHA见[训练清单](c0_engagement_artifacts/ppu-training16.json)、[终检manifest](c0_engagement_artifacts/final-index.json)。
- PPU私有根`/root/compliance-ppu-20261106/issue47`：takeover-01、ppo4701-r1/ppo4702-r1、final-02。源码各独立worktree，原始qpos/历史/失败不公开。旧NVIDIA实验cc576552及共享6023c6e工作树保持原样。
- 26112实际新增RL决策（正式24576＋失败1536），72正式Adam更新；0新Student/监督。32诊断＋196终检回合，另120开发评估和213正式训练完整回合。[保守结算](c0_engagement_artifacts/resource-closeout.json)：9080/14000累计job秒、约6.53/9GiB，未超预算，只用1张PPU，无CPU BFM替代。所有owned训练/评估/渲染结束，无自动续训。
- 196终检无安全终止/关节越限/力矩饱和。最大tilt10.69°；4701末轮有1个5ms双足载荷都<5N tick。q_PD20ms峰62.54°不等于真实q5ms峰3.03°。XYZ/力矩/滑移代理/分段/释放保留[全矩阵](c0_engagement_artifacts/final-summary.csv)及每模型JSON。释放后不考核旧anchor。
- Mac `~/Projects/compliance-bfm-diagnostics/videos/c0-delayed-engagement/`仅3条1×真实qpos MP4＋INDEX：01因果恢复窗口，02同模型新6.5cm局部保留，03新68cm迟到推进及失败。SHA/时长/帧数/完整解码/画面已验证；不是reference动画。主展示4701在终检前固定，另一seed负结果完整报告。

## 失败与接力规则

PPU首次概率路径失败1536样本/0正式更新保留；修复learner与采样同batch-one算术路径后，原3e-4断言/raw概率/std/KL/Adam回滚通过。没有据此否定旧NVIDIA训练。[PPU兼容](PPU_COMPATIBILITY_RESULTS.md)不能当跨硬件逐位相同。初始化/私有symlink失败、离线pre/post时间窗和多点cap定义修复全见报告；原训练reward/物理未变。

新会话先读本文件、RESEARCH_OVERVIEW、当前完整报告与Issue47，再按需读Issue46/M41。**不重启已结束训练，不沿用旧PID/GPU空闲快照**。下一阶段需新科研授权；源码审查和只读分析可继续。禁止修改共享BFM/PD/权重/main、他人任务、SoftSONIC，不做真机。旧Library是历史快照。
