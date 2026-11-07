# CURRENT_STATE — 原生能力共享接触学习已结项

最后核对：2026-11-07 CST。**Issue48全部训练/冻结物理终检已结束，无新增实验在运行。原生主线已独立启动；C0仅冻结基线，不再是前置门槛。**

[Issue48](https://github.com/GaoE05/robot-research-bridge/issues/48) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18) · [本轮结论](NATIVE_CONTACT_RESULTS.md) · [方法](NATIVE_CONTACT_METHODS.md) · [协议](native_protocol.json) · [长期纲领](RESEARCH_OVERVIEW.md)

## 科学结论与当前瓶颈

全新高层actor/critic/normalizer/Adam，仅用同BFM原生goal/motion初始化，不继承I/C0/T30权重、在线输出/隐藏特征或训练标签。普通探索、原生候选协方差引导、同协方差下均值head坐标调整三组各32×768，合计73728真实RL决策；候选6个/实测7回合，初始化441时刻记录/7回合/2000更新，不是441条独立轨迹。一个配对seed4801，不宣称统计优越。

**部分反馈学习有效，局部→及时行走尚未解决。** 坐标组固定末轮C32相对新原生初始化，在新6.8cm局部由3.421cm/2.782N到2.823cm/2.300N；新停止12.523/7.925→8.551/6.936；连接零需求22.542/6.576→0.985/0.812。69cm仍34.920cm/8.756N，饱和67.6%，移动时root只前移约6.97cm；真实moving源新条件67.236cm/9.843N，尚不能适配伙伴或可靠进入/退出。不能归因成BFM没有walking，也不能据局部成功声称general Teacher。

三组开发best仍iteration0，确定性控制函数相同，full SHA不同不等于三个新成果。预登记最差相对cost规则对接近零的无接触基线敏感：C32无接触root约2.2mm→12mm/torso约4.17→6.53°，最差相对项盖过其他8条件改善。**保留原选模，不以封存重选；同时不能把best=0误写成没有PPO收益。** C32仅固定末轮研究模型，不全面替代旧I/C0/S0。

## 已完成与权限

- NVIDIA FP32 batchone/batched联合logprob≤1.23e-4，原3e-4/.02 KL/Adam回滚不变，无PPU路径。B/C初始768真实决策obs/action/logp/value/reward等逐值相同。
- 原生standing/局部goal可执行；motion有真实步态，但对低速牵引不适配。旧reward移动负例保留，没有新reward扫描/资产下载。原buffer复用历史全SHA并核对来源/大小，不冒称本轮重哈希26GB。
- 完成97个冻结物理回合（包含相同函数/重复continue条件，非97独立技能）、全XYZ/分段身体与动作质量、严格共同前缀停止/继续、5N回归。释放后旧anchor误差不计主动跟随。
- 新策略仍为1088D特权Teacher，当前伙伴误差/相对速度和实际力等可见；无未来/任务ID/最终距离/全局clock。没有新Student；ideal-force不是G1真机感知事实。
- 控制K80D8/主10N/回归5N、原BFM/PD、合法自由256D z及10/50/200Hz不变。无在线force→motion规则、无旧高层加载，不限制候选菜单/小半径。
- 全部原始失败保留；无物理安全终止不等于动作质量通过。C32长距q_PD最大37.68°、moving69.06°，是20ms目标变化，实际q/dq另测，未加平滑器或改PD。

## 精确身份与数据

RUN_ROOT=`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-contact-20261106`。

- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- 新原生初始化 `native-initial.pt` SHA `d399174cc2ddbd3efb858953653c9134495070948341381565f5f918f58da5cd`。
- A/B训练runtime `c1a8cc3e50d2258aa6afbf8d3ecf7b889c6250d6`；C runtime `1b99576ebca9bfa2c221cc382714e1d6184f5040`。运行worktree均保留，报告commit不冒充runtime。
- C32 `ppo-coordinate/last.pt` SHA `39cd0db55fddb174de3f77104acbd8d7574fb3a5051f743e4ca39bb6b7d24ad0`。A/B/各best完整身份在[native_artifacts/models.json](native_artifacts/models.json)。
- 账本`ledger.json`、各result/selection/launch、最终closeout是依据；训练status.json末条可能仍标running，是旧采样快照，**不得据此重启**。所有jobs已完成，无后续训练队列。
- 仅Pro5000 GPU0，其他任务不动。阶段上限24000累计job秒/15GiB/98304决策；实际见[native_artifacts/closeout.json](native_artifacts/closeout.json)，无Student/C0新训练。费用包含失败/初始化/资格/训练/评估/两版渲染；worker不重复计时。
- [长期集成分支gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)及实验分支`gaoe05/native-contact-learning`保存代码/小型摘要；最终报告commit以远程HEAD/Issue48精确链接为准，runtime如上不变。

## Mac视频（已实传验证）

`~/Projects/compliance-bfm-diagnostics/videos/native-contact-learning/`仅4条1× MP4+INDEX：01新6.8cm、02新69cm（I/C0/B32/C32四格）；03同C32共同前缀后停止/继续；04Standing/C0/B32/C32连接零需求。使用固定末轮32，不按终检选模型；早期best=0版本在私有分析目录归档。已核对SHA/25fps/正常时长/完整解码。原始qpos及QA不堆到用户视频文件夹。

## 历史、下一决策与接力

保留I `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`；C0 `ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c`；T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`；S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`；D2及全部历史失败。Issue47/4701仍为历史有限收益，PPU不可达，未在本轮新条件重评。M41历史启动退化、C1/F/R取舍、旧Teacher→ideal-force Student成果不被本轮任务抹去。

**唯一下一建议（未启动）**：原生动态步态段引导的时间连续探索，对照当前单时刻协方差探索，重点攻克进入/持续/停止；不恢复旧actor、不增force cap、不再自动追加C0修补。先预登记物理单位的开发退化尺度，避免近零基线比值绑架选择。C32局部/零需求成果可作为有限域Student BC vs同初始化Student PPO候选来源，但不自动展开general Student。

新会话先读本文件、OVERVIEW、Issue48与closeout，核对HEAD/result/ledger，不重启已消费目录。旧Library 2026-10-28仅历史快照。不改main/共享BFM/PD/权重/SoftSONIC或他人任务，不真机；新资源/实验须有新授权。
