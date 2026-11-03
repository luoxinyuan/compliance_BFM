# CURRENT_STATE — 新会话先读

最后核对：2026-11-04 01:54北京时间。**M39-A正在执行，尚未结项。** M38-C有限域双向接触转腰已成立，本轮研究需求分配与真实闭环优化，不重复BFM左转资格。禁止根据此快照重启已运行任务；先查实时ledger和进程。

入口：[研究纲领](RESEARCH_OVERVIEW.md) · [M38-C完整报告](M38C_RESULTS.md) · [当前Issue39](https://github.com/GaoE05/robot-research-bridge/issues/39) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。Master仍保留M38-C结论，M39完成后再统一更新。

## 可靠基线与本轮科学问题

T30→S0有限域右腕直行/停止Teacher→ideal-force Student保留；ideal-force不代表已确认G1真机力传感能力。D0/D2在同一checkpoint实现有限域双向接触转腰，主要是腰相对骨盆，不等于迈步转身。D2仍有零需求偏转、小请求附带转腰及组合干扰。

用户授权从D2开始：A扩覆盖监督；B/C相同初始化的两个独立真实PPO seed。1088D特权Direct→合法256D z→冻结BFM→原PD，原50Hz/200Hz节拍。右腕5N或torso_link两点各2.5N，总5N。没有在线force-to-yaw、M10 Student或方向选模型。当前特权伙伴需求可以给Expert，不是部署Student权限。原BFM/PD、T30/S0/D0/D1/D2和历史失败完整保留。

## 已完成、正在运行与尚未执行

**已完成：**38条独立准备物理回合，独立逐点reward复算；D2/T30已知零需求及平移差距被现有reward正确排序，不改reward。首次准备在29回合后发生NumPy bool序列化失败，修复后复用已有轨迹，只补9回合；不能把29+38重复计样本。原生小±5°来源未优于对应T30，不作训练标签。

A覆盖监督完成3000更新、14条件开发闭环。10条有效源回合1554标签，过滤后1426。零需求、小请求、双点平移改善，左右±15°保留，但早/晚停止退化，组合仍有错误方向；**不作为全面替代**。该组采用合法z监督，与M38-C动作/短序列方法不同，不声称是严格仅改覆盖的单因素因果比较。

**正在运行：**B seed3901 GPU2、C seed3902 GPU3，每批768真实10Hz决策，3worker；各24批18432决策，dev24进行中。actor/归一化/std逐tensor与D2核对一致，critic/Adam新初始化。std固定.008，解析joint KL受控；raw logprob复现误差约1.22e−4。每8批14条件完整均值开发，不用EV或采样平均return宣布成功。

B dev8严重小左退化已按预固定规则回滚、减小actor lr；dev16未胜初始化。C开发选择暂为batch16，综合分略好，但正常直行和晚停止仍退化。开发选择随后续合法开发结果可变，**不是最终模型**。所有候选、回滚、initial/best/latest/last保存。

**尚未执行：**新的±9°/组合/连续变化等封存物理终检、最终因果时间曲线、2–4条真实qpos视频、阶段结项与Master更新。没有新Student训练、PPO方法比较、真机或新接触部位。旧视频不冒充本轮交付。

## 精确运行与模型入口

项目根`/mnt/data/temp_user/yi.gao/projects/compliance_BFM`，以下路径相对`bridge-v2-owned/runs/`：

|身份|路径|SHA256|
|---|---|---|
|D2初始化|m38c-history-20261103/C-train-05/job/D2.pt|9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5|
|D0参考|m38c-history-20261103/A-train-06/job/D0.pt|5e1047b940950208aa514f5ecc26ca75a2dff631f193cbe09457138f9d59c20d|
|T30|stop-sensing-20261103/B1-train-02/job/best.pt|c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33|
|S0|direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt|0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12|
|A覆盖监督|m39a-demand-20261104/A-train-05/job/coverage.pt|ad452ad5efb69bf57e7c121a83a62aef7f1143cc576e7839f39a6d13b1b322be|

BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。

本阶段根`m39a-demand-20261104/`：`compute-ledger.json`是资源/进程入口；`A-prepare-02/job`复用首失败轨迹；`B-ppo-03/job`与`C-ppo-04/job`分别有`status.json`、`training.jsonl`、`selection.json`、全部dev和rollback。检查`best.pt`必须同时核对selection的iteration/SHA，不能根据文件名推断身份。

PPO实际runtime **6f1789269de53a428aa5f5383eff0adcf247c04d**，独立worktree`bridge-v2-owned/trees/m39a-ppo3901/repo`及`m39a-ppo3902/repo`；当前owned parent PID2108915/2108934只是本次快照，必须live核对，不按旧PID操作。准备修复/覆盖runtime **3a4ff2759c15a1d9b4289e3124d76ef6e5c4370e**，主开发worktree`bridge-v2-owned/trees/m39a-demand/repo`。文档/分析更新不得修改两条运行worktree。

## 预算、停止与接力

上限24000累计受监督job秒/15GiB/3空闲GPU；每seed≤98304决策，总≤196608；新增Expert监督≤6000（已用3000）。两条PPO各保留8000秒墙钟上限，预定开发严重退化回滚/降学习率，三次停止；64批后若无预定改善趋势停止。不能把128批上限当必须跑满或收敛标准。资源检查已确认GPU1–3可用后才启动；0/5/6他人任务不动。A结束后GPU1释放，新任务仍须live核验CPU/RAM/I/O。

接力顺序：
1. 读ledger/两个result或status、selection、stderr，确认训练是否仍在运行，**不重启不覆盖**。
2. 待两条结束后锁定开发选定best与last身份，运行[m39a_protocol.json](m39a_protocol.json)封存条件一次；终检不能反选/补训。保留D2/T30/A/B/C及必要D0参考，不拼接方向模型。
3. 使用[m39a_analyze.py](m39a_analyze.py)和[m39a_figures.py](m39a_figures.py)统计真实逐点负担、接触增量、连续变化时间曲线、支撑/连续性/释放恢复。双点比较统一RMS(sum point norms)，不可混用合力或其他force_rms。释放后不计anchor跟随失败。
4. 用已有渲染器和[m39a_render_saved.py](m39a_render_saved.py)从保存qpos输出2–4条1×视频；验证后传Mac`~/Projects/compliance-bfm-diagnostics/videos/m39a-demand-allocation/`，该目录本次尚未交付。
5. 完成报告、CURRENT_STATE、Issue39和Master18；审查全待推历史，仅个人`gaoe05/research`普通FF整合/推送。权重/原始轨迹不进Git。

团队main、共享BFM/PD/权重、SoftSONIC不改，无Bridge/watch、无真机。Library2026-10-28为历史快照。阶段尚未完成，不把本文件的源码推送当作统一需求策略训练成功。
