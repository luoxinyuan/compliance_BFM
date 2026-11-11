# CURRENT_STATE — F1正式收官；Issue61仅S0获准

最后核对2026-11-12。先读[长期路线](RESEARCH_OVERVIEW.md)、[F1最终](F1_RESULTS.md)、[L3匹配结果](F1_L3_RESULTS.md)、[Issue61](https://github.com/GaoE05/robot-research-bridge/issues/61)、[Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。

F1 L0–L3已结束，无训练/封存读取。L3两个父组三个窗口：O-F改善停止、继续有收益及代价、tail-hold与C-P真实轨迹完全相同；不能简单归因未来缺失。C-S planner缺格；BC500不变。完整错误保留，L3首次reset顺序错误已修复，未改容差。运行61b4371；模型身份见[f1_model_identities](f1_model_identities.json)，费用见[f1_l3_cost_ledger](f1_l3_cost_ledger.json)。

后继Steering-first：Goal、Guidance、Execution分离，原BFM无ERI端口不能伪装有；先审查原BFM/P-BFM/I-BFM。S0≤2h；不自动下载大资产、不重建环境、不训练、不S1物理实验。S0结束提交接口资格与S1预登记草案，等待用户确认。GPU0/磁盘≥100GiB实时复查，不能用旧快照。

## 不可丢失的旧结论

F-RL-v2接触前保持机制有效，但A/B均未晋级；v3部分风险坐标改善同时有尾部和旧停止退化，BC500仍是基线。正式科学候选v3-32与末轮128身份不混淆。[v2](ISSUE58_EXECUTION_RESULTS.md) · [v3](FRL_V3_RESULTS.md)。有限直行Teacher→ideal-force Student历史成果保留，不宣称已具备G1真实腕力感知。

FB eager更新一致性、portable状态覆盖、compiled资格、踝交接/运动质量是不同层级。L0/F1仅补portable有效域/恢复证据，没有新FB update或长跑，不追溯改判旧失败。P0-B质量负例保留，不自动重启1M/5M。旧erratum、5403b7未晋级、T30/S0/M38等历史结果保留。


Library旧快照不覆盖GitHub当前状态；仅个人research分支普通fast-forward，main/共享BFM/PD/他人任务不动。
