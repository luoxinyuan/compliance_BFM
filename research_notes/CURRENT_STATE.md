# CURRENT_STATE — Issue60 F1停于L2正式报告点

最后核对2026-11-12。[L2 gate报告](F1_L2_GATE_REPORT.md) · [汇总](F1_RESULTS.md) · [方法/权限](F1_L1_L2_METHODS.md) · [成本](f1_cost_ledger.json) · [Issue60](https://github.com/GaoE05/robot-research-bridge/issues/60) · [Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。

## 当前执行与授权

用户授权的L1/L2已完成并停止；没有L3或训练自动队列。L3仍需Yi明确确认，其3h未使用。原BFM/PD/BC500及所有历史模型不改，未读取任何封存集。GPU占用/资源必须下次实时核对，不能沿用本页快照。

## F1实际证据

L0：当前MuJoCo3.3.3 Newton dense、island禁用下25个NULL数组经156条3s C指针干预通过；27状态有效恢复/测速完成。模型与状态manifest锁不变，规则仅适用于已验证配置。[逐数组审计](F1_L0_ARRAY_AUDIT.md)和[完整L0](F1_PREFLIGHT.md)保留。

L1：12已暴露父组24状态，384连续3s候选；11状态/7父组存在任务/推进/slip/yaw参考合格候选，v3实际选合格8/24。有限存在性，不是共享checkpoint或泛化成功。L2：六父组七初态＋一个相关continue分支，八窗口；C-P参考合格3/8，固定BC5004/8。部分因果反馈降低误差/力，但尾部仍明显转移到滑移/偏航。near的2.36mm进度参考差与尾部严重代价分别解释，未改判据。

L1 O-F与L2 C-P候选数/时域不同，未来oracle依赖未证。C-S只有原ideal-force BC500对照，受限planner缺格，饱和别名未构造。没有训练，没有新的探索→GAE→更新因果比例。[定价/滑移/分段表](F1_PRICING_AND_SLIP_RESULTS.md)与[中立决策](F1_DECISION_TABLE.md)给出证据等级。

## 失败、身份与资产

两次L1完整恢复检查失败原判保留；对allocated efc_JT显式全量恢复并做3s前后物理干预，未新增mask/容差。早期辅助滑移时长重复计点错误已从原接触行重算，不影响原物理/reward/选择。[F1_ERRATUM](F1_ERRATUM.md)。

BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；BC500原SHA `a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`。F1读取BC500零支路扩展及B64/B128/v3-32/128，全部独立冻结。精确路径/模型SHA见[f1_model_identities](f1_model_identities.json)。L1主要runtime f42fb17，L2 ad93901；发布commit不代替实际runtime。

服务器项目相对stage `experiments/f1-l1-l2-20261111`；L1原始l1-01/02/03及合并索引，L2 l2-01，所有失败/预测/真实执行/状态快照保留。含L0与准备估计总3.1590job小时，L3=0，仍计入原240h而非重置。视频见[F1_VIDEO_INDEX](F1_VIDEO_INDEX.md)。

## 不可丢失的旧结论

F-RL-v2接触前保持机制有效，但A/B均未晋级；v3部分风险坐标改善同时有尾部和旧停止退化，BC500仍是基线。正式科学候选v3-32与末轮128身份不混淆。[v2](ISSUE58_EXECUTION_RESULTS.md) · [v3](FRL_V3_RESULTS.md)。有限直行Teacher→ideal-force Student历史成果保留，不宣称已具备G1真实腕力感知。

FB eager更新一致性、portable状态覆盖、compiled资格、踝交接/运动质量是不同层级。L0/F1仅补portable有效域/恢复证据，没有新FB update或长跑，不追溯改判旧失败。P0-B质量负例保留，不自动重启1M/5M。旧erratum、5403b7未晋级、T30/S0/M38等历史结果保留。

## 下一会话接力

先读本页、OVERVIEW、F1_L2_GATE_REPORT及Issue60/Master。当前唯一新增建议是L3的匹配预测权限诊断，尚未获准；不恢复旧PPO或自行占用3h。下一步先核真实进程、磁盘≥100GiB、runtime/模型/状态身份与剩余额度。只改个人research，main/共享资产/其他任务不动，无真机。Library旧PROJECT_STATE/SESSION_HANDOFF仍为历史快照，不用其覆盖实时GitHub状态。
