# F1 L0–L2结果：有限可执行性成立，尾部任务/质量仍取舍

2026-11-12。已完成授权L1/L2并停止，L3未执行。完整正式确认点：[F1_L2_GATE_REPORT](F1_L2_GATE_REPORT.md)。

- **已证实：** 12父组24状态、384条连续3s分支中，11状态/7父组存在任务/推进/slip/yaw参考合格候选；v3评分实际选中8/24。不是新训练或统一网络胜率。
- **已证实：** 六父组八个L2窗口，因果特权重规划参考合格3/8，固定BC500为4/8。P1-13误差/力7.373cm/5.594N→6.954cm/5.081N，slip .318→.312m/s，但yaw17.82°→19.56°；保留收益和代价。
- **已证实：** 尾部可出现大幅减力却不合格：hold-97ff接触窗17.046cm/9.210N→3.183cm/2.028N，窗口slip .179→.762m/s、末端abs yaw9.34°→28.37°。释放后旧anchor不计跟随，阶段表单列。
- **边界：** near-5f62误差/力略好且root/足部运动较少，但腕进度低于原参考约2.36mm，保留未过判定；不把边界小差异称严重失效。q_PD目标跳变与真实q分开，窗口合格不等于动作质量完美。

奖励、滑移三分与完整分段坐标见[F1_PRICING_AND_SLIP_RESULTS](F1_PRICING_AND_SLIP_RESULTS.md)。同状态候选排序与硬质量参考存在取舍，不等于实现bug；本次没有新增学习更新，所以探索/信用/保留的因果比例仍未识别。

## 信息与恢复证据

C-S只用原BC500受限ideal-force输入，C-P不读真实未来；C-S planner仍缺格，别名未构造。L1为O-F 16候选/3s，L2为C-P 8候选/2s，不是严格信息单因素对照。未来依赖仍未证，见[实际方法](F1_L1_L2_METHODS.md)和[逐query审计](f1_info_access_audit.jsonl)。

L0原manifest不变。两次L1恢复失败及allocated efc_JT保守恢复、三秒干预、辅助滑移统计更正均单列[erratum](F1_ERRATUM.md)，未扩大mask、放宽容差、修改科学判据。原始失败和已资格分支分开保存，合并384条不重复。

## 身份、资产与成本

模型全部冻结，主L1补全runtime `f42fb177fe3679d0a162ace03b22acb5196279c4`；L2 runtime `ad93901e9b75791ae918dd37664b7d6f8b8e86bb`；早期有效分支另存f03/e31身份，发布HEAD不是执行版本。模型与原生来源SHA见[f1_model_identities](f1_model_identities.json)。

含L0、所有失败、CPU复算、IO及明确准备估计累计3.1590 job小时；L3支出0，训练决策/优化更新0。详见[f1_cost_ledger](f1_cost_ledger.json)。源/预测/实际执行分开，原始完整状态和大轨迹仅在个人stage保存；公开仅源码、汇总和哈希。

实际轨迹视频索引见[F1_VIDEO_INDEX](F1_VIDEO_INDEX.md)，为完整3s诊断窗，不是完整回合或预测动画。下一单一可判别假设及成本见[F1_DECISION_TABLE](F1_DECISION_TABLE.md)。BC500仍为基线，不反选旧模型、不读封存、不自动PPO/Student/FB/L3。
