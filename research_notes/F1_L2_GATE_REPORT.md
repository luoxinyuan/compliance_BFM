# F1 L2 正式报告点：已停，L3未授权

L1/L2执行完成；本报告不是训练成功、封存晋级或部署资格。BC500基线不变。L3没有启动或排队，3h完整保留。方法与权限详见[F1_L1_L2_METHODS](F1_L1_L2_METHODS.md)，历史与本轮修复见[F1_ERRATUM](F1_ERRATUM.md)。

## 覆盖与实际结果

L1：12父组24状态，384条连续3s候选；11状态/7父组至少存在一个同时满足任务、推进、滑移与yaw参考的候选，v3评分实际选中合格8/24。此为有限存在性证据，不是一个共享网络的成功率。

L2：六父组、七真实起始状态，加一个stop/continue相关分支，共八个3s窗口；C-P每100ms重规划，共240次。固定BC500合格4/8，C-P合格3/8；两个分支或同父组多个状态不能当独立重复。全部是已暴露条件。

|状态|C-S误差cm/力N|C-P误差cm/力N|slip峰m/s C-S→C-P|末端abs yaw° C-S→C-P|完整窗口合格 C-S→C-P|
|---|---|---|---|---|---|
|P1-1|2.569/2.071|2.352/1.845|0.006→0.006|11.64→11.33|True→True|
|P1-13|7.373/5.594|6.954/5.081|0.318→0.312|17.82→19.56|True→True|
|new_tail-sustained-5c67db5a4802|12.456/8.568|9.175/6.900|0.865→0.758|12.02→2.01|False→False|
|new_tail-sustained-97ff10c4dcd9|13.334/8.433|8.354/5.929|0.499→0.595|10.41→2.81|False→False|
|new_tail-hold-97ff10c4dcd9|17.046/9.210|3.183/2.028|0.179→0.762|9.34→28.37|False→False|
|new_near-sustained-5f62e065b5ac|2.786/2.088|2.687/2.027|0.005→0.005|12.12→7.28|True→False|
|new_stop-sustained-5f6380cdec75|12.506/8.081|6.205/5.076|0.292→0.007|6.08→5.35|False→True|
|new_stop-sustained-5f6380cdec75-continue|7.619/6.277|10.085/6.960|0.396→0.811|18.89→1.80|True→False|

每项XYZ、cap、root/脚步、q_PD/真实q、支撑/力矩及分段数据见[f1_phase_metrics](f1_phase_metrics.json)；释放后anchor误差不计接触任务。完整窗口参考不等于完整回合恢复。逐坐标未过原因和进度差见[f1_window_failure_coordinates](f1_window_failure_coordinates.json)，不把边界小差异与严重退化混为一类。旧相对门的完整回合族统计不由这八个短窗替代；窗口相对风险坐标另列，不宣称旧门晋级。

## 因果与执行合同

每个窗口首次查询未来spec干预测试，共8项逐值通过。stop/continue共同前缀核验：`[{'test': 'real C-P stop/continue prefix', 'parent': '5f6380cdec755f630bc43932fd8d3a1b6deddc055f7fcd6bceb4e3d396a812bb', 'physics_ticks': 100, 'commands': 5, 'passed': True}]`。C-S共同前缀核验：`{'physics_ticks': 100, 'commands': 5, 'passed': True, 'fields': {'inputs': True, 'raws': True, 'target_z': True, 'previous_z': True, 'physics': True, 'controls': True}, 'first_force_difference_time': 5.504999999999905, 'first_anchor_difference_time': 5.504999999999905, 'first_restricted_input_difference_time': 5.599999999999903, 'first_raw_difference_time': 5.599999999999903}`。同一实际首命令及首BFM子步对齐240次，原BC500 L1/L2独立重复全量一致=True，见[query alignment](f1_query_alignment.json)。

C-P只用当前伙伴位置/速度/active/yaw的常速预测，不读取实际未来。C-S只有原1639D ideal-force BC500固定对照，**C-S planner缺格**。饱和力隐藏状态别名未构造，不做信息论不可辨识结论。原始恢复快照仍带实际Task spec，future-access字段只指候选/评分接口，不宣称OS级隔离。

L1的16候选/3s与L2的8候选/2s不同；因此当前证据**不能证明必须依赖未来oracle**。具体有限失败也不能证明所有合法latent不可达。

## 延迟与资源

重规划p50 17.754s、p90 18.544s，对应实际0.1s首命令；这是离线反馈测量，不是实时算法。L1实测0.5474h，L2实测1.3431h；含L0、失败、CPU、IO、准备估计累计3.1590h，未授权L3支出0。账本见[f1_cost_ledger](f1_cost_ledger.json)，准备估计与实测分别列出。

预测分支与真实C-P执行分开记账，所有模型冻结；RL决策/训练更新均0。runtime与checkpoint见[f1_model_identities](f1_model_identities.json)及逐query身份，发布HEAD不冒充运行SHA。

## 后续确认点

L3的单一候选假设、反证与预计成本在[F1_DECISION_TABLE](F1_DECISION_TABLE.md)列明。**本次到此停止；未获Yi明确确认不得运行L3或占用其3h。**
