# F1 L3 最终匹配预测权限实验

状态：完成并停止；无训练、封存读取或下一搜索。授权为Issue60追加评论6801908284，替代此前两尾部状态建议。

## 固定比较

stop/continue为同一父组，加一个tail-hold父组，共三个3s窗口、两个独立父组。C-P旧档案复用先验证完整初态、当前input/last-z、八候选描述、首次查询八条完整预测轨迹/评分、全部实际决策输入和三秒physics/control逐值一致。不得当作三个独立样本。

O-F与C-P均8候选、2s预测、100ms首动作/重规划、同一v3评分和原质量参考。只改变伙伴预测轨迹与未来active/release来源；heading成本distance标度仍按当前anchor和2s当前速度推算。O-F不是可部署或因果控制；其共同前缀允许提前产生不同动作，这正是未来信息干预。候选生成仍只读当前真实状态。

|窗口（两个父组）|C-P误差/力|O-F误差/力|slip峰 C-P→O-F|末yaw C-P→O-F|参考合格 C-P→O-F|
|---|---|---|---|---|---|
|new_tail-hold-97ff10c4dcd9|3.183cm / 2.028N|3.183cm / 2.028N|0.762→0.762|-28.37→-28.37|False→False|
|new_stop-sustained-5f6380cdec75|6.205cm / 5.076N|5.766cm / 4.702N|0.007→0.002|-5.35→-5.79|True→True|
|new_stop-sustained-5f6380cdec75-continue|10.085cm / 6.960N|9.344cm / 7.026N|0.811→0.435|-1.80→0.17|False→False|

## 权限与真实执行

90次O-F选中raw均与实际执行精确一致；首BFM子步bfm_obs/z/action/q_PD及首100ms物理q/qdot/force/anchor/support/feet全部逐值对齐（见机器表）。预测分支与真实执行独立标记，未作为PPO或模仿数据。
C-P原未来spec不变性与stop/continue100tick/5command共同前缀验证保留。本轮全实际轨迹精确重放，未改变其权限结论。C-S只有固定BC500，C-S planner和控制相关饱和别名仍缺格。

## 分段真实物理

释放后原anchor误差不再计任务；三秒窗口不是完整回合。表中yaw为有符号pelvis末角，不以转腰代替脚步转向。完整q/qdot、逐点force、cap、torque、支撑和足路径在JSON。

|窗口/层|阶段|误差cm XYZ RMS|力N XYZ RMS|root X推进cm|腕 X推进cm|slip峰m/s|末pelvis yaw°|
|---|---|---|---|---|---|---|---|
|new_tail-hold-97ff10c4dcd9/C-P|holding|1.255/2.603/1.334|0.737/1.593/1.016|17.115|6.457|0.762|-27.53|
|new_tail-hold-97ff10c4dcd9/C-P|released|释放不计|0.000/0.000/0.000|-0.023|-0.398|0.004|-28.37|
|new_tail-hold-97ff10c4dcd9/O-F|holding|1.255/2.603/1.334|0.737/1.593/1.016|17.115|6.457|0.762|-27.53|
|new_tail-hold-97ff10c4dcd9/O-F|released|释放不计|0.000/0.000/0.000|-0.023|-0.398|0.004|-28.37|
|new_stop-sustained-5f6380cdec75/C-P|moving|1.228/3.544/3.136|1.561/2.769/2.916|0.349|2.341|0.006|-7.49|
|new_stop-sustained-5f6380cdec75/C-P|decelerating|6.025/0.662/3.278|5.132/0.403/2.438|0.695|1.682|0.007|-5.47|
|new_stop-sustained-5f6380cdec75/C-P|holding|6.214/0.558/0.407|4.983/0.463/0.337|0.002|-0.285|0.001|-5.35|
|new_stop-sustained-5f6380cdec75/O-F|moving|1.852/2.097/1.256|2.104/1.415/1.061|0.164|1.591|0.002|-6.89|
|new_stop-sustained-5f6380cdec75/O-F|decelerating|5.884/0.177/0.379|5.012/0.159/0.217|0.748|1.748|0.002|-6.16|
|new_stop-sustained-5f6380cdec75/O-F|holding|6.265/0.323/0.068|4.992/0.261/0.058|0.240|0.458|0.001|-5.79|
|new_stop-sustained-5f6380cdec75-continue/C-P|moving|8.546/7.014/3.191|5.964/4.455/2.537|18.030|19.572|0.721|1.24|
|new_stop-sustained-5f6380cdec75-continue/C-P|holding|0.350/5.971/0.432|0.464/4.404/0.360|5.004|1.134|0.811|-1.80|
|new_stop-sustained-5f6380cdec75-continue/O-F|moving|6.179/2.616/3.396|5.190/2.247/2.881|4.121|9.262|0.435|-2.08|
|new_stop-sustained-5f6380cdec75-continue/O-F|holding|8.823/8.570/1.379|5.644/6.045/0.645|8.115|6.585|0.301|0.17|

## 滑移三分（原事件全保留）

原加载>1e-6N、速度>.45m/s保持；前/后速度沿同一约束Jacobian计算。持续支持运动还须看接触点/足路径，不因前后超限只持续单tick就宣称无滑移。q_PD目标峰值与真实q的5ms变化分别报告。

|窗口/层|原超限行/tick|积分后回落行|前后均超限行|超限最小法向力N|最长连续前后超限s|
|---|---|---|---|---|---|
|new_tail-hold-97ff10c4dcd9/C-P|3/2|3|0|29.387836519523923|0.000|
|new_tail-hold-97ff10c4dcd9/O-F|3/2|3|0|29.387836519523923|0.000|
|new_stop-sustained-5f6380cdec75/C-P|0/0|0|0|None|0.000|
|new_stop-sustained-5f6380cdec75/O-F|0/0|0|0|None|0.000|
|new_stop-sustained-5f6380cdec75-continue/C-P|3/2|3|0|23.19352850869941|0.000|
|new_stop-sustained-5f6380cdec75-continue/O-F|0/0|0|0|None|0.000|

## 失败、身份与成本

首运行1773f8f失败115.830369s：为缩短重建跳过前序reset，导致base奖励统计/episode计数与部分数组不等。没有生成科学比较结果。修复61b4371恢复原重建顺序，而非屏蔽字段；原L0 NULL规则与完整状态断言未变。旧错误保留，详见F1_ERRATUM。
实际L3 runtime `61b4371c00eab86e40dd4ef21b5a0123d2cfd355`；BFM/BC500及原L2模型身份不变。逐query/状态/执行文件SHA保留，发布commit不代替运行commit。

实测job 2227.130s＋明确准备/报告估计1200s＝3427.130s（0.9520h）≤5400s。F1原11372.572671389s继续累计，总4.1110h≤12h，计入原240h；不借剩余L3。实际90 oracle查询、24首次C-P复验候选，全部失败/IO/验证计账。完整成本与数据分层见[f1_l3_cost_ledger](f1_l3_cost_ledger.json)、[f1_l3_results](f1_l3_results.json)、[query ledger](f1_l3_query_ledger.jsonl)。

任何改善只证明当前有限搜索对预测权限敏感；任何失败都不证明所有合法z不可达。BC500不变。F1收官，后继仅S0接口审查获授权；S1需用户确认，PPO/FB/Student不重启。
