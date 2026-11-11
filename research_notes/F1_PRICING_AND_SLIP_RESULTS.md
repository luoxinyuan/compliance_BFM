# F1：同候选定价、滑移三分与质量坐标

只做冻结档案复算，不调整reward、成功定义或原风险阈值。候选共享状态、同父组窗口相关；以下不是独立统计检验。

## 同状态奖励排序

L1在有合格候选的11状态内，共569对“合格/未合格”相关候选。旧reward偏好未合格84/569，v3为83/569。v3在3/11个有解状态仍选中未同时满足窗口参考的候选；其余13状态在本候选池未发现合格解。不能把这个候选对比例当PPO真实采样或优势错误率。

P1-13：BC500连续3s为7.373cm/5.594N、slip .318m/s，v3 return −5.939460；所选motion包络为7.627cm/5.402N、slip .955m/s，旧return −5.434299、风险扣除 .400805 后仍为−5.835105。力和heading累计成本的改善抵消了风险成本；这是当前加权评分与硬质量参考的实际取舍，不证明奖励实现错误或PPO失败仅由reward造成。

另两例是hold-7952的更低力与末端yaw>20°取舍，以及new-near的低误差/力但腕进度参考失败。不能把所有未合格候选称为全面物理劣解。原reward各项及同条件分项见f1_candidate_scores.json。

本轮没有产生新的训练样本/GAE或更新，因此没有新增“没采到/没记住/没保留”的因果比例；现有连续控制证据仅缩小可执行性缺口。

## 滑移三分口径

|集合|分支|旧超限contact行/tick|积分后回落contact行|前后均超限contact行/tick|超限最小法向力N|最长连续前后超限s|
|---|---:|---|---:|---|---:|---:|
|L1_all|384|442/294|361|81/61|3.718|0.005|
|L1_selected|24|37/17|34|3/2|25.980|0.005|
|L2_CS|8|6/3|5|1/1|18.184|0.005|
|L2_CP|8|9/6|9|0/0|21.873|0.000|

每个tick可有多个加载接触点，contact行数与tick数不能互换。所有旧风险事件保留；“积分后回落”不是无载擦地，也不保证下一阶段没有滑动。最长连续超限描述仅限0.45阈值，不能推出阈值下真实支撑位移为零。下表同时给材料点路径，路径按每脚每tick最大加载点运动累计，不按接触点数重复加时。

|状态/执行|旧超限tick|持续前后超限tick|瞬态only tick|法向力min/max N|左右材料点路径cm|q_PD最大变化°|真实关节5ms最大变化°|
|---|---:|---:|---:|---|---|---:|---:|
|P1-1 / L1-8|0|0|0|—/—|0.630/0.858|4.43|0.88|
|P1-2 / L1-5|0|0|0|—/—|0.558/0.754|3.95|0.79|
|P1-13 / L1-11|3|1|2|67.088/91.629|2.291/1.694|37.98|1.45|
|P1-14 / L1-6|0|0|0|—/—|1.653/0.575|2.65|0.53|
|new_tail-sustained-5c67db5a4802 / L1-11|2|0|2|33.866/91.067|2.349/1.622|30.60|1.26|
|new_tail-hold-5c67db5a4802 / L1-6|0|0|0|—/—|3.955/1.108|2.11|0.26|
|new_tail-sustained-12c3401ff67b / L1-11|2|0|2|36.813/147.450|2.675/2.009|33.13|2.32|
|new_tail-hold-12c3401ff67b / L1-6|0|0|0|—/—|2.392/0.934|23.41|1.45|
|new_tail-sustained-97ff10c4dcd9 / L1-5|0|0|0|—/—|1.382/1.605|41.39|1.44|
|new_tail-hold-97ff10c4dcd9 / L1-6|1|0|1|25.980/37.851|0.798/0.641|41.64|1.22|
|old_tradeoff-sustained-1f7c1c806ff0 / L1-8|0|0|0|—/—|0.688/0.783|4.14|0.82|
|old_tradeoff-hold-1f7c1c806ff0 / L1-8|0|0|0|—/—|1.478/0.770|21.23|1.51|
|old_tradeoff-sustained-3077ca809d70 / L1-1|1|0|1|138.744/138.744|2.116/1.855|63.63|1.71|
|old_tradeoff-hold-3077ca809d70 / L1-5|0|0|0|—/—|1.211/1.004|0.44|0.10|
|old_tradeoff-sustained-7952b43a7dd4 / L1-11|2|0|2|45.910/200.149|2.117/1.969|38.02|2.20|
|old_tradeoff-hold-7952b43a7dd4 / L1-6|0|0|0|—/—|1.320/0.563|2.88|0.50|
|old_tradeoff-sustained-6fc08121dd42 / L1-11|2|0|2|38.165/116.065|2.700/1.969|45.63|2.62|
|old_tradeoff-hold-6fc08121dd42 / L1-6|0|0|0|—/—|3.139/1.367|37.58|2.64|
|new_near-sustained-5f62e065b5ac / L1-7|0|0|0|—/—|0.324/0.382|4.26|0.72|
|new_near-hold-5f62e065b5ac / L1-12|0|0|0|—/—|0.565/0.733|4.05|0.81|
|new_stop-sustained-5f6380cdec75 / L1-8|0|0|0|—/—|0.636/0.796|4.09|0.82|
|new_stop-hold-5f6380cdec75 / L1-8|0|0|0|—/—|0.970/0.869|20.08|1.29|
|new_stop-sustained-4ea37e71ee89 / L1-11|4|1|3|45.367/420.950|2.568/1.905|49.58|1.47|
|new_stop-hold-4ea37e71ee89 / L1-8|0|0|0|—/—|2.001/1.001|22.74|1.93|
|P1-1 / L2-C-S|0|0|0|—/—|0.636/0.878|4.55|0.83|
|P1-1 / L2-C-P|0|0|0|—/—|0.887/0.866|6.21|1.22|
|P1-13 / L2-C-S|0|0|0|—/—|2.121/1.640|52.49|1.88|
|P1-13 / L2-C-P|0|0|0|—/—|1.592/1.434|52.49|1.54|
|new_tail-sustained-5c67db5a4802 / L2-C-S|2|1|1|18.184/94.743|1.567/1.919|41.25|2.08|
|new_tail-sustained-5c67db5a4802 / L2-C-P|1|0|1|21.873/27.749|1.403/1.257|42.63|1.46|
|new_tail-sustained-97ff10c4dcd9 / L2-C-S|1|0|1|54.556/119.616|2.477/2.692|60.26|3.21|
|new_tail-sustained-97ff10c4dcd9 / L2-C-P|1|0|1|109.811/109.811|2.291/1.631|52.66|1.61|
|new_tail-hold-97ff10c4dcd9 / L2-C-S|0|0|0|—/—|0.856/1.450|30.67|1.30|
|new_tail-hold-97ff10c4dcd9 / L2-C-P|2|0|2|29.388/47.824|1.624/1.556|41.64|1.55|
|new_near-sustained-5f62e065b5ac / L2-C-S|0|0|0|—/—|0.638/0.897|5.58|1.03|
|new_near-sustained-5f62e065b5ac / L2-C-P|0|0|0|—/—|0.498/0.494|6.13|1.02|
|new_stop-sustained-5f6380cdec75 / L2-C-S|0|0|0|—/—|1.914/2.304|37.03|0.99|
|new_stop-sustained-5f6380cdec75 / L2-C-P|0|0|0|—/—|0.536/0.506|4.78|0.69|
|new_stop-sustained-5f6380cdec75-continue / L2-C-S|0|0|0|—/—|2.219/1.722|33.44|1.57|
|new_stop-sustained-5f6380cdec75-continue / L2-C-P|2|0|2|23.194/100.433|2.348/1.575|61.59|2.00|

任务/slip/yaw窗口参考通过，不等于q_PD连续性完全良好。例如P1-13的合格BC500分支仍有约52.49°目标跳变；这不是实际关节在20ms内转过52.49°。实际q、dq、力矩、关节余量与支撑见机器数据。

## 原相对门与窗口坐标分开

原FRL-v2正式门为族中位误差不恶化>max(2cm,20%初始)、力不增>.75N、loaded slip峰不增>max(.20m/s,50%初始)、release yaw不恶化>10°。本次不是完整开发族/完整释放回合，不计算冒充正式晋级的总pass。下表仅报告同状态同窗口相对BC500的坐标；末端yaw不是完整释放yaw。

|L2窗口|误差变化cm|力变化N|slip变化m/s|对应旧slip容限m/s|末端abs yaw变化°|完整释放旧门|
|---|---:|---:|---:|---:|---:|---|
|P1-1|-0.217|-0.226|0.000|0.200|-0.31|未以短窗替代|
|P1-13|-0.419|-0.513|-0.006|0.200|1.74|未以短窗替代|
|new_tail-sustained-5c67db5a4802|-3.281|-1.668|-0.107|0.432|-10.01|未以短窗替代|
|new_tail-sustained-97ff10c4dcd9|-4.980|-2.503|0.096|0.249|-7.61|未以短窗替代|
|new_tail-hold-97ff10c4dcd9|-13.863|-7.182|0.583|0.200|19.03|未以短窗替代|
|new_near-sustained-5f62e065b5ac|-0.099|-0.060|-0.001|0.200|-4.84|未以短窗替代|
|new_stop-sustained-5f6380cdec75|-6.301|-3.005|-0.285|0.200|-0.74|未以短窗替代|
|new_stop-sustained-5f6380cdec75-continue|2.466|0.683|0.415|0.200|-17.09|未以短窗替代|

## 分段实际坐标

阶段由真实伙伴时序与active划分，仅用于评价，不输入控制器。无active时不评价旧anchor跟随；支撑与滑移仍记录。分段短于20ms时q_PD可能无控制边界，标NA不补造。

|状态/执行/阶段|秒|误差cm/力N|slip峰/p90 m/s|pelvis yaw起→末°|root ΔX cm|足部/XYZ等完整数据|
|---|---:|---|---|---|---:|---|
|P1-1 / L2-C-S / moving|0.55|3.755/2.979|0.006/0.005|-9.18→-10.97|1.208|f1_phase_metrics.json|
|P1-1 / L2-C-S / holding|2.46|2.221/1.809|0.004/0.003|-11.00→-11.64|0.222|f1_phase_metrics.json|
|P1-1 / L2-C-P / moving|0.55|3.443/2.439|0.006/0.006|-9.18→-11.21|0.173|f1_phase_metrics.json|
|P1-1 / L2-C-P / holding|2.46|2.031/1.685|0.006/0.005|-11.24→-11.33|1.611|f1_phase_metrics.json|
|P1-13 / L2-C-S / moving|1.79|9.452/7.160|0.318/0.023|-4.57→-21.30|40.291|f1_phase_metrics.json|
|P1-13 / L2-C-S / holding|1.22|1.726/1.396|0.009/0.009|-21.31→-17.82|-1.018|f1_phase_metrics.json|
|P1-13 / L2-C-P / moving|1.79|8.974/6.555|0.312/0.013|-4.57→-19.35|39.017|f1_phase_metrics.json|
|P1-13 / L2-C-P / holding|1.22|1.057/0.790|0.004/0.003|-19.32→-19.56|-1.279|f1_phase_metrics.json|
|new_tail-sustained-5c67db5a4802 / L2-C-S / moving|2.05|13.373/8.741|0.865/0.019|-4.99→-12.34|45.817|f1_phase_metrics.json|
|new_tail-sustained-5c67db5a4802 / L2-C-S / holding|0.95|10.202/8.181|0.004/0.004|-12.35→-12.02|1.622|f1_phase_metrics.json|
|new_tail-sustained-5c67db5a4802 / L2-C-P / moving|2.05|9.549/7.024|0.758/0.016|-4.99→-2.14|21.452|f1_phase_metrics.json|
|new_tail-sustained-5c67db5a4802 / L2-C-P / holding|0.95|8.313/6.624|0.003/0.003|-2.14→-2.01|0.232|f1_phase_metrics.json|
|new_tail-sustained-97ff10c4dcd9 / L2-C-S / moving|0.70|5.307/5.399|0.238/0.026|-11.32→-6.71|0.462|f1_phase_metrics.json|
|new_tail-sustained-97ff10c4dcd9 / L2-C-S / decelerating|0.55|15.181/10.000|0.499/0.041|-6.51→-5.52|6.798|f1_phase_metrics.json|
|new_tail-sustained-97ff10c4dcd9 / L2-C-S / holding|1.75|14.888/8.889|0.181/0.009|-5.59→-10.41|32.516|f1_phase_metrics.json|
|new_tail-sustained-97ff10c4dcd9 / L2-C-P / moving|0.70|5.153/5.212|0.166/0.019|-11.32→-7.66|-0.198|f1_phase_metrics.json|
|new_tail-sustained-97ff10c4dcd9 / L2-C-P / decelerating|0.55|15.346/10.000|0.595/0.027|-7.51→-3.02|8.070|f1_phase_metrics.json|
|new_tail-sustained-97ff10c4dcd9 / L2-C-P / holding|1.75|5.967/4.264|0.138/0.009|-2.99→2.81|6.501|f1_phase_metrics.json|
|new_tail-hold-97ff10c4dcd9 / L2-C-S / holding|2.21|17.046/9.210|0.179/0.007|-8.85→-11.95|23.084|f1_phase_metrics.json|
|new_tail-hold-97ff10c4dcd9 / L2-C-S / released|0.79|—/0.000|0.005/0.005|-11.95→-9.34|0.892|f1_phase_metrics.json|
|new_tail-hold-97ff10c4dcd9 / L2-C-P / holding|2.21|3.183/2.028|0.762/0.011|-8.86→-27.53|17.115|f1_phase_metrics.json|
|new_tail-hold-97ff10c4dcd9 / L2-C-P / released|0.79|—/0.000|0.004/0.004|-27.54→-28.37|-0.023|f1_phase_metrics.json|
|new_near-sustained-5f62e065b5ac / L2-C-S / moving|0.44|4.116/2.722|0.005/0.005|-9.44→-12.19|0.302|f1_phase_metrics.json|
|new_near-sustained-5f62e065b5ac / L2-C-S / holding|1.69|2.316/1.887|0.004/0.003|-12.17→-11.84|1.031|f1_phase_metrics.json|
|new_near-sustained-5f62e065b5ac / L2-C-S / released|0.88|—/0.000|0.003/0.003|-11.85→-12.12|0.177|f1_phase_metrics.json|
|new_near-sustained-5f62e065b5ac / L2-C-P / moving|0.44|4.065/2.624|0.005/0.005|-9.44→-12.23|0.321|f1_phase_metrics.json|
|new_near-sustained-5f62e065b5ac / L2-C-P / holding|1.69|2.188/1.840|0.005/0.003|-12.22→-7.34|-0.062|f1_phase_metrics.json|
|new_near-sustained-5f62e065b5ac / L2-C-P / released|0.88|—/0.000|0.001/0.001|-7.34→-7.28|-0.161|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75 / L2-C-S / moving|0.51|4.421/4.195|0.006/0.006|-7.18→-9.72|0.282|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75 / L2-C-S / decelerating|0.70|5.275/5.505|0.010/0.010|-9.71→-9.17|-0.470|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75 / L2-C-S / holding|1.79|15.674/9.620|0.292/0.016|-9.12→6.08|7.577|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75 / L2-C-P / moving|0.51|4.889/4.313|0.006/0.004|-7.17→-7.49|0.349|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75 / L2-C-P / decelerating|0.70|6.891/5.696|0.007/0.007|-7.45→-5.47|0.695|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75 / L2-C-P / holding|1.79|6.252/5.016|0.001/0.001|-5.47→-5.35|0.002|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75-continue / L2-C-S / moving|2.04|8.499/6.954|0.396/0.017|-7.18→-5.00|13.979|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75-continue / L2-C-S / holding|0.96|5.260/4.496|0.331/0.015|-5.02→-18.89|20.030|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75-continue / L2-C-P / moving|2.04|11.507/7.864|0.721/0.019|-7.17→1.24|18.030|f1_phase_metrics.json|
|new_stop-sustained-5f6380cdec75-continue / L2-C-P / holding|0.96|5.997/4.443|0.811/0.007|1.18→-1.80|5.004|f1_phase_metrics.json|
