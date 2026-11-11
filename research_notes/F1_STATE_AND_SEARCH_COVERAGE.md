# F1状态与信息覆盖（L0停止快照）

锁定清单见[f1_state_manifest.json](f1_state_manifest.json)：12主父组24状态；3原生source参考各1状态，规划15父组27状态。仅首状态实际恢复；null snapshot SHA表示未生成，不能称完整恢复资产。旧sealed不读，均是暴露诊断或source-positive。

选择按角色预固定：P1严格持续正例及同父组后续相位；新dev三个指定mid/stop负例；旧reg按已曝光误差退化排序的2个stop和2个mid；另1near、2stop父组。sustained取onset到需求结束的55%，hold取结束加deceleration后.35s，匹配实际高层决策边界；记录的decision时间是执行后时间，实际快照时间单列，不能混淆。共同前缀父组保留配对身份，尚未执行本轮全前缀核对。

原生来源已有无接触真实轨迹已索引及哈希，L0没有再跑新motion扫描，forward的滑移/yaw局限保留。一次source正控制不算接触walking资格。

别名单独额外1组2支，预登记但未构造；不入任何能力成功率。L1序列候选数、L2重规划数、L3次数均为0。L0仅同一冻结策略的2条重复1秒后果；拟议2候选100ms测速未执行。

权限矩阵见[f1_info_access_audit.json](f1_info_access_audit.json)。当前没有完整新搜索查询，不伪造query ledger；未来选择器字段必须全部记录才可分级。
