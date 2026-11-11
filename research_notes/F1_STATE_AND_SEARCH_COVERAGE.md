# F1状态与信息覆盖（L0续验完成）

锁定清单见[f1_state_manifest.json](f1_state_manifest.json)，文件锁见[f1_state_manifest_lock.json](f1_state_manifest_lock.json)。原source plan hash未变：12主父组24状态；3原生source参考各1状态，共27状态/15个登记ID。3原生参考不作为独立随机重复，也不计接触能力成功分母。全部已实际重建、保存、读回及三次有效状态恢复校验。旧sealed不读。

角色选择沿用首L0预登记：P1两个严格持续正例及同父组后续相位；新dev三个指定mid/stop负例；旧reg已曝光退化的2个stop和2个mid；另1near、2stop父组。未按本轮结果换父组或挑好状态。实际快照时间单列，区分日志中的执行前obs与执行后decision时间。

三原生参考由原始z与原生初态真实执行到6s；qpos/qvel/force/support前缀逐值核对旧存档。只确认这些恢复资产，不以此宣布前进motion的滑移/yaw局限已消失。5个stop/continue父组存档前缀分别500/400/740/320/1020个5ms tick一致，兄弟分支保持同一父组身份。

数组干预是3个状态上156条相关分支，绝不是156个独立父组。2个真实伙伴受力状态与1个无伙伴接触状态各52条3秒干预；逐字段与联合扰动均有未扰动参考。无伙伴力状态仍含真实地面支撑约束。

额外别名父组1、相关分支2，只预登记，不构造、不入能力统计。L1/L2/L3为0。L0另测三次2候选1秒→执行首100ms的端到端开销，及三个状态1s/3s各两次严格重复；均标O-F计时控制，不当能力搜索结果。

权限见[f1_info_access_audit.json](f1_info_access_audit.json)。C-S固定BC500可作受限对照；**C-S planner缺格**；旧C-P因果预测模块未在本L0完成新F1实现/资格。不能用精确未来Task程序补出C-S/C-P成功。
