# F1状态与信息覆盖（L0–L2完成）

锁定清单见[f1_state_manifest.json](f1_state_manifest.json)，文件锁见[f1_state_manifest_lock.json](f1_state_manifest_lock.json)。原source plan hash未变：12主父组24状态；3原生source参考各1状态，共27状态/15个登记ID。3原生参考不作为独立随机重复，也不计接触能力成功分母。全部已实际重建、保存、读回及三次有效状态恢复校验。旧sealed不读。

角色选择沿用首L0预登记：P1两个严格持续正例及同父组后续相位；新dev三个指定mid/stop负例；旧reg已曝光退化的2个stop和2个mid；另1near、2stop父组。未按本轮结果换父组或挑好状态。实际快照时间单列，区分日志中的执行前obs与执行后decision时间。

三原生参考由原始z与原生初态真实执行到6s；qpos/qvel/force/support前缀逐值核对旧存档。只确认这些恢复资产，不以此宣布前进motion的滑移/yaw局限已消失。5个stop/continue父组存档前缀分别500/400/740/320/1020个5ms tick一致，兄弟分支保持同一父组身份。

数组干预是3个状态上156条相关分支，绝不是156个独立父组。2个真实伙伴受力状态与1个无伙伴接触状态各52条3秒干预；逐字段与联合扰动均有未扰动参考。无伙伴力状态仍含真实地面支撑约束。

额外别名父组1、相关分支2，只预登记，不构造、不入能力统计。该句原指L0报告时点；当前L1/L2实际覆盖见下节，L3仍为0。L0另测三次2候选1秒→执行首100ms的端到端开销，及三个状态1s/3s各两次严格重复；均标O-F计时控制，不当能力搜索结果。

权限见[f1_info_access_audit.json](f1_info_access_audit.json)。C-S固定BC500可作受限对照；**C-S planner缺格**；L0时尚无新C-P资格；L2现已实现当前状态常速预测并完成8次未来spec干预和真实共同前缀核验。不能用精确未来Task程序补出C-S/C-P成功。


## L1/L2最终覆盖

L1使用锁定主清单全部24状态/12父组，每状态16个连续3s控制，共384分支。保留所有成功、失败及各自runtime；两次恢复错误与剩余状态续跑见F1_ERRATUM。16候选的精确节点、phase定位和来源见F1_L1_L2_METHODS及f1_l1_l2_protocol.json，不按成功结果换池。三原生参考仅供源/恢复资格，不计接触成功率。

L2固定使用P1-1、P1-13、tail-sustained-5c67、tail-sustained-97ff、tail-hold-97ff、near-sustained-5f62、stop-sustained-5f63，最后一项加同前缀continue分支：六父组、七初态、八窗口。每窗3s，固定8候选（原16池索引0/2/4/5/6/7/9/10）、2s预测，每100ms执行首动作后重规划，240查询。C-S为冻结BC500真实执行，C-P为因果当前特权搜索；均独立恢复同一完整初态。L1为真实未来程序O-F，不能与L2作单一信息变量的因果比较。

别名组未构造，C-S planner缺格；没有借真实隐藏状态模拟器冒充受限预测器。真实stop/continue分叉前100个5ms tick、5个高层命令逐值一致；分叉后真实force/anchor于5.505s起不同，C-S受限输入与命令于5.6s开始不同。兄弟分支只计同一父组。权限明细见f1_info_access_audit.jsonl，实际首命令/首BFM子步240次对齐见f1_query_alignment.json。

旧state manifest及source plan内容/hash完全不改，未读封存。L1/L2快照、proposal全序列、预测/真实执行、所有失败保留于独立stage；L3未执行，3h未挪用。
