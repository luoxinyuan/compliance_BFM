# Issue62 Pilot v3：用户接纳共同探索合同

2026-11-12，Yi明确接受全29维action std=.01、clip=.06为A/B/C共同新合同，授权A从共同初始化重跑，B/C同合同。见Issue62 comment6822542264，机器锁为interaction_pilot_lock_v3_approved.json。

## 历史与判断边界

全维降噪的单窗峰值筛查仍记未通过，不追改为通过。用户接纳理由：该门是保守单窗筛查、非§5.1判据，实际关节越界消除、误差/力及滑移分布改善；残余尾部由训练后dev按原科学/风控口径裁决。不是调高法向阈值、改变滑移定义或放宽真实关节0.02rad边界。

Pilot首读仍为单seed、dev方向性及B_I探针；按原§5.1指标报告不等于当前独立组数已满足正式结论。sealed不执行，不能选左右/不同任务checkpoint拼成功。

## 共同实施合同

- A/B/C均从原seed6201共同网络模板及standing原生初始化开始；fresh optimizers/replay。不是BC500初始化，也不是旧A65续训。BC500仅冻结外部评价参考。
- 每臂32000真实50Hz transitions，128×250、每轮2完整agent.update、batch32。网络/loss/reward/任务/父组/控制节拍/BFM/PD均不改。
- 仅实际29D action采样std=.01、clip=.06；一次Gaussian draw后原[-1,1]执行映射。实际执行action进入replay，mean执行不加噪声。此为当前off-policy交互FB Pilot，不混称高层Gaussian PPO。
- exploration参数随每轮完整checkpoint保存；旧包无该字段时明确解释为旧.05/.3。恢复时锁与checkpoint必须相符，不能加载旧A65后静默切噪声续训。
- 初始化测试实际sample逐值等于声明的一次抽样；RNG恢复，不消耗训练随机序列。旧均值、真实历史、权限测试不改。
- 每轮完整不可覆盖checkpoint，监督器异常回收owned进程；单卡A→B→C→独立dev三臂/initial/BC500→CPU B_I首读。物理或正确性失败保存停止，不跳过失败父组。

## 预算与IO

Pilot累计上限54000秒/15h、8GiB输出，项目240h不重置；承接22156.050969840097s保守上界，另保守预记1200s准备/代码检查/发布。所有新job、失败、评价、存盘/恢复进入sidecar；历史输出PREP/旧Pilot/repair/noise及新stage聚合计数，inode去重。保留原数据，无自动删除。

剩余可用约8.51job小时；按原实测约8–10 transitions/s热态及128轮存盘，三臂约3–5h，独立dev及冷启动另留约2h，具体以实测及护栏为准。输出按每轮实测增长，8GiB可能先于时间触发停止，不能擅自扩输出。共享自由空间≥100GiB，单GPU0重任务串行，其他用户进程不触碰。

新stage interaction-pilot-v3-20261112；旧stage与源SHA保持。发车前本文件仅是实施与授权记录，实际状态和计数以新jobs/ledger及CURRENT_STATE为准，不宣称已获得行为收益。
