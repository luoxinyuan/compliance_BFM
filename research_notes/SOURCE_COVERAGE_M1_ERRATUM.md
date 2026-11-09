# M1独立勘误01：源分层筛选载荷单位

原runtime 2c649e546e4baabc56dd783f7af56dc1165781eb，私有runs/standing保留。新脚本将e.last_support布尔量与20N比较，误报standing无可用源；该运行0查询/0标签/0训练，不影响既往RETENTION或#53。

修复只读取真实physics row的foot_normal_force_n，保持原预定20N门槛及速度/腕部条件。原物理/BFM/PD/reward不改。归档无接触真实行验证见load-unit-regression.json；新运行用standing-r1，不覆盖失败目录。

## 采集准备补充：有载荷滑移口径
旧contact_tangent_speed_max_mps包含所有足地接触，未按法向载荷过滤，不能直接称有载荷峰值。三个准备进程在尚无Teacher查询时停止，目录保留；本轮只新增只读逐接触法向力与切向速度记录，原reward仍使用原指标，不改物理。门槛用真实载荷>1e-6N的足地接触，旧指标并存。无接触高度边界沿原.45m，不新增.65m限制。此前为准备阶段，不输出科学判读。
