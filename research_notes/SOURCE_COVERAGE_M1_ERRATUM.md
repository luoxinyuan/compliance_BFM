# M1独立勘误01：源分层筛选载荷单位

原runtime 2c649e546e4baabc56dd783f7af56dc1165781eb，私有runs/standing保留。新脚本将e.last_support布尔量与20N比较，误报standing无可用源；该运行0查询/0标签/0训练，不影响既往RETENTION或#53。

修复只读取真实physics row的foot_normal_force_n，保持原预定20N门槛及速度/腕部条件。原物理/BFM/PD/reward不改。归档无接触真实行验证见load-unit-regression.json；新运行用standing-r1，不覆盖失败目录。
