# Issue46: real-feedback coexistence videos

仅3条关键视频。真实保存qpos、1×/25fps；不是reference或q_PD动画。未按任务拼接策略。

01：新6cm。左上I、右上C0、左下RL4601末轮、右下RL4602末轮。
02：新72cm，同样四个冻结模型；保留C0长距失败。
03：同一个RL4601末轮，共同物理前缀到7.37s后，上方伙伴停止、下方继续。每行左侧固定世界视角、右侧俯视。

两个RL末轮是预定负结果对照，不是开发选定新基线；两seed开发均选择初始化I。新测试后没有补训或重选。
完整接触与释放恢复保留。释放后的旧anchor几何误差只是描述，不是任务误差。黄箭头真实力、红点anchor、青点实际接触点、蓝线root路径。

模型身份：
- I: 72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m39a-demand-20261104/B-ppo-03/job/best.pt
- seed4601_best: a04001b13457e73664057b0d13efb044130fd5fee2bd914efa06344c5ee5208f; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105/ppo4601/best.pt
- seed4601_last: 8fa4759d3766568dee8fbba0367e772a97977694d19a503868708efa6d8c9228; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105/ppo4601/last.pt
- seed4602_best: 92bc6ba6c42a6ef8c3c7ba7cea8cd04bc947b171e04f9d843063ec998a0ba3ee; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105/ppo4602/best.pt
- seed4602_last: e9a1972166b25e3b915b8d75e97d3377e540583e215d91f269e8dba4550310c9; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105/ppo4602/last.pt
- C0: ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/prior-recovery-20261105/correct-C0/C0.pt
- F4501: 52956080217e6346b1a8b196ad411cbc84a07e248a8e747eeedf2dcc775e12f6; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105/F4501/F4501-2000.pt
- R4501: 298af356d386f79377ca103b4fc82baecbf6e18a997a958633dade2cc7733a37; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105/R4501/R4501-2000.pt
- 01-holdout_local6.mp4: 8.16s; SHA256 0ee73d1aadb67a5a744a011cb44c772838b731d2cda2eb8072fd164702a57fb5
- 02-holdout_long72.mp4: 14.96s; SHA256 cbd8314035b6ffb0ce90d63391d79e2fd441d96bcddf59bb68a2d84ea914839d
- 03-common-prefix-stop-continue.mp4: 15.00s; SHA256 00b16ef5640b0ae055268a00fab576e841be35003b1621772385ba45abc8bb4c

视频条件实际全接触指标（cm/N）：
- seed4601_last__holdout_local6: 4.190cm / 3.492N; boundary=None
- seed4601_last__holdout_long72: 5.478cm / 4.511N; boundary=None
- seed4601_last__holdout_stop: 4.956cm / 4.038N; boundary=None
- seed4601_last__holdout_continue: 5.468cm / 4.501N; boundary=None
- seed4602_last__holdout_local6: 4.790cm / 3.990N; boundary=None
- seed4602_last__holdout_long72: 6.290cm / 5.179N; boundary=None
- I__holdout_local6: 4.452cm / 3.726N; boundary=None
- I__holdout_long72: 7.632cm / 5.877N; boundary=None
- C0__holdout_local6: 2.293cm / 1.895N; boundary=None
- C0__holdout_long72: 30.886cm / 8.625N; boundary=None

完整结果：Issue46 / COEXIST_RESULTS.md。所有模型是特权Teacher；不是新增Student或实机验证。

补充读图：视频从真实高层接管时刻t=4s起，包含全部接触前等待、移动、保持及释放恢复；此前预稳定/源前缀另存服务器。03画面“research negative”表示未被开发规则选中的研究对照，不是说继续分支没有收益。seed4601在72cm明显改善，但停止和局部动作质量有取舍。6cm时它减少脚步却伴随root下沉约6.8cm；C0才是更好的局部伸臂参照，C0的长距失败同样完整显示。
