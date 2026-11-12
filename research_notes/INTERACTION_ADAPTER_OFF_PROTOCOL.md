# Issue62 B19 adapter-off 冻结反事实协议

2026-11-12，Yi明确授权“做 adapter-off 对照，结果出来再定配方”。本轮不训练、不改三臂共同合同、不启动C/dev。

从B19 round19完整checkpoint（core SHA 0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984）恢复真实11.8s状态，保留全部物理/solver/历史/上一z/RNG及所有权重。父组8dde57eac4325d7701fafe97d48f57cb962cd8ff121a9e4770a1bbdffa10db4e，release10.8s。

唯一干预：执行时不加29D adapter输出；仍由原B19的U生成合法z，原BFM和PD执行。权重与arm身份不变。分别执行offsample（一次原std.01/clip.06噪声）和offmean；对照已归档逐值复现的sample/mean。初始模型、RNG、物理状态、首z验证一致，后续闭环历史各自真实维护。

每分支最多3s/30次高层调用，到原hard boundary立即停止，不跨episode重置。对共同0.51s窗口及各自实际完整窗口分别报告。已释放，不将旧anchor误差作跟随失败或改善依据。重点真实关节范围、q_PD、支撑/滑移、root/yaw/残余速度及稳定性。

若off不越界，只证明此B19释放状态中的执行adapter有因果贡献，不证明删除adapter使完整任务/训练安全，更不自动修改配方。若off也越界，记录首次失效和实际范围，不追加扫描。

继承累计31066.150840760278 job秒；准备/分析/发布保守预留1200秒，两个GPUjob合计上限1800秒，全部计入15h/8GiB pilot和项目240h账。GPU0串行，shared free≥100GiB。输出/失败/初始化/IO全部记账，sealed不读，所有历史保留。
