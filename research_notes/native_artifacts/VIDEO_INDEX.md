# Issue48 — Native capability contact learning videos

仅4条关键MP4，真实qpos/1×/25fps；拼入同一实际回合保存的真实源前缀，非reference/q_PD动画。
01：相同新6.8cm需求，左上I、右上C0、左下原生引导raw-head末轮32、右下原生引导coordinate-head末轮32。
02：相同新69cm需求，同样四个模型；保留所有失败，未按任务换新模型。普通探索全矩阵保留在报告，视频固定展示两种同协方差的均值head。
03：同一coordinate固定末轮32checkpoint（开发未选中，不改原选择），共同前缀8.27s后真实伙伴停止/继续。每行固定世界视角＋俯视。
04：已连接零需求，左上换为原生Standing，其他三项保持同一冻结模型，展示无请求动作与牵拉；不同于关闭真实接触。
黄箭头实际力、红点anchor、青点接触点、蓝线root轨迹；release后旧anchor误差是非任务几何描述。
两条新策略均为特权Teacher，不是新Student或真机感知验证。
本版本替换早期开发best可视化：三组best都为初始化，无法展示PPO学到的变化。这里统一用预先保存的末轮32，不按条件挑模型，不修改开发选模，早期视频在服务器/private分析目录完整保留。
模型源：新策略fresh高层网络，原生goal/motion continuation初始化＋真实PPO；I/C0仅冻结对照，不参与新策略初始化。

## 模型身份（固定末轮用于学习变化诊断，不按封存重选）
- guided_last: SHA256 7195af0f637f310f8db40c6400f06a93fd643421b08fce9d8550883414c01e52; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-contact-20261106/ppo-guided/last.pt
- coordinate_last: SHA256 39cd0db55fddb174de3f77104acbd8d7574fb3a5051f743e4ca39bb6b7d24ad0; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-contact-20261106/ppo-coordinate/last.pt
- I: SHA256 72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m39a-demand-20261104/B-ppo-03/job/best.pt
- C0: SHA256 ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/prior-recovery-20261105/correct-C0/C0.pt

较短分支完整结束后保持末帧以同步展示；非慢放。若开发选择iteration0，它就是原生初始化，并不表示PPO胜出。
视频中的B32/C32使用相同探索协方差，仅训练均值head坐标不同。未展示的A普通组与B引导组同时改变方向和总方差，因此A/B不能单独归因方向性。

## 视频与实际全接触指标
- 01-new_near.mp4: 10.08s/252frames; SHA256 1dc44ca1a629b2d21096e308fec7816c4fcaab085cf9a01e89d199ca1eeef8b9
- 02-new_long.mp4: 16.48s/412frames; SHA256 5d552915c24340844c41a7da947ee02a2126eefb97f2dee82ad27e3136fe134f
- 04-zero_connected.mp4: 12.20s/305frames; SHA256 58e02b060b3d44952ac9f6f9ad93c239afecfaada8aafc020c61b2668de63df9
- 03-guided-common-prefix-stop-continue.mp4: 16.48s/412frames; SHA256 eb5b83d29cfb9536d2e683f4261c7218ee8c144249c7bc2ac198e83a1e7e9adf
- guided_last__new_near: 3.277cm / 2.666N; boundary=None
- guided_last__new_long: 36.210cm / 8.856N; boundary=None
- guided_last__new_stop: 9.663cm / 7.624N; boundary=None
- guided_last__new_continue: 36.210cm / 8.856N; boundary=None
- guided_last__zero_connected: 19.447cm / 6.210N; boundary=None
- coordinate_last__new_near: 2.823cm / 2.300N; boundary=None
- coordinate_last__new_long: 34.920cm / 8.756N; boundary=None
- coordinate_last__new_stop: 8.551cm / 6.936N; boundary=None
- coordinate_last__new_continue: 34.920cm / 8.756N; boundary=None
- coordinate_last__zero_connected: 0.985cm / 0.812N; boundary=None
- I__new_near: 3.492cm / 2.909N; boundary=None
- I__new_long: 5.904cm / 4.853N; boundary=None
- I__new_stop: 5.467cm / 4.511N; boundary=None
- I__new_continue: 5.904cm / 4.853N; boundary=None
- I__zero_connected: 0.510cm / 0.415N; boundary=None
- C0__new_near: 2.245cm / 1.870N; boundary=None
- C0__new_long: 24.857cm / 7.949N; boundary=None
- C0__new_stop: 8.387cm / 6.843N; boundary=None
- C0__new_continue: 24.857cm / 7.949N; boundary=None
- C0__zero_connected: 0.532cm / 0.430N; boundary=None
- native_standing__new_near: 3.804cm / 3.097N; boundary=None
- native_standing__new_long: 41.715cm / 8.952N; boundary=None
- native_standing__zero_connected: 0.462cm / 0.376N; boundary=None
