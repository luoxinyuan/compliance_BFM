# 停止反馈与传感诊断：4条关键视频

全部为保存真实MuJoCo qpos的1×同步回放，25fps。每一行左侧固定世界斜视、右侧俯视；青色接触点、红色伙伴anchor、黄色实际力箭头、蓝色heading/路径。字幕的connected和anchor仅用于评价可视化，不输入Student。

从t=3s ready开始，省略相同的3s站立初始化；保留全部接触、停止、释放与恢复。不是目标动作或参考motion动画。本地完整解码、时长、非空、SHA和画面已核验。

|文件|上/中/下三行|该看什么|
|---|---|---|
|01_matched_updates_stop_coverage.mp4|原r0 / 原数据再训A505 / 增加停止数据B505|相同新晚停止；全接触cm/N为7.015/4.313、7.579/4.145、7.087/3.800。观察停止后超前与XYZ，不仅脚步。B不在所有条件胜过r0。|
|02_force_sensing_limits.mp4|B505理想力 / 同B505强观测压力 / 重训无力N505|相同新晚停止；7.087/3.800、10.348/4.250、12.223/4.531。压力只改输入，真实cap仍5N；底行确实删除力列并重训。|
|03_teacher_stopping_initial_vs_selected.mp4|Direct初始化 / PPO开发选中30批 / 末轮40批|已暴露开发晚停止；选模使用三个开发条件，不是挑视频。查看早中段与释放，不以一帧更近判断整段。|
|04_no_force_actual_cap_comparison.mp4|同一个N505，真实cap5 / 7.5 / 10N|12.223cm/4.531N、7.629/4.443、8.048/6.024。7.5N明显改善，10N不继续改善且力更大；没有重训，不能认定感知机制已证实。|

01/02/04时序：onset4.08s，9.45s开始减速，10s伙伴停止，14.35s释放，20.4s结束。03为开发晚停止：onset4.25s，9.2s减速，10.05s停止，13.85s释放，19.9s结束。

模型SHA/全部seed/条件/负结果见Issue34和源码报告。没有把两个训练seed当独立硬件或任务统计。N505是预先固定的视频种子，不是事后挑最好；N506也记录在报告中。

## Run与身份

服务器共同根目录：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/stop-sensing-20261103`。
01源A-prepare-03/job；02源A-prepare-03和C-final-05；03源B1-train-02/job/dev-0000、0030、0040；04源F-final-07/job。

- `01_matched_updates_stop_coverage.mp4`：17.40s，SHA256 `a1db4d4194d48faf6f1f709ce1b7934fab0e04825cc6298df2a1e63145168a2a`
- `02_force_sensing_limits.mp4`：17.40s，SHA256 `8193a877f3d61b2b3d18b9ae1687265f261e8539684262d608d38466fd12b56b`
- `03_teacher_stopping_initial_vs_selected.mp4`：16.88s，SHA256 `b9a474f8b2d83cc7d62e098d1808a7062250df50d719b3605ae82563eb2d269e`
- `04_no_force_actual_cap_comparison.mp4`：17.40s，SHA256 `bbf4ad4a8a80aab26e32c5b328013a5680485d5ad7622785391a1e3cd6e7d331`

关键模型：
- A505: `c908575f558c5b6a67368a90f468bdfe29cc0aad2e469d999205f75998634142`
- B505: `78c818478b059cfd8aeec868918c275b9429a3863aac53dd1eb84e08dd62d4ff`
- N505: `038c218829f85ce0b661e38397c19121d5e49ec3ac38fde8c483d87f151893d6`
- stopTeacher_best: `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`
- stopTeacher_last: `fcda12ee84fb6564c0602ab20671aecf8889951e38fedb6676f33772fbd928df`
- 原r0: `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`
- Direct初始化: `a86ff5f9ca71ed2ae8dc95ceeab0ac980f82ab6ed8d3759c039eb03a62efe488`
