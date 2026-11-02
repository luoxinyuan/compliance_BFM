# Direct Teacher → Student：视频索引

2026-11-03。以下均为保存的真实 MuJoCo qpos，1×速度、25fps；每行左侧固定斜视、右侧俯视，显示实际力、接触点、anchor和heading。从ready t=3s展示至完整恢复，省略此前初始化，不是参考motion或q_PD动画。

|文件|从上至下的控制器与条件|时长|主要观察|
|---|---|---:|---|
|01_teacher_student_new_slow.mp4|Direct初始化Teacher / 修正版初始化Teacher Student；.805m、ramp8.28s、onset4.62s|17.92s|误差/力7.992cm/4.301N vs7.115cm/3.742N；修正版为已暴露条件复测，不是新盲测|
|02_ppo_generalization_failure.mp4|M10 / Direct初始化 / 新B1第10批；.855m、7.62s、4.02s|16.68s|7.472/3.734、7.455/4.055、8.111/4.284；PPO未保住新快条件|
|03_stop_transfer_and_failure.mp4|旧PPO15 Teacher / 其纯模仿Student / 修正版初始化Teacher Student；.38m、3.8s、4.35s|13.20s|9.375/3.784、9.409/3.728、16.263/4.435；短行程也改变启动加速度，不能单归因于停止|
|04_same_student_same_prefix_stop.mp4|同一PPO15 Student：正常82cm牵引 / t8s开始.6s真实伙伴减速停止|17.08s|8s前1001物理点完全一致；停止后8.10s改变z，root最终X .812→.503m；仍有约4.1cm保持超前|

Student只使用本体、真实动作/力历史、已知腕部位置及自身命令，属于ideal-force-observation Student，尚非真实G1力传感验证。所有误差/力为全接触3D RMS cm / N。

## 身份与数据

- Teacher初始化 SHA256：a86ff5f9ca71ed2ae8dc95ceeab0ac980f82ab6ed8d3759c039eb03a62efe488
- 旧PPO15 Teacher：92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe
- PPO15 Student（纯4000次监督更新）：0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12
- 修正版初始化Teacher Student：8ff5a5435702738b39df91340f8278720e7f850cab97d5c5479bb1e12d5738ec
- B1第10批：d18529b96b8343ac6bb3d0484de5fb8f29934bed28af71ddec4709dcfffa6bd0
- 服务器数据根：/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/direct-transfer-20261103
- 01/03：A-final-04/job；02：B1-final-05/job/evaluation及冻结基线；04：A-prepare-01/job原轨迹与A-final-07/job停止分支。
- 精确原始轨迹文件、SHA与渲染参数：服务器 videos/manifest.json、videos-stop/manifest.json。
- 完整报告：https://github.com/luoxinyuan/compliance_BFM/blob/gaoe05/research/research_notes/TRANSFER_RESULTS.md

全部文件已与服务器SHA核对，并通过完整解码和时长检查。保留代表性失败，不按新seed挑视频。
