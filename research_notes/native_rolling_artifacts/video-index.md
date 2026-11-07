# Issue50：原生来源与滚动接触反馈——3条关键视频

全部是保存的**真实MuJoCo qpos**，25fps、1×仿真速度，包含完整接触和释放恢复；不是reference、q_PD动画或搜索预测。共同0–3s站立预热省略，画面从约3s开始，接触onset5s。当前没有训练新网络：MPC是昂贵的特权物理搜索程序，每100ms动作约需23s计算。

## 建议先看02

`02_native_feedback_comparison.mp4`（12.8s）：同72cm、平均0.15m/s、相同standing初态、K80/D8/总10N。

|位置|实际控制器|全接触3D误差 / 力RMS|
|---|---|---|
|左上|冻结旧原生C32|40.446cm / 9.086N|
|右上|原生motion资格来源，按时进入/返回|25.195cm / 9.171N|
|左下|本轮原生goal/motion＋滚动反馈|7.417cm / 5.465N|
|右下|冻结旧I能力参照|10.265cm / 6.682N|

注意接触手、身体路径、脚步及侧向/竖直力。MPC不调用旧I/C0/T30 actor。q_PD峰值与真实关节变化分开评价，不能只凭画面宣布全部质量问题解决。

## 03：同一程序如何在需求改变后分开

`03_common_prefix_stop_continue.mp4`（12.8s）：上排伙伴继续，下排伙伴在7.8s开始.7s平滑减速；每排左为固定斜视、右为固定俯视。

7.8s前实际physics、BFM历史、规划输入和命令逐值一致，第一条不同高层命令8.1s。程序没有未来停止时刻。停止分支保持期2.074cm / 1.730N，完整接触7.605cm / 5.457N；仍保留约20°额外骨盆偏航这一限制。两支释放后都保留4s恢复，允许稳定保持合理手臂姿态，不强制回唯一站姿。较短回合结束后最后画面保持，并显示ENDED。

## 01：为什么原生步态不能直接当接触专家

`01_native_source_speeds.mp4`（16s）：左上/右上/左下为同一原生motion的平均.09/.15/.22m/s伙伴条件；右下是I在.15。均72cm，但持续时间不同；结束后仅保持末帧，不能按各格同时刻累计量判断谁更好。

原生motion三档全接触误差60.523 / 25.195 / 19.256cm，力仍约9N。保留侧漂、转偏和反向牵拉，不能简单说.214m/s步态“太快”或速度匹配就解决了问题。

## 图例与模型身份

青C＝接触点；红A＝anchor；黄箭头＝真实施力（同尺度0.065m/N）；蓝＝实际heading/root轨迹；绿＝名义目标heading。RELEASED后的旧anchor几何标为non-task，不计主动跟随失败。

- BFM SHA：33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
- MPC实际runtime：32b3502262399409dd9091b067c89ba619d20873；渲染/统计e0bc634fd36e124ea628660d20ea3b5e56b1f673。
- C32 SHA：39cd0db55fddb174de3f77104acbd8d7574fb3a5051f743e4ca39bb6b7d24ad0。
- I SHA：72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a。
- motion reference SHA：211a192ddf6e3d534a1ef273c22e9cf977510ccb235e1c88dc2f0b24e8d6f2c0，含已记录M10离线手臂/腰修正，经同权重goal推断；不是M10 Student。

服务器完整数据：stage `experiments/native-rolling-h100-20261107`，资格`runs/qualification-03`、实际反馈`runs/rolling-02`，每条输入轨迹均有独立SHA。当前只验证有限.15反馈与其停止分支，不宣称新Student/general或真机能力。

已在Mac核对远端SHA、时长、帧数、H264/YUV420与全片解码，并检查关键画面；本文件夹只保留3个MP4和本索引。

|文件|帧数|SHA256|
|---|---:|---|
|01_native_source_speeds.mp4|400|db07ab9deb5051ee7210bfd227622adc26643c819d60a87459ab908da78c98b3|
|02_native_feedback_comparison.mp4|320|5a45c1f1825ee2bbd0d143f029331d4ee3985ce4cbb0449f3e3d6d60cbe9d074|
|03_common_prefix_stop_continue.mp4|320|901d3f8f39450e77f9afe70e1b4d92f9de406abba9b740b7ab05c8bfb01e932e|
