# 本轮仅保留两条关键视频

都是保存的真实MuJoCo qpos按1×、25fps重放；省去0–3s共同warmup，完整接触、停止和4s释放恢复保留。不是Teacher搜索预测、参考motion或q_PD动画。

1. **01-clean-label-increment-short-and-long.mp4**（12.8s）：左列冻结干净BC500，右列本轮34条正确标签得到的S-interactive500；上排6cm小请求，下排59cm持续牵引。同一模型分别完成两条件。看小请求的改善与长距三维受力增加，不能只看走没走。
2. **02-student-RL-transient-gain-and-regression.mp4**（12.8s）：同一个已暴露72cm开发条件，上排冻结BC500，中排seed5401第4批，下排第7批；每排左侧固定世界斜视、右侧俯视。它展示中期收益与后续退化，不代表新封存成功；第二seed已按用户要求停在第4批，未拼入此视频。

画面A=实际anchor，C=接触点，黄箭头=实际施力，cyan=root轨迹；XYZ力是该帧值，不是回合RMS。RELEASE后旧anchor几何差仅用于描述，正式跟随指标只在真实连接期间计算。上排短回合结束时标记并保持末帧，与长回合对齐，不加速播放。

主要指标（误差cm/力N）：6cm BC3.200/2.671→交互2.899/2.399；59cm继续10.794/7.201→11.689/8.400。视频02开发72cm BC11.316/7.485→batch4 8.303/5.403→batch7 18.494/8.922。

模型SHA、实际run、协议和完整结果：[Issue53续跑](https://github.com/GaoE05/robot-research-bridge/issues/53)；源码research_notes/INTERACTIVE_INCREMENT_RESULTS.md。所有模型独立保留，本轮不替换BC500、不再训练。
