# BFM-CTFT v1 正式 U 硬停记录

2026-11-18，Issue #67。正式 runtime `4f1ce5d14f8166ce2aa2c660caa67da5b70f675f`，共同目标 131072 条真实 50Hz transitions/臂。U 已停，T 未启动。完整终点对比尚未开展。

| 项目 | 实际记录 |
|---|---:|
| U 已认证提交 | 17408 transitions、17 轮 |
| 完整更新 | 136 组，其中 72 组含 Actor |
| 最近的完整包 | round-17408，manifest SHA df27aad380f2b84504fb9369161ead8a8a8c2f9242a56978ba0affd5e012bcdd |
| 此后已完成物理行 | 605 条，未进入提交或更新；另有触发停机的未完成控制 |
| U 全流程实际 5ms ticks | 73256 |
| 配对执行器墙钟 | 1775.634s，stage 保守累计 36054.351s |
| 16k 复核 | 5.789h/臂、234.176GiB 全阶段保守投影，均在合同上限内 |
| 最近八轮联合更新中位 | 122.921s/轮，其中 save+verify 90.041s、优化 2.734s；CUDA peak reserved 12.584GiB |

## 首次硬停

第 33 个回合 `native-family-0001-0`，g0 原生动作继续，无伙伴施力。源为 `walk3_subject5` case10/control171，在源时间 6.42s 接管；真实控制执行 182 个完整 20ms 命令后，于物理时间 10.075s 的下一命令中触发。真实左踝 roll q=-0.2830281 rad，下限 -0.2618 rad，超限 0.0212281 rad，大于硬门 0.020 rad；dq=-1.36596 rad/s。腕力 0N，足底法向约137.08/215.23N，双脚支撑，躯干倾角25.17°、身高0.6468m、无非期望地面接触。根据 runtime 中 `m10_physics.py:79–91` 的同一 MuJoCo 关节范围与 datastate 重算。这不属于表示层容差问题。

归因限度：同一原生源的既有完整轨迹在源时间 10.075s 机械超限为0。它是不同动作及历史下的参考，单个配对时刻无法区分随机动作、动作后的累积状态变化和学习更新的贡献，也不能评价 U/T 对照胜负。伙伴没有施力是直接观测。

## 存档与状态

`formal-pair-01/ledger.json` 为 STOPPED_ARM_FAILURE_NO_AUTOMATIC_NEXT_ARM，子进程退出码1；`formal-U-01/stopped.json` 保存账号边界。最近的 round-17408 与独立 failure-store 包均实际读回；失败包含未提交状态，state SHA 59f5d27b45466d266223131dd80a0097fd1810b057201a443c4365e8b1ab4155。完整控制的轨迹分段到 10.060s，触发帧以失败 Task 的 mjdata 和接触数据定位。加载器的通用 result.json 仍显示初始 running 占位，以执行器、退出码、停机记录和原子 latest 为准。

复核只读取既有失败包和源轨迹，未新增物理或更新。已固定 U 停机与 T 未启动，合同物理门仍适用。
