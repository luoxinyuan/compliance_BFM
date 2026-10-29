单场景 Contact Projector：已完成监督训练 → 冻结 BFM → 完整6秒真实物理闭环。

**结果：Student在选定的 +X 3.5cm 开发场景中基本复现了教师收益。** 保持阶段3D误差 RMS：standing BFM **12.920mm**、教师 **7.596mm**、Student **7.502mm**；Student较baseline降低41.93%，恢复约101.8%的教师误差改善。Student与教师仅差0.094mm，不据此宣称优于教师。4个6秒回合均完成、未跌倒。这是刻意训练同一条轨迹的工程/物理闭环验证，不是泛化或held-out成绩。

固定条件：原standing skill、右腕+X3.5cm@1.1s、seed0、K80/D8、cap5N、ramp2s、release4.5s。继承 #21@03ef478（其根为 #19@bb08338），未合并PPO/mask/noise。原BFM、normalizer、Tether、ComplianceEnv、50Hz BFM/200Hz PD/physics和初始状态均保留。

数据与训练：补录同一教师6秒轨迹的全部50Hz观测，得到60个10Hz决策、300个真实29D动作/q_PD目标。baseline与旧数据完全重现；补录教师的qpos/qvel/force/wrist/root/torque最大差均0。每个10Hz raw latent同时对齐随后五帧不同的真实50Hz观测/目标，释放gate按50Hz更新。

Student是**474→128 tanh→128 tanh→256的MLP（110,336参数）**，只输入冻结归一化的`state/last_action/history_actor`和最近3帧实测XYZ力/20。没有anchor、世界腕位置、Jacobian、未来轨迹、时钟、教师残差或latent标签；也没有teacher-forcing previous latent。输出`tanh→standing切空间→residual_scale=.3→原球面norm16→冻结BFM→原29D action/PD`，不直接加右臂动作。教师仍可以用特权生成目标。

固定Adam lr1e−3，2000次更新，每批12/60决策、seed0、grad clip1；loss=右臂q_PD MSE+0.1非右臂MSE+1e−6 tanh(raw)²，active/release/pre权重5/2/1。使用第2000步固定checkpoint，无挑选。另做8时刻×2门控×200步的单latent可行性诊断，共3200步；它们不是学生更新，也没有PPO。

| 同一教师观测上的离线对齐 | 右臂q_PD RMS：训练前→后 rad | 非右臂q_PD RMS：训练后 rad | 全29D action RMS：前→后 |
|---|---:|---:|---:|
| 移动 | 0.010813 → 0.000347 | 0.000137 | 0.003222 → 0.000365 |
| 保持 | 0.015430 → 0.000156 | 0.000156 | 0.004545 → 0.000445 |
| 释放 | 0.005199 → 0.000095 | 0.000056 | 0.001505 → 0.000151 |

总加权loss 1.38313e−4→7.91213e−8。首步梯度norm1.936e−4且所有记录有限；原BFM参数无梯度/更新，直接可微actor与public inference差0。离线latent norm误差≤1.91e−6、最大偏角2.243°（上限atan(.3)=16.699°）、切向点积误差≤4.77e−7、tanh幅度≤.3323；真实闭环最大偏角2.151°。

**第一个实质接口边界是原力幅门控与教师释放不兼容。** `force_norm/5`在本例保持阶段只有约.106–.133，释放后立即为0，但教师还有0.2秒因果fade。固定200步投影在3.1s的右臂RMS残差为.003093rad，4.6s为.012969rad且gate0时无改善；不能把它解释成Student学习失败。

本次提前固定采用基于实测力的接触gate：10Hz检测`norm(force)>1e−6N`，活动时1、观测到释放后保持同一个0.2s线性fade并在50Hz执行。它延续 #20 active gate1与#21因果释放的接口，不读取预定onset/anchor，也没有合并旧gate/mask分支。此理想仿真传感阈值尚未验证真实噪声鲁棒性。

| 稀疏时刻 | 原力幅gate投影后右臂RMS rad | 接触/释放gate投影后右臂RMS rad |
|---|---:|---:|
| 1.2s接触开始 | .000117 | .0000358 |
| 1.3s | .000315 | .0000312 |
| 3.1s保持开始 | .003093 | .0000057 |
| 3.5s保持 | .001886 | .0000045 |
| 4.6s释放 | .012969 | .0000429 |
| 4.7s释放 | .005295 | .0000154 |

两组均按同一100ms raw latent对齐五帧动作，接触/释放gate下未发现阻止本次拟合的10Hz/50Hz表达边界；非右臂残差仍被统计，并非强制mask。

| 真实6秒闭环条件 | 阶段 | 3D误差 RMS/峰值 mm | 实际力 RMS/峰值 N | root最低 m / 最大tilt ° |
|---|---|---:|---:|---:|
| standing BFM | 移动 | 8.733 / 13.318 | 0.749 / 1.078 | 0.780514 / 2.643 |
| standing BFM | 保持 | 12.920 / 13.229 | 1.033 / 1.058 | 0.780300 / 2.649 |
| standing BFM | 释放 | 18.375 / 22.102 | 0.000 / 0.000 | 0.779987 / 2.593 |
| 特权教师 | 移动 | 5.096 / 6.753 | 0.437 / 0.549 | 0.780464 / 2.550 |
| 特权教师 | 保持 | 7.596 / 8.314 | 0.616 / 0.672 | 0.780245 / 2.500 |
| 特权教师 | 释放 | 17.103 / 21.346 | 0.000 / 0.000 | 0.779945 / 2.412 |
| Student | 移动 | 5.264 / 7.035 | 0.451 / 0.574 | 0.780451 / 2.534 |
| Student | 保持 | 7.502 / 8.155 | 0.607 / 0.658 | 0.780220 / 2.519 |
| Student | 释放 | 16.738 / 21.090 | 0.000 / 0.000 | 0.779933 / 2.428 |

接触前B/T/S状态差=0、无额外Student动作；几何误差RMS42.915mm来自原初始化/未激活anchor，不作跟踪成绩。释放误差也仅为inactive-anchor几何；force0来自Tether关断。移动阶段腕世界+X位移B/T/S=22.668/28.824/28.482mm，腕相对root +X=−.620/7.359/6.779mm，root +X=23.288/21.464/21.703mm：确实有相对右臂响应，不只是root平移。保持平均实际力XYZ B/T/S分别[.935,.421,−.116]/[.531,.301,.071]/[.538,.271,.057]N。

安全与连续性：Student接触后双足实际正法向支撑比例100%、总法向力最小344.151N，切向接触速度峰值2.385mm/s，分阶段足部body水平位移≤1.984mm（不把body位移等同足滑）。实际关节超限、动作/力矩饱和、非足触地均0。保持阶段相对自身standing动作的非右臂q_PD残差RMS=.000154rad、最大=.000608rad。释放最大q_PD步长L2：B/T/S=.002217/.006602/.006657rad，Student最大单关节=.004776rad。动作增量于4.8s已精确为0；gate浮点尾项2.13e−14到4.82s才数值严格归零，对4.8s动作无影响，未为此重复运行。原初始化的q_PD大步长/初始接触瞬态保留，未称全程零滑移或硬件安全认证。

负对照与未解决边界：

- Student无Tether的完整6秒轨迹，与既有无力BFM的qpos/qvel/force/wrist/root最大差全部0；gate全零。这证明不会在无接触时自动执行固定腕动作，但很大程度由gate保证，不能独自证明MLP学到了接触意图。
- 在同一教师h、固定proprio和gate下，仅将力历史置零/反向，右臂q_PD分别变化RMS **.000568/.001165rad**（全身.000279/.000573rad）。网络确实使用了力通道，但这仍是离线输入反事实，没有测其物理方向正确性；单条轨迹的记忆解释尚未排除。
- Student已访状态相对教师输入（按训练scale归一化）最近样本RMS距离均值.03273/最大.04350，同时间均值.03302/最大.04448；本轮没有出现“离线拟合、物理崩溃”，也没有追加数据聚合。上述距离不构成广泛可观测性证明。

核心diff只涉及个人research_notes：在已验证runner增加单dev2、50Hz观测记录、MLP训练/Student执行；新增ContactGate/受限latent/监督loss及稀疏可行性检查、5项CPU测试、复用supervisor的本次路径/预算、纯离线结果审计。原科研源码/共享checkout/旧权重与终检无diff，未push、未启动Bridge。

实际1次启动成功，共24模拟秒（baseline、teacher、Student、无Tether各6秒）；2000学生更新+3200诊断优化步，未启动PPO。5/5 CPU测试通过，完整24秒原PD审计最大q_PD误差6.46e−8rad、力矩误差0。进程墙钟186.003秒，4CPU、RSS峰值7.081GiB、GPU5本任务峰值610MiB；PID/PGID3522515已退出、owned组为空，其他任务保持运行。

复现与归档：

- 个人分支：`gaoe05/contact-projector-single-scene`；运行commit `9ef5b35714ed6d4e19c206451f97fc75c2effff0`。
- 工作区：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/issue-21-contact-projector/repo`。
- 输出O：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-projector-20261030T004000Z/attempt-01/outputs`。实际开始时间见manifest/ledger：2026-10-29T16:24:36Z；run-id仅目录标识。
- O包含`teacher-dataset.pt`、`dataset-manifest.json`、`projection-spot-checks.json`、`fit-{before,after,trace}.json`、`contact-input-perturbation.json`、`dev-2-{baseline,teacher,student,student-no-tether}.json`、`summary.json`、`offline-audit.json`、`result.json`。stdout/stderr/telemetry在O上一级，全部数据/日志留服务器，不入GitHub。
- `O/contact-projector-final.pt`仅453293字节；SHA256 `14b5a32798586d0810ff5e5461ce745890a90c717d1e6de2e8d6dc3f2a296fd8`。Projector模块SHA256 `1a212c186d4a58ca86c53cf8f23471121589daa1e76c7c7c8d3d06056ae509f2`；运行入口SHA256 `83646b54ca79a4d2370ca94de28ff8fd9443230c274175d93d5720494d6d39dc`。

重现命令（仅说明，不再次执行；使用新唯一run-id并复核资源）：

```bash
cd /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/issue-21-contact-projector/repo
/mnt/data/temp_user/yi.gao/projects/compliance_BFM/.venv/bin/python -B research_notes/contact_projector_supervisor.py --run-id issue21-projector-NEWUNIQUEID
```

**下一项决策：保留这个固定Student，下一轮只做同一+X场景的一个未训练接触起始时刻对照，检验反馈响应与轨迹记忆的区别，再决定是否需要一次最小教师查询聚合。** 本轮已完成第一个监督型训练闭环，到此停止；没有扩到−X/新motion、没有追加训练或重开原终检。
