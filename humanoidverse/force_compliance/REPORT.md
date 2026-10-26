# 主动全身顺应：实现与验证报告

实现位于 `/home/xl521/BFM-Zero/humanoidverse/force_compliance/`。原有 BFM 架构、训练与推理入口未修改；开始任务前已有的数据及演示视频修改未覆盖。

## 新增文件

- `config.py`、`default.json`、`no_force.json`：交互、初始姿态、奖励、控制频率与随机种子。
- `interaction.py`：世界系弹簧阻尼 tether，独立锚点轨迹、力限幅、偏移接触点力矩。
- `environment.py`：原生 BFM 观测/动作接口、逐物理子步施力、冻结模型、稳定性奖励、终止与可视化。
- `policy.py`：高层高斯 actor/critic，站立 latent 的有界切向残差和球面投影。
- `ppo.py`：独立 PPO，持续动作奖励聚合和持续时间折扣。
- `run.py`：基线、小规模训练、配对评估、JSON/曲线/视频导出。
- `README.md`、本报告、包初始化，以及 `tests/test_force_compliance.py`。
- 本地 `.git/info/exclude` 排除下载权重、实验产物和 Python 缓存。

## 官方模型

下载自 https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/main/new_model_for_training_code_inference ，固定版本 `62b4206d68e026de5e5dc7efb1529bccfb95164c`。

位置：`pretrained/official/new_model_for_training_code_inference`。模型文件 3,384,968,272 字节，SHA256 已与官方 LFS 元数据核对：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。

仅下载权重和模型/环境配置，没有下载 replay buffer，没有重训 BFM。没有跨 checkpoint 借用旧站立 latent；使用当前模型 `goal_inference` 编码默认站姿，并实测稳定性。实际 latent 维数 256、范数 16；低层 50 Hz，高层 10 Hz，物理 200 Hz。

## 公式

`x_r = x_body + R_body p_local`，`v_r = J_point(q) qdot`。

`F_raw = K_h (x_h-x_r) + D_h (v_h-v_r)`；`F = F_raw min(1, F_max/max(||F_raw||,eps))`；`tau_COM = (x_r-x_COM) × F`。Jacobian 只计算接触点物理速度，不生成运动参考或控制目标。

默认从回合开始保持连接，延迟 1 秒后移动锚点，2 秒 smoothstep 运动，再保持至 6 秒回合结束。`x_h=x_start+(3u²-2u³) delta`，速度采用解析导数。锚点不跟随机器人重新定位。

`r_F=-mean_substeps(||F_applied||²/F_scale²)`，叠加根部角速度、关节限位、非脚落地、力矩、关节加速度、动作变化与 latent 变化惩罚。跌倒/数值异常单独失败，附带剩余回合的吸收态力成本；不惩罚根部位移或迈步，不使用运动跟踪奖励。全部权重与完整公式见 README。

`z=project_z(s+alpha*g(F)*v_tan)`，其中 `s=project_z(z_stand)`、`v=tanh(a_high)`、`v_tan=v-s(v·s)/(s·s)`，`g=clip(||F||/F_gate,0,1)`。零力时精确返回站立 latent；PPO 概率比对原始高斯动作计算。

`R_high=sum_i gamma^i r_i`，`discount_high=gamma^k`；GAE 在高层决策边界计算，终止不跨回合 bootstrap。

## 实测

- 全套 46 项回归测试通过（含 10 项新增测试）：力随跟随减小、限幅、世界系偏移点力矩、旋转点速度、真实 G1 施力、接触/锚点时序、数值异常失败、latent 约束、PPO 更新与变时长折扣。

- 无外力站立 6 秒：没有跌倒，施力为零；这不是顺应成功证据。
- 默认 tether 基线：锚点位移约 3.12 cm，峰值 4.579 N，RMS 4.246 N，没有跌倒。
- PPO 仅 2 次更新、64 个高层决策（320 个低层步，6.4 秒训练交互）：损失有限，没有跌倒，actor 输出层从零初始化产生非零权重。优化器只包含高层网络，BFM 全部参数禁用梯度。
- 3 组 6 秒配对评估，绝对锚点轨迹逐样本完全相同，初始接触点相同；基线/策略均未跌倒。

| seed | 基线力 RMS (N) | PPO 冒烟策略 RMS (N) | 跌倒：基线 / 策略 |
|---|---:|---:|---:|
| 0 | 4.2461 | 4.2429 | 0 / 0 |
| 1 | 6.1779 | 6.1808 | 0 / 0 |
| 2 | 2.8193 | 2.8196 | 0 / 0 |

**没有有意义的性能提升。** 当前训练量只验证算法链路；结果不支持“已经学会主动全身顺应”或“已经能通过迈步消力”。冻结站立基线能在这些小扰动下保持平衡，但仍存在持续非零接触力，不能据此声称它能主动把外力降至零。

当前 settling 阈值为 5 N；数值 0 表示锚点停止时已低于该阈值，不表示接触力为零。seed 1 在回合结束前未达到该阈值。support_changes 只是接触支撑变化，不等价于已确认的迈步。

## 产物和命令

高层 checkpoint：`artifacts/force_compliance/train/policy.pt`。

配对指标、逐步数据、力曲线及 6 段视频：`artifacts/force_compliance/comparison/`。无外力视频：`artifacts/force_compliance/no_force/000_standing.mp4`。测试日志与配对一致性检查：`artifacts/force_compliance/validation/`。

运行命令见同目录 README 的 Commands；核心入口为：

```bash
cd /home/xl521/BFM-Zero
MODEL=pretrained/official/new_model_for_training_code_inference
.venv/bin/python -m humanoidverse.force_compliance.run train --model-folder "$MODEL" --output artifacts/force_compliance/train_next --iterations 2 --rollout 32 --device cuda
.venv/bin/python -m humanoidverse.force_compliance.run evaluate --model-folder "$MODEL" --policy artifacts/force_compliance/train/policy.pt --output artifacts/force_compliance/comparison_next --episodes 3 --device cuda --video
.venv/bin/python -m unittest discover -s tests -v
```

## 剩余工作

需要更充分的高层训练、方向/幅值系统扫描及更长回合验证，才能判断冻结 BFM 的有效顺应范围。当前仅单环境 MuJoCo；Isaac 批量施力、推接触模型、可靠步数检测、左手/躯干多接触泛化未验证。未加入低层微调。

初始版本延迟启用 tether 的试验保留在 `artifacts/force_compliance/pilot_delayed_attachment/`；上表采用修正后的从回合开始连接版本。
