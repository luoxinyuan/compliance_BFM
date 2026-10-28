# Issue #19：接触时刻锚点修正与无外力动作回放消融

本轮按用户直接授权执行，不开发或启动 Bridge，不重放已消费的队列任务。完整保留上一组 reset-fixed anchor 实验、旧 launcher、旧失败日志及模型资产。

## 结论

1. **锚点位置预载修正成功。** 原始闭环 BFM 跑满 6 s、未跌倒；接触开始时三维位置误差和弹簧力均为零。实际力峰值 0.901718 N，远低于旧实验的 5 N。
2. **动作回放消融得到真实负结果。** 保持相同 Tether、初始状态、latent、原低层 PD 与物理反馈，逐项回放无外力动作；执行到 3.70 s 时 root 高度降到 0.396216 m，触发未修改的 0.4 m 失败阈值。没有继续调参或重试这个科学负结果。
3. 结果支持：在本次接触条件中，在线 BFM 状态反馈对稳定有贡献。它尚不证明意图理解、主动协助、纯 X 跟随、接触专家或 LPP。有初始阻尼力及后续侧向力；不能把锚点向 +X 移动直接等同于外力始终沿 +X。

## 固定协议与复用路径

保留原始 `FBcprAuxModel`、`ComplianceEnv`、`Tether`、`run.evaluate`、`HumanoidVerseVectorEnv`、观测/归一化/history/动作接口、MuJoCo 和低层 PD。源代码基线为个人分支父提交 `3934d79c079b496d0a13717bbb0b43592aa691e7`；共享 checkout 仍为 `6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`。

Checkpoint SHA256：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。继续使用既有只读 model-view 和 LaFAN；不安装依赖、不下载或改权重。standing latent 仍由原始 `goal_inference(default standing reset observation)` 生成，norm=16，所有条件相同且全程固定。原模型 eval/frozen，所有参数 CPU，无梯度、无 CUDA。

从上一组已记录的无外力轨迹中，选择 **physics post-step t=1.0000000000000007 s 的右腕世界坐标**，它就是下一次施力之前的状态，而不是 t=1.005 s 的状态。该基线 JSON SHA256：`ef6e2dfab4e314b59636870f1abc719c304192758420be190af307991bcf4898`。本轮又实际运行无外力条件，核验完整 qpos/qvel/action 与归档一致。

共同 anchor origin = `[0.13686332656697187, -0.1968667867765935, 0.848818048880822]` m；delta=`[.03,0,0]` m。通过既有 `Tether.reset(..., trajectory=(origin,delta))` 输入，不改 Tether 实现。参数保持 K=80 N/m、D=8 Ns/m、max_force=5 N、activation=1 s、anchor_delay=0、2 s smooth ramp、4.5 s release、6 s episode、seed=0。仍为维持连接的弹簧/阻尼 tether，不是单向推触。

三种条件：

- `no_force`：原始 BFM，Tether disabled；作为新鲜配对基线和动作序列来源。
- `closed_loop`：相同状态/轨迹，Tether enabled，原始冻结 BFM 每 20 ms 根据实时观测与历史计算动作。
- `replay`：闭环完成 6 s、未跌倒且锚点/配对/力公式核验通过后才启动。仅在原 `model.act` 的调用位置返回匹配时刻的无外力 action；不调用 BFM actor。原 wrapper 接收这些动作，使用实时物理状态计算 PD 扭矩和推进历史。回放越界立即停止，绝不重复最后一个动作。

## 数值结果

下面统计各自真实运行窗口；回放没有 3.70 s 之后的数据，不补齐，不计算其 release 恢复。

| 指标 | 无外力 | 闭环 BFM + Tether | 无外力 action 回放 + Tether |
|---|---:|---:|---:|
| 实际模拟时长 s | 6.00 | 6.00 | 3.70 |
| 物理样本 / 控制决策 | 1200 / 300 | 1200 / 300 | 740 / 185 |
| 真实 BFM actor 调用 | 300 | 300 | 0 |
| 原始失败判据 | 未触发 | 未触发 | root height <0.4 m |
| 力 RMS / 峰值 N | 0 / 0 | 0.584745 / 0.901718 | 3.018507 / 5.000000 |
| 力达到 cap 的活跃样本比例 | 不适用 | 0 | 41.8519% |
| 最低 root 高度 m | 0.779834 | 0.779995 | 0.396216 |
| 最大 root 倾角 deg | 2.524867 | 2.564620 | 63.417573 |
| 双足接触比例 | 99.4167% | 99.4167% | 98.5135% |
| 足部 XY 从 reset 最大位移 mm | 16.8611 | 16.8970 | 53.3153 |
| action / YAML torque 饱和比例 | 0 / 0 | 0 / 0 | 0 / 0 |

**同一 0–3.70 s 窗口**比较：闭环与回放的力 RMS 分别为 0.638369 / 3.018507 N，最低 root 高度 0.780467 / 0.396216 m，最大倾角 2.564620 / 63.417573°。两者施加相同的 partner law 和 anchor 轨迹，实际力随各自机器人状态变化，因此不是相同外力时间序列的比较。

### 完整力方向与 preload 核验

两种接触条件在施力开始时，位置误差 `[0,0,0]` m，spring force `[0,0,0]` N。腕速度为 `[+.0284880,-.0346713,-.00738782]` m/s，anchor 初速为零，首个实际力完全来自阻尼：`[-.227904,+.277371,+.059103]` N，模长 0.363824 N。它包含负 X 分量；未把这种速度阻尼误报成位置预载或纯 +X 拉力。

闭环全程力 XYZ RMS = `[.374216,.443128,.074348]` N；各轴绝对峰值 = `[.653690,.657542,.159254]` N。活跃期 31.24% 的样本 Fx<0；在 |F|>=.05 N 的样本中，与 +X 夹角中位数 50.67°。保持段 [3.5,4.5) s 的 XYZ RMS 为 `[.523453,.625132,.032897]` N。侧向分量不可忽略。

保存了 200 Hz 完整 force vector、anchor、前后 wrist、速度、spring/damping 分解和原始施力值。用分解重新计算实际力，与原 Tether 的最大误差仅 4.44e-16 N。分解仅用于审计，未参与控制。原 `settling_time` 阈值仍是 5 N，与本实验 cap 相同，故不作为 force relief 的成功证据。

### 手腕、root 与 BFM feedback

三种条件初始 qpos/qvel/actor observation/latent 一致，前 1 s 状态与动作逐项一致。接触使两种条件在 1.005 s 首次偏离无外力状态。闭环在 1.02 s 首次修改 actor action，闭环和回放的物理状态在 1.025 s 开始分离。

闭环晚保持段相对无外力的腕世界 XYZ 增量为 `[4.4605,6.6606,.5942]` mm；扣除 root 平移后为 `[2.6930,4.2471,.3460]` mm；在各自旋转 pelvis frame 中为 `[1.1198,3.0671,.7026]` mm。root 世界增量 `[1.7675,2.4135,.2482]` mm。腕相对 anchor 起点沿 X 移动 23.4716 mm，但无外力本来就会移动约 19.0111 mm，因此不能把整个 23.47 mm 都归因于引导。保持段 anchor tracking error 的三维 RMS 为 10.1308 mm；平均 XYZ 误差 `[6.5284,7.7350,-.2640]` mm。

闭环保持段 normalized BFM action RMS delta=0.00269940、max abs delta=0.00808989，latent variation=0。回放已经执行的 **185/185** 个 action 与对应无外力动作逐项完全相等，在线 BFM 调用为零；但 PD 扭矩确实随新状态改变，低层反馈保留。

闭环释放后外力归零，6 s 时相对无外力腕 X 残差 2.6006 mm、相对 root 为 2.3391 mm，root X 残差 .2615 mm。恢复窗口仅约 1.5 s，不证明渐近收敛。回放在 release 之前失败，因此没有 release 指标。

### 支撑和首个真实失败边界

闭环接触开始后所有采样均有双足接触；无接触只出现在初始 reset settling 的五个早期样本。回放在 3.68 s 有一个双足均无接触的物理样本，随后 **3.6999999999999433 s** 第一次触发 `ComplianceEnv.failed`：root height=0.396216 m，低于原阈值 0.4 m；倾角 63.4176°对应 upright cosine 仍高于 .4；qpos/qvel 均有限。控制层在 3.70 s 结束，exit code=2 表示已记录的科学负结果。

不能仅凭回放多数时间仍有双足接触或没有 action/torque 饱和，就声称稳定。其 root 最终显著倒向后方，脚部法向力下降，物理轨迹保留在原始数据和图中。

## 工程尝试、测试与资源

- Attempt 01：无外力和修正锚点闭环均正常；回放输出保存时，原环境的失败分支返回 `numpy.bool_`，标准 JSON encoder 报 `TypeError`。这是本轮首个工程失败边界。该次没有完整落盘的 replay 轨迹，不将其报成完整独立重复实验。
- 仅修复日志标量转换，增加失败数据保存前的里程碑；同时修复提前终止比较，只比较共同时间窗，绝不相减 3.7 s 与 6 s 的末帧。未改控制器、动作、扰动、阈值或科研目标。修复 diff 保留在本地；两次 launcher 的完整副本保存在各自 attempt。
- Attempt 02：完整记录上述负结果。两次 no_force 和 closed_loop 的原始 JSON **SHA 完全相同**，证实报告修复未改变它们的轨迹。发现 replay 的科学失败后停止，不调参重试。
- 最终 Mac 与服务器依赖无关单元测试各 10/10；真实保存数据在两侧重新计算、核验通过（跨 Python 版本容差 1e-12，最大差约 1.99e-13）。复核通过不等于回放存活。
- CPU affinity 4、计算线程 1、GPU 0、8 GiB 进程组 RSS watchdog、累计墙钟上限 3600 s；两次实际合计 274.164 s，最大 RSS 和 7.101883 GiB；两个进程组均已退出。Trusted SSH 不是 OS 原生隔离。

## 下一步最小实验（未执行）

先补一个 **无 Tether 的完整 6 s action replay 对照**，要求 300 个动作和物理轨迹重现无外力基线。这能排除移除在线 actor 调用本身引入的未观测执行差异。目前已验证相同初态、185 个精确回放动作、1 s 精确前缀及实时 PD，但尚未完成这个完整控制条件。

若通过，再用同一预先固定协议做一个反向（-X）配对，区分状态稳定反馈和方向相关的腕部引导能力，并继续记录实际三维力。保留 D=8 时不应要求首个力为零；若要更纯的方向性干预，应另行预先定义速度连续的 anchor 轨迹，而不是事后更改本次结果。暂不训练 LPP/PPO，也不把回放负结果等同于已有合格接触专家。

## 复现与文件

个人工作区：
`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/issue-19-a8187804319761a4/repo`

运行目录：
`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-wrist-anchor-20261028T062503Z`

正式完整数据在 `attempt-02/outputs/{no_force,closed_loop,replay,result}.json`；另有 standing latent、`anchor_replay.png/.svg`，完整 stdout/stderr/telemetry 和两次 launcher。保留 attempt-01 和上一轮 run，不清理或覆盖。Git 仅保存脚本、精简结果 JSON 和本文，不提交权重或逐步大日志。

使用新的唯一 run ID 才能重新运行（本轮不再运行）：
```sh
cd /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/issue-19-a8187804319761a4/repo
python3 -B research_notes/right_wrist_anchor_supervisor.py --run-id issue19-wrist-anchor-NEWUNIQUEID
```

仅分析与验证已保存数据，不调用模型：
```sh
python3 -B research_notes/test_right_wrist_anchor.py
python3 -B research_notes/right_wrist_anchor.py --analyze RUN/attempt-02/outputs
python3 -B research_notes/verify_right_wrist_anchor.py RUN/attempt-02/outputs
# 使用既有项目 Python/matplotlib，并将 MPLCONFIGDIR 指向本 run 的 outputs/cache/MPLCONFIGDIR：
/mnt/data/temp_user/yi.gao/projects/compliance_BFM/.venv/bin/python -B research_notes/plot_right_wrist_anchor.py RUN/attempt-02/outputs
```
