# M39-A 当前执行入口（2026-11-04）

用户新授权 **需求分配与真实闭环优化**，不是继续左转资格检查。M38-C双向能力有限成立的结论及旧失败保留，下方为前阶段快照。

[当前阶段 Issue39](https://github.com/GaoE05/robot-research-bridge/issues/39)。独立分支 gaoe05/m39a-demand-allocation；seed3901/3902独立worktree。D2 SHA 9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5 已实核。T30/S0/D0/D1/D2及原权重均保留。

已启动：GPU1上的冻结D2/T30需求分布物理与奖励复算、有效覆盖数据采集，runtime 1f2cccab6286171904900ddd4b97a2494040bd0c。尚未启动：3000更新覆盖监督、两个独立seed真实PPO及新封存终检；方案存在不代表训练成功。

预算：24000累计job秒/15GiB，最多3空闲GPU，PPO每seed≤98304决策，监督总≤6000。本轮初始核验GPU1–4空闲、CPU98%idle、RAM912GiB、IOwait0。新job必须再次核验。运行根 bridge-v2-owned/runs/m39a-demand-20261104；查看compute-ledger.json确定实时进程，不重启已消费任务。

固定1088D特权Direct、BFM/PD/总5N。零需求、无接触、小角度旋转、平移、组合和同回合需求变化；无未来轨迹/任务ID/clock输入。训练期可读当前特权需求，不能称部署Student。当前不做Student/PPO方法对比或真机。完成后统一结项；不将旧阶段停止说明解释为禁止本轮已授权PPO。

---

# CURRENT_STATE — 新会话先读

最后核对：2026-11-04（北京时间）。**M38-C已完成并停止。D0/D2在同一个checkpoint实现开发及新±12°正确接触转腰；完整双向保旧与连续转换仍未证明。** 无训练排队，无PPO、Student蒸馏或M39训练。旧T30/S0、C0/C1/C2及全部失败不覆盖。

入口：[研究纲领](RESEARCH_OVERVIEW.md) · [M38-C完整报告](M38C_RESULTS.md) · [Issue38](https://github.com/GaoE05/robot-research-bridge/issues/38) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。[M38-B](M38B_RESULTS.md)、[M38](M38_ROTATION_RESULTS.md)及全部历史保留。

## 当前可靠能力与真实新结果

- T30→S0仍是有限域右腕直行/停止Teacher→ideal-force Student基线；用户认可可执行行走和小幅上半身旋转。ideal-force不等于已确认G1真机力传感能力。本轮D0/D1/D2是**特权Direct Expert，不是新Student**。
- BFM/PD不变；rigid torso_link两点[0,±.13,.15]m各2.5N，总5N；双向保持接触Tether，非单向按压。原双点平移Student仍有旧腕退化，不替换S0。
- **右向有真实额外反馈：**同一运动末，C2右接触−11.54°、同框架无力−6.52°、无力高层命令＋真实接触回放−7.47°。左接触反而从命令回放−3.69°恶化为完整C2−8.22°。高层变化参与了错误，不能仅归因BFM不会左转或被动外力。
- C2在接触前已逐步分叉。左3.8/4.5/5.5/7/10s及右7s完整前缀匹配接管均有效，新增768真实恢复决策标签；**只有6个独立C2分叉起点**，不是768独立偏离状态。直行5s接管误差略降但力升，141标签未入训练。
- **三组匹配各3000更新、同C2初始checkpoint：**D0原数据额外监督；D1加18.75%新有效恢复短块；D2同D1抽样和预算，5高层决策使用自身预测z递归。物理/本体/动作历史仍是真实记录，D2不是可微物理rollout。无网络/reward/cap/PD变化。
- 新封存±12°，onset4.45/ramp8.05/release14.5/end19.5，固定终点一次评估。**D0：+8.41/−11.03°；D2：+9.49/−11.46°；D1：−8.60/−11.09°。** D1开发左正确、封存左反向；没有测试重选/补训或按方向拼接。
- D2新左1.264cm/1.932N，新右.738/1.185；相对同时刻自身无力的torso增量+15.09/−6.11°，主要腰相对骨盆，不等于迈步转身。D0也成功，所以短序列不是已证明唯一必要修复。
- **代价仍明确：**D1开发直行6.213cm/3.719N、局部2.850/2.128较好，但泛化左失败；D2直行7.086/3.892、局部5.427/3.930且额外转腰约13.38°。T30直行5.826/3.599、局部4.121/3.107。新零需求D2仍−9.57°、1.675cm/2.695N（T30−2.63°/.517cm/.829N）；新17cm/+10°组合D2为8.158/4.419，未形成可靠连续组合。
- 99完整物理回合无记录安全边界/跌倒/支撑丢失。D2新左/右q_PD峰.720/.419°每20ms，实际q峰.089/.093°每5ms；直行q_PD峰43.17°但实际q峰2.33°，不得混称。解除后允许自然收步，不计旧anchor跟随失败；零需求偏置发生在仍接触阶段。

## 当前首要决策

问题已从“是否能学左转”推进到**一个模型能否在零需求、局部请求和真实转换状态中合理调用已学双向能力**。D1/D2的取舍说明仅增加访问轨迹或降低离线误差都不自动保证泛化。新增标签以专家恢复后继为主，未全面覆盖新模型访问域；只做了一个训练seed，不作统计普遍性宣称。

唯一建议：下一阶段以冻结D2作为双向旋转候选、T30/S0保留旧基线，在有限真实同回合平移→旋转→继续移动中核查可查询反馈域，并并列保留零需求/局部请求。**可以进入M39小规模过渡能力验证；尚不能直接宣称连续共享策略资格或启动大规模训练。** 本轮预算已完成，不自动追加训练，不重新引入force-to-yaw状态机。无需先清零所有历史误差。

## 精确版本与数据入口

长期个人分支`gaoe05/research`；最终文档commit见Issue38 M38-C结项的精确链接。实际接管/训练runtime **e88f259adf4b7677eb3f34c254567419dce37102**；封存runtime **97471c7924cda9d3118acd1ab1e0d9b81f28150e**；诊断/渲染 **c19fbf4075b1d542c3fe95eef79eca908187b185**。

服务器根：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m38c-history-20261103/`。

|模型|相对根路径|SHA256|
|---|---|---|
|D0|A-train-06/job/D0.pt|5e1047b940950208aa514f5ecc26ca75a2dff631f193cbe09457138f9d59c20d|
|D1|B-train-04/job/D1.pt|1801ec464f455c48cae6b416f05ee1da86bed495f9f324ae0c9eef0ee5e6287e|
|D2|C-train-05/job/D2.pt|9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5|

T30：`stop-sensing-20261103/B1-train-02/job/best.pt`，SHA c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33。S0：`direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt`，SHA 0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12。BFM SHA 33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。这些路径均相对项目`bridge-v2-owned/runs/`，原始资产不在Git。

[结果](m38c_artifacts/results.json) · [物理分段](m38c_artifacts/physics.json) · [因果时间对照](m38c_artifacts/contact-contrasts.json) · [真实接管](m38c_artifacts/recovery.json) · [同状态查询](m38c_artifacts/same-state.json) · [账本](m38c_artifacts/resources.json)。原始轨迹和权重仅服务器。

Mac：`~/Projects/compliance-bfm-diagnostics/videos/m38c-history-consistency/`，仅3MP4＋INDEX：新左D1失败/D2成功与无力；新右C2/D2接触vs无力；T30/D2直行代价。真实qpos1×、SHA/时长/全解码/画面核验，不为视频重采样，也不以不同模型拼接成功。

## 成本、安全与接力

9000监督更新、PPO0、Student0；99完整回合，另有15次诊断ready重建。累计受监督job-wall3264.463s，约2.00GiB。首启动稀疏检出缺文件的加载前失败保留，仅物化新worktree原tracked源码；无安装、共享配置修改或资产下载。GPU1/2/3全部释放，GPU0/5/6其他任务不动。

新会话先读本文件/纲领/最新报告，核对live Git、checkpoint SHA、ledger和实际资源再行动。旧C0/C1/C2/r1/r2与旧实验失败完整保留；不要重放已完成任务或根据文件名best推断身份。团队main、共享BFM/PD/权重、SoftSONIC不改，无Bridge/watch、无真机。Library 2026-10-28快照是历史；当前状态以本文件和GitHub最新Issue为准。
