# Issue53 — 路线A小型交互闭环与FB影子提名（预登记）

2026-11-08。新授权阶段，不恢复Issue52 Expert PPO。[Issue53](https://github.com/GaoE05/robot-research-bridge/issues/53)。精确条件、预算与停止规则见 `interactive_protocol.json`。

先使用S4000实际闭环的完整状态测Teacher恢复域与成本。首批4个成本点：standing4.0s、接触前4.7s、短请求5.4s、长请求7s。不同初态/需求的20个状态来自4个整回合；仅原7候选/2.4s作正式Teacher，FB额外每状态最多2序列。候选物理是预测分支，第一命令反事实查询、完整Teacher接管、Student实际执行严格分档。

搜索查询从真实Student状态冷启动phase=0，无上一计划；不能照搬另一Teacher回合的phase。接管连续重规划后，phase仅按其实际执行的motion命令累计。原生motion热启动是来源先验，不是事实步态相位。伙伴短期常速度预测只取当前信息。每次完整状态恢复须逐值相同；原BFM/PD/solver/动作/force/last-z历史保留。

FB只在当前进程将原冻结 `_forward_map` 放到GPU，核对原normalizer、实际全BFM观测（含原机器人privileged_state）、原29D低层actor action及合法z；不改变原权重/控制器。`min_heads(F(s,a,z)·u_forward015/150000)` 为固定机器人状态reward的近似后继价值，非接触回报。F只看第一时刻，不预测后续变z的完整2.4s。B/reward来源仍为已保存6个合法z，不新生成。物理对比7vs7替换池，F-top4保留原池物理最优的比例另报，不能把9候选优于7候选算方法收益。

若至少20次跨回合比较当前提名0胜、无物理边际收益或恢复贡献，暂停当前FB提名；并不否定整个FB。监督仅用真实有效接管域；同S4000/seed/网络/旧归一化/1000更新比较旧数据续BC与交互数据。两臂共2000更新，新增专家成本单独报。默认不启动Student PPO；有实际收益且安全时才按授权条件启动，失败证据则及时停止。

开发/封存按整回合、共同前缀族分组，不随机分相邻帧。原S4000、新S-init、历史S0身份均核验，不互换。Student1639D仅本体465、真实上一z256、理想力历史900与已知安装18；不输入anchor/真值伙伴速度/时钟/active/候选答案或特权特征。真机传感尚未实现。

资源为用户上限12000 experiment-job秒/8GiB/96完整查询/40额外2.4sFB序列/2000监督更新/12288StudentRL决策；各类预测/真实/训练/评价/视频分别记账。所有运行只个人隔离worktree，代码固定后发车；不改main/共享BFM/PD/权重/SoftSONIC/他人任务。


## 条件启动实现补充（物理接管进行中，尚未训练）

匹配BC两臂从同S4000开始，原归一化冻结，1000更新/臂、同seed/旧批次；交互臂将其中35%换为实际验证域标签。开发只用short08/normal72/stop66，安全优先，`mean(force+.03*error_cm+cap)+.1*worst`；另mid/none回归及预封存59cm前缀/6cm条件不选模。same-state完整5子步检查只用于诊断。若有实际安全收益，条件性Student PPO每seed8×768，两seed5301/5302同起点；actor和critic均只读受限1639D，物理reward使用环境特权，fresh critic/Adam，原std.008/概率/KL回滚。critic不继承旧BC随机值函数。固定8批主终点，4批保留，不依封存重选。此处为实现，是否发车由本轮真实证据决定；不是已训练/成功。

新增样本按实际恢复窗分组，首个Student访问查询与同一接管窗后继合并为同组，逐状态去重；不能把首个单点另当一个episode高频重采样。独立回合数、恢复窗数和相关后继分别报告，不称几十帧为几十个独立任务。
