# CURRENT_STATE — Issue50原生来源＋短程滚动反馈结项

最后核对2026-11-07。[完整结果](NATIVE_ROLLING_RESULTS.md) · [方法/复现](NATIVE_ROLLING_METHODS.md) · [Issue50](https://github.com/GaoE05/robot-research-bridge/issues/50) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。

## 当前执行状态与可靠基线

本轮有界物理阶段完成并停止；无新PPO、Student或C0专项自动续训。Issue49正式结论不改，I/C0/T30/S0、有限域Teacher→ideal-force Student、M1–M49全部旧模型/成功/失败保留。新滚动程序不继承或在线调用旧I/C0/T30/C32 actor。

冻结BFM SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`、原PD、合法Direct256D、10/50/200Hz、K80/D8、总10N。用户本轮授权H100级NVIDIA后端，非默认Pro5000/PPU。原始replay完整SHA `46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b`。

## 最新真实进展与失败

1. **原生reward推断确已完成**：同150000 successor、seed5001、3固定robot-state配置×普通/weighted共6z；保存embedding，原API对照差0。没有把force/anchor当B支持的零样本reward。6候选尚未成为可靠低力接触walking；weighted更弱，不继续扫描。
2. **速度不再只测极慢.09**：同72cm与完整standing onset，平均.09/.15/.22（峰值1.5倍）。motion全接触误差60.52/25.20/19.26cm、力仍约9N；侧漂/yaw与保持期反向牵拉保留。不能只因速度相近就称合格Expert。
3. **昂贵但真实有效的反馈获取**：7个native提议、2.4s真实物理窗口，只执行首100ms后重规划。完整.15牵引7.417cm/5.465N，C32为40.446/9.086，motion为25.195/9.171，冻结I为10.265/6.682。相对I误差/力−27.7%/−18.2%，主要改善移动及侧向负担；保持力与I接近。
4. **同程序停止有效**：7.8s前760physics/190BFM步、38次查询的完整状态fingerprint/历史/首指令逐值一致，8.1s后命令才分开。提前停止保持2.074cm/1.730N，释放后不再考核旧anchor，保留完整4s恢复。
5. **并非全质量解决**：停止分支总pelvis偏航约−20.29°；q_PD20ms峰56.91°不等于真实关节5ms峰（继续1.395°、停止2.495°）。无失稳中断仍不能自动等同自然性最优。新增reward-z在223次实际决策中均未选中，收益应归于原生goal/motion＋反馈搜索。

当前backend四状态恢复校验通过；局部/未起步有预测改善，但明显超前状态的所有候选仍10N，不能作为有效恢复标签。新增冻结停止参照I10.995cm/7.125N，MPC7.605/5.457；I保持期仍继续前移19.87cm，MPC0.695cm。完整短请求/.22滚动未运行，不把来源资格当反馈泛化。

## 当前主要瓶颈与唯一下一建议

已证明可从原生来源与有限物理搜索获得一个有用的窄域特权反馈程序；尚未证明它能实时执行、广泛恢复或泛化。每100ms控制约23s搜索，远慢于实时。未训练新神经Teacher/Student；原生reward本身质量不足，不能归因成新PPO学习成功。

下一建议：把这一有限有效域的真实首动作反馈摊销到独立的受限观测Student，先含短停/继续与少量速度/自然初态覆盖，再做冻结BC vs同初始化小预算自身闭环RL。保留当前yaw/源状态恢复边界；不继续奖励prompt扫描或回到C0修补。本建议**未启动、未获得下一阶段预算**。

部署actor只用本体、真实已执行动作/z历史、已知接触安装、明确传感假设下的力及因果记忆。1088D特权query与1639D ideal-force query分开保存；不能输入anchor/伙伴真值速度、任务clock/active真值、未来轨迹或MPC内部模式/相位答案。仿真理想力不等于G1已具备该传感。

## 精确身份、源码、数据和视频

- 长期个人分支`gaoe05/research`；实际滚动runtime **32b3502262399409dd9091b067c89ba619d20873**；后处理/当前backend诊断源码 **e0bc634fd36e124ea628660d20ea3b5e56b1f673**。文档最终提交不冒充物理runtime，最新公开HEAD以Issue50/Master结项链接为准。
- native初始SHA d399174c…a5cd；C32 39cd0db5…4ad0；I **72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a**；C0 **ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c**。无新Teacher checkpoint；Teacher身份=runtime＋protocol/原生资产manifest＋实际执行历史。完整SHA见报告。
- 项目相对stage：`experiments/native-rolling-h100-20261107`。运行树`repo`固定32b3502，文档/分析树`notes-repo`。`runs/reward-inference-01`、`qualification-03`、`reward-free-01`、`rolling-02`、`diagnostic-03`、`stop-reference-01`分别保存推断/资格/无力/真实滚动/四状态/冻结停止对照；`ledger.json`与launch、models、完整physics/control/searches均保留。
- 真实qpos1×视频Mac：`~/Projects/compliance-bfm-diagnostics/videos/native-rolling-feedback/`，仅3条MP4＋INDEX；原生速度来源、主反馈同条件比较、严格共同前缀停止/继续。预测分支没有混入视频。
- 外部重启改变挂载与现有runtime；旧隔离输出暂不可见而非删除。当前Torch2.7.1+cu128/MuJoCo3.3.3仅做必要差异检查；BFM/PD、backend内精确前缀和12概率回归通过（最大.00019837<.0003），无阈值放宽/PPU默认修复/共享安装。旧cu118/3.3.7数据不混作严格因果对照。

保守累计8900.57/18000job秒（含原673.38s、中断850s和CPU300s预留）；可见runs .477GiB，含源码worktree约.734GiB，旧挂载输出精确大小暂不可复核，以4GiB保守预留，仍在10GiB内。223实际滚动决策、749280候选physics步、4460实际执行physics步；四状态预测13440/实际首动作80，原生150000重标记/450000 B-map examples分别计数。正式RL和Student更新均0。

3条Mac MP4已核远端SHA、帧数、时长、完整解码和关键画面，INDEX说明对照身份及失败；没有原始轨迹/大数据塞进视频文件夹。全部owned作业退出，GPU compute为空，无队列/watch。最终文档/小型指标提交和远程分支SHA见Issue50结项链接，实际主runtime始终32b3502。

## 跨会话接力

先读本页、OVERVIEW、完整报告及Issue50；核对实际ledger/boot、runtime/模型身份和当前GPU权限，不以旧running/PID或空闲快照重启实验。Issue47/C0及旧Library为历史，不恢复其专项。Master保留全史；推送代码不等于新训练成功。未经新授权不启动PPO/Student、扩大候选或资源、不改main/共享BFM/PD/权重、SoftSONIC/他人任务，不做真机。
