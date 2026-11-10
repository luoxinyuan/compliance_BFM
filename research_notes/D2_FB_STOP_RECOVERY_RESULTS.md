# Issue57 D2：旧FB边界归因与新停止包验证

结论：旧198次边界的触发关节已定位；新完整停止包、原子发布、精确读回和
继续采样验证通过，但**一次真实FB更新后的严格模型/Adam一致性未通过**。
本轮停止，长跑禁止。不能把“文件保存成功”当作整个训练恢复合同已关闭。

## 旧244回合的只读归因

使用原固定runtime `147b22b684b0c03a613559fa6d5e5881d380ab7f` 的244条回合记录、
198条完整失败物理轨迹及对应原生源前缀。机械范围来自原MuJoCo实际XML，SHA
`2e42e1a493a88ddf85b37a8530a3dc39e1e7ad3dc0a72cacc1e93487df5ce2c4`。
未改原q_PD映射、PD、reward、10N或0.02rad边界。

|归档事实|数量|
|---|---:|
|完整记录回合 / physics_review|244 / 198|
|可匹配完整边界轨迹|198；缺失0|
|首个触发关节left_ankle_roll / right_ankle_roll|194 / 4|
|真实接触onset前触发|143|
|触发当下Tether inactive（含释放后）|156|
|交接后≤100ms触发|104|
|记录native failed=true|0|
|原生源最后一帧机械越界量最大值|0rad|
|触发时该关节q_PD超出机械范围|198|

全部198条首次触发均包含实际关节越界>0.02rad；没有由高度/倾角/非足接地/native failed
单独触发的替代解释。三类源分别70 moving、67 arm、61 standing。
真实关节速度绝对值中位约1.895rad/s。q_PD在范围外不等于实际q立即达到该目标，
也不单凭q_PD越界判定动作失败；本表依据真实q越过原物理边界。

|示例|真实q / 机械范围rad|qdot rad/s|q_PD rad|action / torque Nm|相对任务|
|---|---|---:|---:|---|---|
|第0回合，19.100s|−.283408 / [−.2618,.2618]|−.788808|−.504483|−.230055 / −3.12525|交接后12s，已释放；首次FB更新前|
|第243回合，5.580s|−.283976 / [−.2618,.2618]|−2.283892|−.592310|−.270105 / −.554431|交接后80ms，onset6.2s前；约496更新后执行|

逐回合CSV保留首个触发tick、实际q/dq、q_PD/action/torque、逐轴力、交接/onset、
源和实际控制索引对应的更新阶段。更新计数由真实控制累计及1024/16合同复算，
不是根据回合序号或课程概率猜测。

### 三种解释的边界

- **随机actor交接**：原生源用冻结原actor均值，交接后用mutable actor实际随机样本；
  第0回合在任何更新前已越界，证明不能将所有失败归为更新后学习退化。
  只在该更新前回合，用归档同一BFM观测/z复算初始化均值：末控制左踝q_PD
  均值−.277885rad，实际采样−.504483rad。随机动作显著改变命令，但这不是
  一条关闭噪声的真实物理反事实，不能据此给出噪声因果百分比。
- **更新后无接触退化**：后期接管后几十ms、实际力仍0，且采样窗口明显缩短；
  这支持问题不依赖当时接触力。旧512更新后模型/Adam/replay已丢失，
  不能恢复其均值/采样、更新前后或单个loss的因果对照。
- **边界标定**：查明真实机械范围、q与超出量，而非角度单位或索引猜测。
  q_PD可在机械范围外，MuJoCo软约束也允许小幅实际越界；
  没有证据据此宣称0.02rad设错，更不能自动放宽。198次均不等于198次摔倒。

## 新独立停止包

新runtime只用于有界工程验证，不是原1M续跑。提供：

- SIGTERM/SIGINT只登记停止意图；STOP_REQUEST文件由同步循环读取；
  在实际native控制边界保存，不在异步signal handler里写模型。
- 明确采集块start/target、块内位置、累计真实采样/物理/高层调用、更新数。
- 不可覆盖的rolling包、用户停止包；随机临时目录写完、逐文件hash/fsync，
  再原子rename发布，更新最新完整包指针。未完成临时包不冒称完整checkpoint。
- model含online与targets、六Adam、原/新normalizer状态、z_buffer、完整新replay、
  原生expert连续片段、Python/NumPy/Torch CPU/all CUDA/Task/physics RNG、
  真正执行过的29D action前缀、last-z、身体/solver/动作/BFM/接触历史及任务记忆。
- 新进程从同一原生源重建callback图，再重放已执行action而非用更新后actor重算；
  核对完整指纹及q/qdot/qacc/warmstart/constraint/ctrl/external-wrench等solver数组。

保存的物理分支是诊断/恢复重放，不写入新的学习replay冒充采样。
本实现没有自动恢复或长跑入口；原runtime和旧结项没有被替换。

## 实际短程回归与失败

control先采集1024真实transition并进行1次完整FB agent.update，
在下一采集块内第13个transition（累计1037）生成停止包；
不中断对照再采27个transition并完整更新1次。
新进程读取1037包，做相同27步和1次更新。原网/物理/reward均固定。

|对照|运行代码|包读回/实际继续采样|后续完整更新严格一致|
|---|---|---|---|
|control01 / resume01|e0c5c6b|逐值一致|失败：model/Adam不同|
|control02 / resume02，确定性Torch/CUBLAS|d990dff|逐值一致|同样失败|
|同一02包，仅重建agent入口统一为原factory|8ecd211|逐值一致|同样失败|

第一对量化：132个model条目有差异，最大绝对值 `2.3841858e−7`；
254个Adam条目有差异，最大绝对值 .0234375，对应原二阶矩最大约99434。
原日志标量、RNG、z_buffer、replay、物理前缀/继续采样均一致。
差异涉及discriminator、critic、actor及相应targets/Adam；F/B与auxiliary对应项无差异。
第二/第三次得到与第一对相同的差异后hash，不能把它解释为已经证实的随机GPU扰动。
模型布局检查亦未发现stride不同；原factory对照记录的模块training、requires_grad、
normalizer顺序、input filter和config均相同。

这些量级不等于严重策略退化或FB不能学习，但**严格更新恢复测试未过就是未过**。
没有事后调大容差、放宽PPO logprob/联合KL、修改力或物理边界使其通过。
不因文件/RNG一致就宣称更新过程完全可恢复。

额外CPU负测试通过：已有包覆盖被拒、hash篡改在建模前被拒、写入失败不发布、
原最新完整指针保留、SIGTERM只置意图。它们不代表已验证断电或SIGKILL下自动保存。
三轮失败和原始结果均保留；前两轮旧诊断入口的result.json停在初始running状态，
最终job侧车标为失败并链接traceback；后续入口增加独立failure/finish记录。

本轮实际短程更新总计**7次完整agent update**，新增真实采样**2209 transitions**
（两个control各1064，三个新进程各27）；重建物理前缀单独记账，不算新采样。
最初自登记6次工程更新后，基于失败增加一次构建入口对照并事先登记为7；
12000累计job秒/48GiB存储上限保持。没有正式PPO/Student/1M或5M启动。

## 未关闭项与下一项单一主要干预

旧512更新后权重丢失永久保留为[原保存勘误](DUAL_TRACK_FB_CHECKPOINT_ERRATUM.md)，
本轮包不能补回旧模型。新停止包保存/读回/继续采样可靠性有正证据，
但训练更新严格一致性仍未关闭，本轮不能发放长跑资格。

下一项最可判别的**单一工程诊断**：固定完全相同replay/expert mini-batch和RNG，
将一次agent.update拆到六优化器的前向、梯度、step、target更新边界，
定位首个不同张量；保存构建外执行状态及键/参数绑定顺序，
不先更改reward、噪声或边界。现有指标提示应从discriminator路径优先查看，
但这是定位顺序，不宣称它已被证明是根因。

该诊断未启动；本轮到此停止。原生源/接触质量的后续研究与这个工程阻塞应分开归因。
原始证据位于本轮私有stage的d2-audit、control/resume各版本、update-diff与negative-tests目录。

补充：实际更新恢复测试为eager配置，compile/cudagraph恢复未验证。
最终新入口补上真实外部stop返回后立即保存退出的gate，CPU回归检查块内13/1024游标；
未再追加完整物理/更新回归，不以该补丁发放长跑资格。
