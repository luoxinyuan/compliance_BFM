# P0-B执行合同 V1（物理评价前锁定）

本包属于Issue58、≤12job小时；不是正式FB1M。原BFM/PD/10N/K80D8/200-50-10Hz、实际q越界.02rad、97D新神经输入和robot-only discriminator不变。

## 变换及独立证据

原29关节的MuJoCo joint range从实际模型读取并保存；PD offset/scale/原clip从当前动作合同读取。原roll目标scale约2.192886rad/action，旧actor std=.2可给出约.4386rad的一轴扰动尺度，超过机械roll范围的噪声不能当作合法目标。旧198真实roll越界及143次接触前终止保留。该尺度计算仅解释风险，不证明新配方成功。

锁定工程候选：先按原clip映射得到q_PD，再只将双ankle-roll目标限制在各自原range内缩.05rad，最后反映射为实际action。其他关节保留原PD裁剪结果。真实q的.02rad边界绝不变；目标内缩也不保证动力学不越界，必须实测。硬clamp在内区可微，外区梯度为0；统计激活率，不能隐去饱和。

交接固定.35s，smoothstep(当前交接elapsed/.35)，原source actor当前同obs/z均值与mutable提议一次混合，再作同一安全变换。它是已知执行器交接，不是force→skill规则。记录raw proposal、混合动作、实际action、q_PD、alpha、激活位和来源。

所有mutable actor采样（真实执行、FB目标及actor优化）均通过同一PhysicalDistribution，实际action进入critic/F/replay。单次原分布采样，不加第二份噪声。该分布不宣称Gaussian密度、不可用作高层raw PPO logprob。

handoff alpha是当前自身执行记忆，以独立metadata key存obs/replay并通过normalizer原样传递，仅执行变换读取；不进入actor/F/B/critics的神经input_filter，不改变已批97D字段。不是任务ID、未来contact onset或伙伴意图。原native source仍完全由冻结原actor执行。

## 事前配对与安全保护

seed58021，standing/goal_arm5/motion_walk_task各10独立源profile（实际hash另核）。各父组完整原生前缀不清history；三个控制分别为原生均值、原step0随机、新安全随机，后两者使用同一随机流。无实际Tether，每次交接后8s真实执行。单一Task/callback对象反复reset，避免重叠callback；原生前缀、低层/5ms轨迹及失败均存，不覆盖同名回合。

风险合格规则：新配方30单元无真实物理边界，且若原step0边界≥4则至少减少75%。任一配对源生成失败都报告资格缺失，不用seed补齐。native质量允许：slip峰最多增加.20m/s；末yaw绝对值最多增加10°；moving前进≥原生均值75%；standing/arm水平root漂移最多增加.05m。所有绝对参考线仍报告，以上是工程退化线而非放宽真实q物理边界。未通过则停止该配方，不进行工程更新或调margin找补。

## 无接触保持与工程更新

若配对通过，工程采集/恢复校验合计最多8192真实online 50Hz transition；首1024 warmup不更新，随后16完整agent.update/1024。先采到6144（80更新），块内6157保存不可覆盖用户停止包；不中断续1011到7168并完成16更新；新进程恢复6157后实际续1011并完成同一16更新。两路径实际新增online采样6144+1024+1011=8179≤8192，保持原更新节奏；重建旧物理前缀另记物理成本，不当新replay样本。保存step0/每1024/最后停止包。每个原生来源的CPR8帧片段额外保存真实执行z/action，按源/父组去重；保持不使用B重推的不同z冒充当时source命令。

对合法原生无接触CPR样本，以相同obs及实际z计算充分交接的mutable均值action/q_PD，目标为当时真实原生action的原PD映射。排除原生target已落在新roll安全区外的冲突点。loss额外系数固定.1，q_PD平方尺度.01rad²；接触训练replay不强迫复制原动作。记录有效域比例及保持项。task actor loss和新保持loss的有效actor梯度也按固定顺序显式求和，避免新分支重新引入恢复数值歧义；数学损失不删。原FB/GP/reward不修改。

若最近10完整回合至少5个真实边界，保存停止；这是更新后即时崩塌保护，不宣称取得学习收益。后续正式1M始终未授权。

更新后另在seed58022的三源各3个新前缀做无接触原生均值/新actor配对；这些是物理验证，不进入train replay，不计学习收益。训练读取完整停止包须重建有界动作函数、交接metadata、真实z/action CPR与全部物理历史，并通过严格读回/继续采样/16更新对照。任何失败保留，不发1M资格。

成本更正（尚未物理发车）：初始全stage只读盘点未独立计时，P0-B准备/CPU保守reserve改为3600s以覆盖该不确定开销（非实测），替代先前900s；不扩大12h上限。所有实际job仍另列精确sidecar。
