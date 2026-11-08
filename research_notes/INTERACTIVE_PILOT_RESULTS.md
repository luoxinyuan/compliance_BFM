# Issue53 — 路线A交互pilot与FB影子提名结项

2026-11-08。本有界阶段已停止，无训练/队列/Student PPO。[Issue53](https://github.com/GaoE05/robot-research-bridge/issues/53) · [预登记/有效性勘误](INTERACTIVE_PILOT_METHODS.md) · [完整分项/身份/成本](interactive_pilot_evidence.json) · [学习曲线](interactive_pilot_development.png) · [视频](INTERACTIVE_VIDEO_INDEX.md)。Issue52独立Expert PPO保持结项，旧I/C0/T30/S0及全部失败不覆盖。

## 结论

1. **Teacher在有限Student访问域确实有用**：早期2秒接管后交回同一S4000，整段接触55.217cm/9.218N→7.706/5.539；新验证的post-onset1.4秒窗口2.688cm/2.775N→2.319/2.098。不是普遍恢复率，较晚长距6秒源域未完成合规复验。
2. **旧数据续BC已取得真实有限学习收益**：在同一S4000起点、fresh Adam/lr1e−4的额外500更新模型上，新6cm/59cm继续/停止均改善。同一受限Student，未输入伙伴真值、未来或时钟；不是PPO，也不是新增交互数据收益。
3. **本轮未建立交互聚合的可信优势，未启动Student PPO**。我在跨reset恢复实现中遗漏接触callback闭包，导致原post-onset两组anchor偏移。2000监督更新已经消耗，其中交互臂混入40个受影响状态。原始训练/物理结果保留并隔离，不推广该模型，不把失败成本扣除后偷偷重训。旧数据臂完全不含这些新标签，冻结评价可保留。
4. **FB当前影子提名no-go**：20个同reset配对、40条额外候选，0次带来边际收益；F-top4仅5/20保留原7池物理最优。原7候选正式Teacher不改，不否定整个FB/reward prompting。

## 当前信息与控制合同

冻结BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；原PD、Direct256D/norm16、10/50/200Hz、K80D8/10N、原物理reward不变。原7提议/2.4s搜索只读当前伙伴位置/速度，恒速外推；每次实际只执行首100ms。Student状态查询冷phase0/无旧计划，此后相位只累计本段实际执行命令，不复制另一Teacher轨迹隐藏历史。

Student1639D=465本体/动作（关节角29、速度29、gravity3、角速度3；上一低层action29；4帧93D历史）＋真实上一z256＋100帧三槽XYZ理想力900＋已知安装18。原BFM `privileged_state/max_local_self`、time、anchor/error、伙伴速度、active真值、任务ID、候选kind/phase均不拼入actor。model normalizer按key独立。具体源码审核/哈希见evidence。ideal-force假设不等于G1有已验证腕部传感。

同权重合法z直接监督；固定真实上一z与5个50Hz观测作动作检查，不添加近恒等投影、不把固定观测重放叫可微物理。PPO代码是条件准备，从未执行，不能写成学会或失败。

## Teacher恢复与真实后果

| 真实接管起点/窗口 | Student继续：cm/N | Teacher窗口：cm/N | 后果与边界 |
|---|---:|---:|---|
| normal72，4.7–6.7s | 10.298 / 6.306 | 5.835 / 5.050 | Teacher之后交回Student，整段7.706/5.539，真实行走保留 |
| short08，5.4–6.8s（修复后） | 2.688 / 2.775 | 2.319 / 2.098 | 交回Student后整段仍10.043/6.152，对照10.879/7.097；短窗有用不等于持续保持已解决 |

只替换normal4.7s首100ms，随后Student整段16.868cm/8.098N，也优于原55.217/9.218。首个同BFM观测控制子步q_PD RMSE约.237°、单关节最大.715°；其后状态已经不同，不能继续当同状态逐帧因果比较。控制/身体历史敏感值得研究，但没有由此证明唯一是上一z输入问题。

早期干预发生在onset前；原onset_actual合同在5s仅对齐一次，因不同前缀实际腕姿略不同，后续世界anchor起点约差(.382,.850,1.529)cm，非人为追随机器人。早期20标签通过fresh连续任务重放：特权/受限输入、上一z、q/qdot/q_PD、实际力、active-anchor全部误差0。尚未建立所有偏离状态的有效反馈域。

## 发现并修复的实现缺陷

原`rolling_snapshot.BranchState`按同reset分支设计，本轮却保存跨多个reset的快照后恢复。它恢复数值、MjData/solver、BFM/history，但没有恢复已安装的函数及其`spec/base`闭包绑定；指纹与保存前缀相同并不足以保证下一tick相同。原short5.4和normal6.0分别产生约(0.116,.683,.928)cm与(.382,.850,1.529)cm非预期anchor偏移。原3/3资格和对应40独立状态标签撤回，原日志/model/gate不覆盖，另存erratum。

新局部[CrossResetState](interactive_snapshot.py)恢复callback对象图、原spec对象、原点cell、cfg/history/initial/prefix及完整MjData；不修改共享接触公式/BFM/PD。[验证](interactive_snapshot_check.py)在pre4.7、post5.4、post6.0三个真实状态，比较不间断Student与先运行另一真实回合、再恢复继续的结果：所有physics/control字段逐值一致。旧类负对照真实分叉保留。历史Issue50/52读取的用法是每个reset内立即分支，不能据本轮跨reset误用否定它们。

修复后只用剩余14查询验证post短请求，查询总数96不超限。原早期有效20＋正确post14=34个去重状态（仅2个Student源起点，其余为相关Teacher接管后继）；分别保存真实Teacher执行、反事实查询与预测，预测标签0。正确34标签**尚未训练**。

## 匹配监督和有效Student控制

原计划从S4000共同起点，同seed5301、网络259456个actor参数、归一化不变、batch128、fresh Adam/lr1e−4，各1000更新。旧臂只用原311个真实Teacher train标签/3回合；交互臂35%替换为恢复窗口数据，按窗口采样、首点去重，避免单点独立成组高频重采样。两臂更新数匹配，不声称标签/查询/计算成本相同。

原跨reset问题使交互臂数据有效性不足；其500开发正常11.255/6.863、短请求7.394/5.581仅保留诊断，不宣称方法赢。旧数据臂在0/500/1000、3条件开发规则中选择500，随后冻结评估原预封存6cm与59cm共同前缀族，不按终检改选：

| 条件 | 原S4000：cm/N | 旧数据BC500：cm/N |
|---|---:|---:|
| 新6cm | 15.656 / 9.178 | 3.200 / 2.671 |
| 新59cm继续 | 49.505 / 9.724 | 10.794 / 7.201 |
| 新59cm停止 | 42.808 / 9.685 | 9.127 / 5.916 |
| 24cm回归 | 13.760 / 8.528 | 5.874 / 4.912 |

新59cm停止BC500为9.127cm/5.916N，对照42.808/9.685。新6cm无10N饱和，root全回合约+1.45cm、足部路径1.16/.78cm；新59cm继续移动段root+47.53cm，有真实支撑转换。同一checkpoint，而非按任务拼模型。stop/continue在8.4s前各模型880physics/220BFM步逐值相同，之后因真实需求改变才分叉。

依然存在质量边界：新59继续37.24%接触tick限幅，保持期11.958cm/8.573N；停止保持期3.704/3.238明显更好。完整回合yaw约−18.43°/−22.62°；继续释放后root仍+17.95cm，停止释放后约−.15cm。解除后不评价旧anchor跟随，但多余运动与偏航照常记录。无接触BC500左足最大偏移5.89cm（原S4000约4.97cm）、37个5ms单足低载tick，不能称零无请求动作。新长距q_PD20ms目标峰约60.24°，实际q相邻5ms峰1.56–2.10°，二者不能混报。所有有效冻结回合无安全提前终止，不等于自然性已解决。

这证明额外旧数据BC可以改善，不证明只有训练量不足：warm-start/较低LR/新Adam也构成续训配方；没有额外seed统计，不宣称continuous-body或general泛化。新6cm与59cm继续/停止是3配置、2个前缀族，不是6独立研究任务。

## FB与reward-z诊断

20状态来自4条Student回合，包含12个接触、8个precontact状态；原S4000未形成walking，不能外推到所有正在行走状态。原7池物理排名与两项F提名替换后的7池比较，未静默把9候选更多预算当收益。原6reward-z去重后提名，不新增reward/seed/buffer。F使用原机器人全观测、第一插值z及低层action，保守head内积于固定forward015投影前embedding；不是2.4s变z/外部接触的完整return。F-top4物理最优保留5/20，因此本轮无依据减少物理查询。

单独补测实际Student已有运动的7.3s状态（root vx.283m/s，单足卸载；完整真实前缀q/qdot/q_PD重放0）：查询23.56s，原池reward015被选，但预测18.65cm/9.55N，未资格化为有效Teacher标签。不能写成reward-z任何状态都不被选。

五类假设分别归档：
- 已知有效Teacher按外部原contact reward移动−2.487/s，Standing−5.626/s，外部目标认可实际反馈；同一机器人状态prompt则Teacher均值.1168、Standing.1337，不能混为同一reward。
- 抽查原150000 successor索引的15个高分片段；forward015前5个不同env/time-bin中心.957–.984，但前后1.2s平均.018–.068且无reset。瞬时高分不保证持续行为；这是有偏小样本，非全replay统计。
- B聚合可能混叠多模态/时序，但本轮没有重推断/干预聚合，不能认定唯一根因。
- entering与maintaining域可能不同；moving单点已有reward池被选，但实际恢复未知。
- 当前F排名未对接触物理后果显示实用性；0/20只暂停当前任务域提名器，不否定整个FB。

## 成本、身份与复现

原7完整查询p50=24.086s、p90=24.472s；每条2.4s物理候选p50=3.364s；F提名p50=0.02730s；完整状态保存/单次恢复p50=0.0221/0.0226s（非可相加的inclusive剖析）。原7查询主要成本仍为真实物理候选，每个有效首动作约24秒；20/14次闭环查询才资格化2/1.4秒恢复窗口。原字段BFM_calls实际记控制子步，已校正：712候选、85440控制步、170880 actor调用（含standing诊断），F另记。不是实时部署Teacher。

实际96完整搜索、40额外FB序列、341760候选physics步；监督2000、Student PPO=0。累计job含300秒CPU预留 **4375.24s/12000s**，视频核验前阶段输出约 **1.037GiB/8GiB**。失败/停止评价全部计入；中断评价已保存22960physics步，未保存尾段只给0–2160上界，不假报精确。初始化/渲染/后处理成本计job-wall，显式任务tick与底层初始化范围分开。完整ledger见evidence。

| 运行 | 精确runtime |
|---|---|
| 原Teacher/FB诊断 | 6f658a85e77fd1f1ac0d3d284844c856c183d0d2 |
| 两臂监督 | 3e2a4f792df87e52397b99f7f269d41d98af26f9 |
| moving成本/中断BC评价 | f61591a7b3cbc1897fc2328ed5304811882711d0 |
| snapshot正确性验证 | 59fa001923a734c8f6b021b92e6c0c93fada4c17 |
| 正确post-onset复验 | 8b5b9105b888bbedacaafdc3ad611161ab4bfe1d |
| 早期20标签连续验证 | b5bf8043227c8120ab8cad8ace54cf0091cf26a1 |
| 有效旧BC冻结评价 | 4ae78adb1aa5169d2ccc38578e5a192e9c5723bb |
| 最终分析/渲染 | 7617d96298c4d19ec94adbfc90a11a3a7140fc14 |

S4000 `8f5e7287e974deedba1a8d841dd65d7692e944e278bd4a091f1ec224f167a479`；旧数据BC500 **`a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`**。新S-init `f414ea3fd12a4b90ae8a03ad273672dfd6c18e3d848a6d6e2e0efde5384561ba` 与历史S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`身份核验分别保存，不混同。

私有stage `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/interactive-contact-pilot-20261108`。checkpoint在`runs/bc-fit-01/old_data_only-500.pt`；正确34标签在`runs/final-analysis-01/corrected-new-labels.pt`；所有原始预测/实轨/失败/中断/校验各自目录不覆盖。公开code不含weights、raw轨迹、大buffer、密钥。复现需同checkpoint与原生资产，先读protocol/ledger，用独立授权输出；旧6f代码仅作失败复现，跨reset使用新CrossResetState。不要直接在已消费stage运行旧fit/PPO入口。

## 视频与下一决策

Mac已实际核验：`~/Projects/compliance-bfm-diagnostics/videos/interactive-contact-pilot/`，仅3个MP4+INDEX。01早期Teacher介入的因果对照；02正确post接管与交回后失败；03同一BC500在新6/59cm条件的真实受限Student收益。均1×真实qpos，含完整接触/释放恢复，未混预测、未挑seed，SHA/帧率/时长/全解码/画面检查通过。

**唯一下一建议：** 冻结已在新条件有效的旧数据BC500作为新的有限域S-init，下一有界阶段比较冻结BC与同初始化Student自身小预算RL，稀疏使用已验证域Teacher恢复。不要把重做混合BC完美当永久前置，也不要将当前未完成的交互/未运行RL宣称成功或方法失败。若仍要隔离新增标签的价值，可复用本轮旧数据训练批次与模型，只另行授权正确交互臂；本阶段不自动执行。
