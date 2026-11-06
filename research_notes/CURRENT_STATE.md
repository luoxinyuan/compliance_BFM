# CURRENT_STATE — 新会话先读

最后核对：2026-11-06 CST。**Issue46已结项，Issue47正式C0/I接管与PPO尚未启动。用户新提供PPU服务器并授权隔离兼容验证：原BFM推理/梯度、I/C0同状态输出及8s真实MuJoCo检查已通过；没有后台训练或自动排队任务。资源条件已从“没有可用后端”变为“PPU基础链路可用、Issue47完整后端回归待执行”，不可把兼容探针当作C0恢复证据。**

[完整结果](COEXIST_RESULTS.md) · [预登记协议](COEXIST_PROTOCOL.md) · [Issue46](https://github.com/GaoE05/robot-research-bridge/issues/46) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18) · [BFM原生复用最小路线](BFM_REWARD_REUSE_ROADMAP.md)。RESEARCH_OVERVIEW是长期纲领；ChatGPT Library2026-10-28是历史快照。

## 科研目标与可靠基线

Contact-conditioned whole-body behavior selection/modulation/transition：依据真实接触与身体状态，合理组织局部手臂、躯干/root及支撑响应；不预设先手后脚，不以root走远替代任务合适性。冻结BFM/PD。有限直行Teacher→ideal-force Student、原生局部伸臂、有限双向转腰均已成立；没有验证真机精确外力感知。

- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- 实际行走基线I `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`。
- T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`；S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`。
- D2 `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`。
- C0局部有效，72cm持续需求下也会真实迈步，但启动及持续推进迟到；不再仅概括为长距失败。C1恢复walking却失去局部优势。Issue45 F/R均未稳定共存，冻结I参数不等于闭环保旧。全部旧权重/失败保留；不按任务拼模型。

## Issue46实际结果

1. 原reward同条件复算认可C0局部和I长距有效解，未改reward/网络/历史/绝对z/BFM/PD。10N主物理、K80/D8、原观测尺度；5N回归。两个seed同I actor，独立fresh critic/Adam，std=.008，各32×768；无新增监督初始化/Student。
2. 两组固定开发规则均选择iteration0。末轮分1.619/1.612，I1.500；预登记联合延长条件不成立，未追加训练。best两个文件不同critic SHA，但均值actor/归一化与I相同，最终评估去重，不能算新学习成功。
3. **部分正结果保留**：同一4601末轮在新6/24/72cm误差4.190/4.617/5.478cm，I4.452/5.290/7.632cm；72cm真实力5.877→4.511N。4602长距改善但新6cm退化4.790cm。不是两个seed稳定共存。
4. 停止取舍：新共同前缀停止4601为4.956cm/4.038N，I4.387/3.602；旧早停止6.062/4.967 vs4.707/3.889。退化发生在仍接触的移动/保持阶段，不能用释放后自然收步解释。共同前缀7.37s前实际physics/命令一致，分叉后可反馈调整，但不是最优行为分配。
5. 动作质量限制：新6cm4601足部路径.33/.75cm，远少于I20.80/.80cm，却伴随root降低6.77cm；世界腕相对肩X仅+.90cm，C0+4.32cm且误差仅2.293cm。减少脚步不等于自然伸臂已经成功。继续保留XYZ、相对运动、滑移和连续性。
6. 242完整审查回合无安全终止；seed4602中距有共10个无支撑标记tick（单段10–15ms），未隐去。最大q_PD20ms跳变68.29°不等于实际q5ms3.60°。无接触稳定，释放后不考核旧anchor跟随。
7. 探索噪声能改变同状态BFM动作，未证明std太小；但单步方向覆盖不等于持续有效反馈。raw logprob复现≤.0001221，联合KL≤.01996，critic EV提高不能代替actor成功。每seed208完整采样回合，减速决策仅136/213；覆盖及优化共同限制尚无唯一根因。

## 当前授权：Issue47 C0局部→全身参与（进行中，正式实验未启动）

[详细记录](C0_ENGAGEMENT_RESULTS.md) · [Issue47](https://github.com/GaoE05/robot-research-bridge/issues/47) · [协议](c0_engagement_protocol.json)。[方法定位与自主学习思考](CONTACT_LEARNING_RATIONALE.md)。用户观看C0后更认可其先局部、后身体参与的组织，当前优先研究delayed engagement；旧原生reward-prompting建议改为后备工具，未执行。

讨论结论：继续既定C0/I接管有价值，用有限同状态干预区分条件运动能力、启动/维持反馈与恢复域；证据支持后才进行原授权PPO。最终主线是接触/身体状态条件的自主行为选择，不是人工goal答案拼接。人工goal用于资格、初始化和探索；BFM风格先验及PPO均不保证任意状态/latent序列自然或全局收敛。此次仅文档澄清，不改实验变量、预算或封存设置，不启动多部位/新力范围训练。

进一步讨论已写入CONTACT_LEARNING_RATIONALE第8节：原生候选可提供有起点的探索，真实源状态/共同前缀需求变化应常态覆盖，后续明确研究“蒸馏初始化→部署观测Student真实闭环RL→可靠专家聚合”。均是待验证方法候选，不宣称std=.008够不着已被证实、不保证原生候选便宜且合格或Student RL必胜；严格最差任务/频繁回滚也需防止冻结学习。当前Issue47协议/预算仍不变；2026-11-06另完成用户提供PPU的基础兼容检查，未自动启动原实验或新方向。

C0精确SHA `ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c`，`runs/prior-recovery-20261105/correct-C0/C0.pt`。I起点、750条六类示范、2000监督更新；没有后续PPO，也没有在线先手后脚规则。仍是1088D特权Teacher，不是部署Student。

已完成保存72cm轨迹100ms/50Hz分析：C0首次持续10N约8.18s，左足约10.22s卸载、10.35s落地，前进约5.90cm但root几乎未推进；约12.43s才建立持续root前向速度。最初至7.5sC0误差3.36cm/力2.93N，优于I4.77cm/4.25N；后续明显追赶不足。C0 7.5s目标z已偏离standing约52.6°，不能说高层一直没发命令。尚未隔离高层、BFM反馈和外力贡献，也没证明手臂已耗尽自然可达余量。

待执行：C0/I完整历史双向接管（72cm的6/7.5/9/10.3/11s及少量近距/共同前缀）；仅有效恢复证据成立后，C0起点两seed4701/4702各16×768，开发共同收益可等预算延至32。新留出组合已固定。预算14000累计job秒/49152决策/9GiB/最多3张实际空闲GPU。原BFM/PD/Direct、reward/std/历史、10N/K80D8不变；5N回归。

历史NVIDIA资源检查（2026-11-05，不能作为当前快照）：初查全部卡有任务；1–6短暂空闲，但发车二次核验出现新多卡任务，故未发车。实际新增GPU任务/物理回合/BFM推断/RL/监督更新均0。不要CPU替代BFM、不要共用他人GPU或沿用这次快照。接管脚本仅语法验证，未声称物理通过；PPO尚未实现为已运行任务。Issue47开放，本阶段未完成/未科学否定。恢复时先重查资源，再核对协议与基线历史物理一致。

## PPU迁移检查（2026-11-06，独立兼容探针）

[结果与复现](PPU_COMPATIBILITY_RESULTS.md)：PPU-ZW810E约95.6GiB，厂商PyTorch2.9.0，独立venv保留原MuJoCo3.3.3/NumPy1.26.4/joblib1.5.1；原完整BFM及I/C0 SHA核对通过。18个NVIDIA保存状态的BFM action最大差5.66e-7、q_PD差2.73e-5°，高层I/C0 raw差2.24e-8；真实latent梯度有限非零。新standing +X3.5cm兼容回合8s/1600物理步，力1.618N RMS/2.049N peak，tilt最大2.79°，无关节越限、力矩饱和或安全终止。0新增RL决策/研究监督更新。它不是原72cm接管/新PPO成功，不取代Issue47正式回归。

新服务器运行根目录`/root/compliance-ppu-20261106/`，私有模型资产`/mnt/workspace/compliance-ppu-20261106/`；Mac摘要`~/Projects/compliance-bfm-diagnostics/ppu-compatibility/`。源码快照仍`cc5765522826cbe130635321f0a3056362faa184`，仅新增隔离诊断入口；原服务器实验与共享源码未动。SDK必须按会话加载；缺GL库私有解包、旧joblib读取失败与测试误用no_grad的失败均保留。测试已退出，PPU最终空闲，无自动发车。恢复Issue47前先复查资源，运行冻结C0/I的原任务后端回归与同后端完整前缀一致性；跨平台逐位物理相同、PPO并行吞吐尚未验证。

## Issue46历史身份与成本（不表示当前GPU空闲）

训练runtime `f6ca5c5f6c4de205b3457bef4a024345e2105009`；最终评估/渲染 `057f5c6bd302185097770c3c1ed4d10dfb2ce109`。长期个人分支`gaoe05/research`；最终文档commit见Issue46链接，勿混作训练runtime。

- 4601 last32 `8fa4759d3766568dee8fbba0367e772a97977694d19a503868708efa6d8c9228`。
- 4602 last32 `e9a1972166b25e3b915b8d75e97d3377e540583e215d91f269e8dba4550310c9`。
- [完整模型清单与服务器路径](coexist_artifacts/final-frozen-models.json)；[资源账本](coexist_artifacts/resource-closeout.json)。
- Run：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105`。初始/每8批候选/best/last、原始rollout、所有失败均保存。
- 总49,152高层RL决策、194实际Adam更新；0新增Student/监督更新。7711.53 GPU-job秒＋165.02分析秒＋180秒保守CPU余量，约4.24GiB，低于授权18,000秒/12GiB/98,304决策。训练两独立worktree未修改。
- Mac：`~/Projects/compliance-bfm-diagnostics/videos/feedback-coexistence/`，仅3个1×真实qpos MP4＋INDEX；6cm、72cm四模型对照，以及同4601停止/继续。SHA/时长/完整解码/视图已核对；未挑seed重跑。
- owned进程均结束，GPU2/5/6释放；共享checkout仍`6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`干净。未改BFM/PD/原权重/main、SoftSONIC/他人任务，未做真机。

新会话先读本文件、RESEARCH_OVERVIEW、C0_ENGAGEMENT_RESULTS及Issue47，再按需回顾COEXIST_RESULTS/Issue46；核验实际运行/模型SHA与新授权，再检查GPU/CPU/RAM/I/O，不复用旧空闲快照或PID。历史Issue44/45、M41仍有效，不能用这轮局部正例抹去停止与跨状态失败。
