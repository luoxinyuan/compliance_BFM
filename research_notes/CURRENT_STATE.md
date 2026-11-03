# CURRENT_STATE — 新会话先读

最后核对2026-11-03。**Contact Geometry → Translation / Rotation（Issue37）有界阶段完成，全部owned任务退出，未启动下一阶段。** Issue36及此前历史完整保留；M11 continuation仍停止，不重启旧A/B。

入口：[研究纲领](RESEARCH_OVERVIEW.md) · [本轮完整报告](CONTACT_GEOMETRY_RESULTS.md) · [Issue37](https://github.com/GaoE05/robot-research-bridge/issues/37) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。上一轮[HEADING_TRANSFER_RESULTS](HEADING_TRANSFER_RESULTS.md)/Issue36与[T30迁移](T30_TRANSFER_RESULTS.md)/Issue35不覆盖。

## 可靠基线与本轮科学结论

- 有限域Direct Teacher→ideal-force Student直行、减速、保持与释放已经成立；**S0和Teacher30继续是基线**。不再以释放后旧anchor几何距离否认正常收步。准确外力的真机来源未知。
- 旧左路径末端切线11.769°、同前缀改向5.833°；旧heading项确实小，但不是唯一原因。新单腕对称曲线±22.55°终检：A20左误差8.682→6.629cm、力4.249→4.141N，但pelvis heading RMS16.89→18.60°。B20左位置也改善却朝向更差。**手沿曲线跟随≠身体可靠转向**，两组新PPO不替换T30。
- 新增真实刚性torso双点，locals `[0,±.13,.15]`m，各2.5N、sum norms≤5N，与单腕5N匹配总作用预算。共同SE(2)伙伴框架，逐tick累加force/COM torque一次并正确减除；每点历史保留身份与坐标。它是保持接触Tether，不是单向按压。相同总力不等于相同力矩/做功。
- 一个共享1639维ideal-force Student已训练4000更新（8独立回合/1227标签），同时能执行旧腕和双点。新24cm双点平移Standing/T30/Student为14.761/3.960/4.740cm，sum-force RMS4.556/3.333/3.495N。新26cm/−10°组合Teacher/Student4.718/4.178cm、4.226/4.018N，但Student实际torso转角约+0.66°，**组合收益主要是平移，不能称旋转通过**。
- 共享Student旧正常6.530→7.280cm、早停止7.330→7.719、晚6.892→7.646、局部3.680→4.445cm，均保留失败，不能替换S0。无接触只有毫米级root漂移，没有完整新动作自动播放。
- 当前原生motion左右调用在纠正各自period/进入退出后仍不适合作为原地旋转源；不能据此否定全部BFM。仅两项明确native waist-yaw goal正控制证实小幅左右转腰可执行：±15°下1.576/0.678cm，sum-force2.535/1.089N，torso约+9.23/−14.52°。主要腰相对pelvis约±10°，**不是新迈步转身能力**。
- 将goal与旧T30行为混合做4000更新Direct监督初始化，右向有部分效果，新−12°为1.072cm/1.696N，但左不足、旧直行退化至53.310cm。该研究模型未推广。专家状态q_PD平均对齐约0.044–0.078°，却在接触前已闭环偏离；离线拟合不等于保住稳定域与运动能力。详见first-divergence和视频3。

## 当前瓶颈、下一决策与未执行项

**首要瓶颈：** 不是仅把heading权重加大或继续加力；已有有限旋转能力在Direct闭环中的获取、保旧与真实接触条件调用仍不可靠。旧T30对纯力偶近乎无响应，新PPO未学出；goal监督初始化暴露接触前稳定域/分布敏感，不能唯一归因网络或数据量。

**唯一推荐下一步（尚未执行）：** 用已验证的小幅原生转腰作为行为源，针对接触前偏离做保留无接触稳定域和旧运动能力的同状态动作约束、有限真实访问状态聚合，先得到保旧左右旋转反馈专家，再迁移到现有共享contact slots。不要扩force cap、更多点或大角度迈步，不无界搜索latent。

已完成：两组40批PPO、原生motion/goal有限资格、共享Student训练、goal监督初始化、冻结终检、视频。**没有**继续聚合混合质量的接管标签、把goal研究模型升级基线、启动下一轮PPO/Student、Bridge/watch或真机。Teacher接管4次前缀物理逐值相同，部分改善、部分变差，未笼统判全部标签有效。

## 精确身份与数据

长期个人分支：[gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)。文档/整合commit不等于运行commit。

- PPO runtime `b96fe456fb77d3d98602772d98d23652d0613c42`；共享Student `596b598749db82d17b2bc97e08dec76d013095c1`；goal初始化 `704de7e00736cc6845222672c17cc6e5350d8770`；终检 `940377c1fb3271abb7ca3ec5aae2f04a70e9c422`。所有source pins见[resources](geometry_artifacts/resources.json)。
- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- T30 SHA `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`：`bridge-v2-owned/runs/stop-sensing-20261103/B1-train-02/job/best.pt`。
- S0 SHA `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`：`bridge-v2-owned/runs/direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt`。
- A20 SHA `189e2fa24b6a2fdb3cecb8c14fb4f51071334d27ab487fab73e81248e3fbd610`；B20 `62004810623df6ef8d5164ed04d3793cec5998fc57c6486012a5b5abdfc0fce8`。
- 共享Student SHA `c055315215615f496f86f22fbbf7df5fbc20f0dc119892cf01cbe03c29157001`；goal研究模型 `2059302f4b72c204b020309f316d17d46dd8cb7718d1ceaa655185eca592941e`。
- 本轮服务器根 `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/contact-geometry-20261103/`；A/B模型分别在`A-train-04/job/best.pt`、`B-train-05/job/best.pt`；共享`C-final-08/job/shared-r0.pt`；goal`C-goal-10/job/goal-initialized.pt`。初始/末轮/失败全部保留。[结果索引](geometry_artifacts/results.json)、[模型SHA清单](geometry_artifacts/models.json)可审查。
- Mac `~/Projects/compliance-bfm-diagnostics/videos/contact-geometry-20261103/`：仅3条MP4＋INDEX。左曲线收益/heading局限、双点共享Student、左转腰来源/学习失败；真实qpos、1×、完整接触＋5秒释放恢复，SHA/时长/解码/画面已验证。旧视频不删除。

## 资源、安全与跨会话接力

本轮累计8990.33受监督实验/渲染job秒（含失败），约4.29GiB；RL61440高层决策，新增Expert监督4000及Student4000更新。授权上限21600秒/15GiB/最多3卡，**余额不是下一阶段授权**。使用GPU1/2/3，结束后均退出，其他用户GPU0/5/6任务未操作。共享repo保持clean，BFM/PD/权重未改；所有工作在个人worktree，没有push main/强推。

Student input仅本体/动作/自身z记忆、每点理想力历史与已知mount身份，没有anchor、伙伴速度、目标朝向、active真值或任务clock。未知G1传感配置仍未知，不等于实机部署。双点同时接触已测，但未验证任意接触配置或同回合腕↔躯干切换。

新会话先读本文件、纲领、Issue37与协议，再核对Git、模型SHA、ledger/进程及实际资源，不重放已消费任务。普通工程按新授权自主处理，不改main/共享BFM/PD/权重、SoftSONIC或他人任务，不下载大资产，不做真机。Library中2026-10-28 PROJECT_STATE/SESSION_HANDOFF是历史快照；实时入口以GitHub和本文件为准。
