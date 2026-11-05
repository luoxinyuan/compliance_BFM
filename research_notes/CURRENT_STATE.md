# CURRENT_STATE — 新会话先读

最后核对：2026-11-05 CST。**Issue43（BFM latent先验去偏/reward审查）已完成并停止；无训练或下一阶段排队。**

入口：[完整结果](LATENT_PRIOR_RESULTS.md) · [训练与选模](prior_artifacts/training-final.json) · [终检物理指标](prior_artifacts/final-metrics.json) · [Issue43](https://github.com/GaoE05/robot-research-bridge/issues/43) · [Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。长期纲领RESEARCH_OVERVIEW不变。旧Library2026-10-28仅历史快照。

## 总目标与可靠基线

Contact-conditioned whole-body behavior selection/modulation/transition。真实接触与身体状态应决定合适的局部手臂、躯干/root、支撑与步态响应。保持冻结BFM/PD。有限域直行Teacher→ideal-force Student、有限双向转腰等成果保留；不宣称general或G1具有精确力传感。

- BFM SHA 33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
- I (M39-A best64) SHA72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a；T30 SHA c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33。
- S0 SHA0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12；D2 SHA9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5。
- T30→C2→D2→I的实际actor/标签/replay/runtime见prior_artifacts/identity-chain.json。执行无在线CyclicFeedback/M10 Student；高层探索在已学raw均值附近，不是随机重学BFM。

## 本阶段结论

1. 复算43条旧轨迹reward；32条来源/共同前缀物理资格：原生局部伸臂、有限左右转腰、行走成立，额外30cm躯干候选效果差、排除标签。reward认可好的局部解，不能简单说只奖励walking；active阶段的运动经济性代价很弱。
2. A为I旧数据自监督匹配控制，B为多行为初始化。两轮各1500更新/臂；第二轮严格统一I真实前缀仍未修复负迁移。局部改善不等于保住全身行为。
3. A/B各32×768=24576真实RL决策，same10N/reward/std/课程/新critic/Adam。开发0/8/16/24/32后两臂best均iteration0，selected actor与pre-PPO逐值相同；没有可推广的新PPO收益。终检没有被用于选模型。
4. 44条冻结终检/回归：新8cm，A5.174cm/4.298N→B2.993cm/2.474N；B几乎不调脚，确有局部价值。但新55cm A5.557cm/B22.692cm，80cm A6.019cm/B34.851cm，B长距启动严重退化。新arm-ready20cm A4.902/B6.740cm，未证明身体配置泛化。5N长距B48.081cm，物理域敏感性仍在。
5. 44条没有安全终止或双足同时失去支撑tick；q_PD最大51.38°而实际物理步关节最大2.99°，不能混为实际瞬时转角。释放后不再评价旧anchor跟随，完整恢复保留。

## 当前瓶颈与下一决策（未授权执行）

有效局部/行走来源均存在，reward也并非必然把局部排到后面；当前主要证据指向多行为监督的闭环负迁移，32批PPO未修复。尚不能唯一归因为latent loss、隐藏信息、奖励或优化。

唯一优先建议：在B真实早期起步偏离状态核实I接管的有效窗口，以有效恢复短段和局部数据联合形成匹配初始化对照；不能把后期不可恢复状态硬当正确标签。保旧仅限可靠运动域，不冻结全部身体。运动经济性reward独立方案见报告，暂不同时改reward/force/history，也不原样续训。

M41已证实启动失败具有闭环状态/历史积累，晚状态I不一定能救；Issue42已证实旧82cm先有手臂响应、80cm需身体参与。两项历史结论不被本阶段覆盖。已完成训练不代表“减噪无效”或“PPO不能学”。

## 精确身份、成本和交付

PPO runtime982f5bf544d7027fb080cd57036bb6ee36561e13；终检/渲染runtime4f46fdc，完整值见ledger。长期个人分支gaoe05/research；最终文档commit见Issue43结项，非main。
A selected SHA7cc9f2bda24fd1d382d8dc346f0845aa0211e98680eb5a8e38f1dcb20ab86804；B selected SHA4380df4f8566b1bc2ac8d7849a633c6cda7da0c209029f987c24a3f02a07a24e，均为各自PPO iteration0，序列化critic/metadata导致与监督文件SHA不同。

run `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105`。frozen-final-manifest、ppo-A/B、final-eval、videos与所有失败完整保存。49152新增RL、6000监督更新、7748.37记录job秒加约28秒结项CPU及短检查、约4.0GiB。阶段GPU/monitor全部退出。没有Student训练。

Mac `~/Projects/compliance-bfm-diagnostics/videos/latent-prior-audit/`：01局部能力正控制；02 A/B长距成功/失败；03新自然arm-ready状态20cm。3条1×真实qpos MP4+INDEX，SHA/全解码/画面已核验。不是新PPO成功视频，不替换旧I/T30/S0。

接力：先读本文件、报告、Issue43；用户决定下一阶段后再核验资源和模型。不复用PID/空闲GPU快照、不重开本阶段。共享checkout6023c6e clean；不改BFM/PD/权重/main、SoftSONIC或他人任务，不做真机。force-history消融仍未执行。
