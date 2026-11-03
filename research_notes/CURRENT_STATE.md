# CURRENT_STATE — 新会话先读

最后核对：2026-11-03。**上轮Direct迁移阶段已完成；新授权停止覆盖/传感阶段已启动；结果尚未验证。** 旧M11 continuation仍停止；A160/B114不是完整等预算终检。

## 新授权执行入口

[Issue #34](https://github.com/GaoE05/robot-research-bridge/issues/34)、[STOP_SENSING_PROTOCOL.md](STOP_SENSING_PROTOCOL.md)和stop_protocol.json固定本轮计划。主线为匹配更新的停止覆盖对照、Teacher停止分布PPO、重新训练无显式力Student及力观测压力。预算新登记18000累计进程秒/10GiB/最多3GPU，不能套用旧账本。未来椅子/多接触/全方向demo不在本轮。

运行身份：runtime `32c87b4db3aba007e4d6cf15ed5495cdffdfcb75`；新账本 `bridge-v2-owned/runs/stop-sensing-20261103/compute-ledger.json`，A-prepare-03 GPU1/PID1845112，B1-train-02 GPU5/PID1845030。快照不能当未来空闲/重启依据；C等待冻结停止数据。A-prepare-01稀疏检出缺源码在加载前失败，完整记录后仅展开新worktree，不改BFM。

## 上轮已验证科学结论

目标仍是Contact-Conditioned Whole-Body Motion Transitions / General Compliance。纲领见[RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)，完整历史见[Project Master #18](https://github.com/GaoE05/robot-research-bridge/issues/18)，当前阶段见[Issue33](https://github.com/GaoE05/robot-research-bridge/issues/33)及[TRANSFER_RESULTS.md](TRANSFER_RESULTS.md)。旧Library2026-10-28快照不是实时状态。

- Direct初始化已有可用右腕直行反馈，不在线调用CyclicFeedback/M10 Student。旧PPO15单条件收益未保住邻近条件，见[CLEAN_EXPERT_RESULTS.md](CLEAN_EXPERT_RESULTS.md)。
- **本轮Teacher→ideal-force Student最小迁移成立。** 两个冻结Teacher各824标签/4000监督步，合法z直接蒸馏，无多余逐状态投影。Student只读本体/真实动作、力历史、固定腕部位置和自身命令，不读anchor、伙伴速度、clock、active或Teacher隐藏状态。未验证G1实际力传感。
- PPO15的纯模仿Student：新慢条件7.407cm/3.990N，新快6.912cm/3.913N；Teacher对应6.243/3.648和8.966/4.198。不是每个条件都优于Teacher，不把最终测试用于选择统一部署模型。
- 真正同前缀伙伴停止：8s前1001个物理点完全相同，Student在8.10s改变z，最终root X由.812变.503m；误差6.596cm/力3.842N。证明响应物理反馈，而非始终播放相同z轨迹；不区分显式力与本体反馈各自贡献。
- 初始Teacher Student的短行程启动/较早停止仍失败（16.263cm/4.435N）；该条件同时改变加速度，不能只称停止失败。同前缀停止为7.136cm/4.184N。
- **窄分布Teacher PPO仍未可靠升级。** B1/B2各15360决策，B1选10批，新快8.111cm/4.284N劣于初始化7.455/4.055；B2最优仍初始化。全部模型和负结果保留，不无界续训。
- 一次聚合实现范围错误已更正：有效7s接管不能验证其后所有状态。旧r1留作未充分验证结果；修正版仅接受实际7.10s一个查询点，固定2000更新，未证明聚合独立收益。PPO15接管未改善，不聚合。修正版原终检已暴露，明确为复测，不称新盲测。

## 已有瓶颈与本轮优先级

迁移接口可用，主要瓶颈仍是Teacher优化稳健性及恢复覆盖，而不是缺少可训练Student。无接触漂移约3–4mm，但局部15cm请求仍有约14–15cm root移动；部分q_PD峰值36–58°和短行程失败保留，不称自然全身协调已完成。

本轮用户已调整优先级：先区分停止覆盖与额外训练量，并检验传感约束；raw-z与action/q_PD保持保留为有依据时的方法对照，不再唯一主线。保留上轮迁移成功；Teacher未完美不阻塞Student优化。当前A采集/匹配更新与Teacher训练已启动；多接触/椅子仅远期候选。

## 可靠旧能力与边界

M5–M8有限接触LPP/共享策略和访问聚合、M7同回合转接、M8下沉保留；M9主动行走/有限转弯、M10任务适配行为仍是可靠历史对照。M1–M4中间姿态/恢复失败、M11均值执行失败及提前结束的continuation完整保留。新直行模型未证明保留全部旧腕/躯干技能，不替代M8/M9B/M10。未开展真机、PPO-vs-LPP正式比较或general compliance验收。

## 身份和位置

个人长期分支[gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)，报告/文档HEAD以该文件Git记录为准，不混作runtime。

- 本轮初次训练runtime `f736c9a34b1791eed5a24f9970c87f2383b8547a`；查询范围修正版 `2d6bfdd71ae960459c26f2a19a8eaad9007ee208`。
- 冻结BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- Direct Teacher初始化SHA `a86ff5f9ca71ed2ae8dc95ceeab0ac980f82ab6ed8d3759c039eb03a62efe488`；旧PPO15 `92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe`，源runtime05b313e。
- PPO15纯模仿Student SHA `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`；初始化Teacher Student r0 `97235f33dc3ac3fd9d21eefbc1c57197d7b86126fa957857b9ee58a486fe8303`；修正版`8ff5a5435702738b39df91340f8278720e7f850cab97d5c5479bb1e12d5738ec`。
- B1选10批SHA `d18529b96b8343ac6bb3d0484de5fb8f29934bed28af71ddec4709dcfffa6bd0`；仅研究候选，不推广。

`P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`，SSH `pro5000`，`source $P/env.sh` / `$P/.venv/bin/python`。本轮`R=$P/bridge-v2-owned/runs/direct-transfer-20261103`，A原始`A-prepare-01/job`、修正版`A-final-04/job`、B1/B2 `B1-train-02/job`/`B2-train-03/job`、B1终检`B1-final-05/job/evaluation`、同前缀停止`A-final-07/job`。每项launch/ledger/model SHA齐全；原始A wrapper漏记监督次数，实际以training.json统计，报告已纠正。

Mac仅4条关键视频+INDEX：`~/Projects/compliance-bfm-diagnostics/videos/direct-transfer-20261103/`。旧clean-contact-expert、m10-task-appropriate和m11视频不变。新视频实际qpos1×，已同步并校验。

## 接力与权限

1. 先读本文件、纲领、TRANSFER_RESULTS和Issue33最新回复；报告是完成事实，不是下一轮启动指令。
2. 核验ledger/result、精确PID、runtime与工作树；status.json可能保留末批running快照，以completed result/ledger为准。不要自动重播旧任务。
3. 本阶段上限3GPU/21600累计job秒/12GiB输出；启动前重新查GPU、CPU、RAM、I/O，不能继承旧空闲卡状态或因无GPU改CPU运行BFM。共享盘剩余不是个人配额。
4. 仅个人worktree/输出，代码获准整合个人research；不改main、共享BFM/PD/权重、SoftSONIC或其他用户进程。无自动下载大资产、Bridge/watch、真机或扩大资源。
5. 阶段结尾源码/报告、CURRENT_STATE、Master、准确runtime/模型/失败/视频同步；普通实验不重写全史。

本轮累计受监督计算4984.76秒、新输出2.466GiB；RL30720决策，监督共12000步（含保留的错误聚合2000），已全部退出。
