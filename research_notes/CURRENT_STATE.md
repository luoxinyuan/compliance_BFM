# CURRENT_STATE — 新会话先读

**最后核对：2026-11-03，clean Expert首轮及冻结终检已完成，无训练运行。**

长期目标及方法见 [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)。[Project Master #18](https://github.com/GaoE05/robot-research-bridge/issues/18)保留完整历史；[窄域Expert #33](https://github.com/GaoE05/robot-research-bridge/issues/33)记录当前阶段，[M11 #32](https://github.com/GaoE05/robot-research-bridge/issues/32)保留旧训练。Library中2026-10-28的PROJECT_STATE/SESSION_HANDOFF仅是历史快照。Bridge仓库`handbook/`先前副本不是第二个实时维护源；以本目录和最新Issue为准。

## 1. 当前状态：clean Expert首轮训练与冻结终检完成；无运行训练

2026-11-03：A Structured与B Direct各20批/15360决策完成，旧M11不重启，无新Student训练。真实完整报告见[CLEAN_EXPERT_RESULTS.md](CLEAN_EXPERT_RESULTS.md)。新接口已去掉在线CyclicFeedback和M10 Student：A学alpha/phase-rate/tangent，B直接合法global z；监督初始化不等于RL收益。

Direct第15批在开发条件6.713cm/3.684N，优于自身初始化7.502cm/4.059N；但预封存新条件8.291cm/4.200N，差于初始化6.954cm/4.066N及旧专家7.445cm/4.036N。第20批开发也退化至8.541cm/4.261N。Structured20开发29.033cm，新条件48.306cm，不能作为有效新专家。首个明确边界是PPO改进未迁移到邻近条件，不能宣称稳定学习成功。

新条件已暴露，不再作为新盲测。保持全部初始/最佳/末轮和旧失败。下一建议：保留Direct可用初始化与接口，预固定窄任务分布和多条件均值开发选择，新的未见组合终检；不是继续当前单轨无界训练或扩大force cap。尚未启动该后续实验。

实际runtime `05b313e6c4a14aff1d2e3ee08432f9d231c6822f`；Direct15 SHA `92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe`，Structured20 SHA `74f2590f6c344ee96e8839c43f5bd1ecf17520e3665954c401aaec95983d00fe`。数据根 `$P/bridge-v2-owned/runs/clean-contact-expert-20261102`，精确checkpoint/轨迹身份见报告及JSON。全部ledger已结束，无保留待执行任务；后续启动须重新核对GPU/CPU/内存/I/O，不沿用旧空闲状态。

总预算14400累计job秒/122880决策/12GiB输出未超限；实际总35328决策包含4608中断，修复了步后才停止导致KL超限的问题，事务回滚测试5项通过。文档集成在独立maintenance worktree，不覆盖运行源码；仅个人research分支推送小型源码与统计。

## 2. 可靠基线、实际结论与边界

- **M10是当前任务适配行为起点，不是万能专家。** 进入walking放手、下沉附带抬手、右转航向较M8/M9C改善，用户看视频确认；新直行改进专家5.643cm/3.296N、Student10.451cm/4.430N。但偏离状态接管、Y/Z负担、连续性仍不足。M8旧腕/躯干及M9B移动基线均保留，不以最新模型自动替换。
- M5–M8已验证有限接触技能的LPP、共享策略与访问状态聚合；M7同回合转接、M8躯干下沉有真实证据。M9有主动行走/有限转弯，不等于可靠全方向牵引。M1–M4的中间姿态泛化、恢复失败保留，不宣称general compliance已完成。
- **M11初轮**开发直行误差/力由M10的12.661cm/4.610N变为7.832cm/4.144N；新直行均值却由8.493cm/4.139N退至37.667cm/4.588N。左右跟随改善但力未降，不是全面成功。M11从M10部署Student先验学习，而非从最好的手写专家热启动。
- 同一冻结M11 checkpoint在失败条件完整采样9.329±1.167cm；仅记忆噪声未解决问题。force/GAE/raw logprob未发现对应硬错误；5N饱和压缩force惩罚，但已有误差/未限幅需求项把失败评得更差，没有force高斯sigma。
- continuation零均值＋同噪声18.119±14.218cm、4.302±.477N，对应学习＋噪声9.329±1.167cm、4.298±.269N（三执行seed，不是三次训练）。学习反馈对误差/尾部有贡献，但力几乎相同，尚无可靠均值改进证据。

**首要瓶颈：均值部署仍不可靠，实际交互力没有稳定降低。** 已停止宽任务续训；clean Expert本轮结果见上方；下一实验尚未启动。不能把这次提前结束解释为BFM/LPP不可行或已经证明减噪无效。

当前Student使用仿真实际Tether力及已知接触位置，尚无真机传感/估计验证；“可部署输入”是信息权限设计，不是已证明准确实机力可得。

## 3. 源码与模型身份（不要把文档HEAD当runtime）

长期集成/审查分支：[gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)。本次在严格快进后增加文档提交；文档提交身份以GitHub该文件的commit记录为准，避免自引用SHA。旧实验分支保持不变；新性能runtime在独立worktree，长期分支只作快进及文档整合：

|用途|精确commit|
|---|---|
|旧M11 continuation运行源码|`c98abd7912e173f1228afaca996ae43ed6009ac1`，`gaoe05/m11-throughput`|
|续训前114批源码（完整保留）|`38c7b25dcb6a3a1124acd9cf44c4eb116364ac60`，`gaoe05/m11-privileged-expert`|
|M11初轮完整报告/源码|`466266385fab2fec9852253d661ecd24cd675c64`|
|M10完整报告/源码|`49639ed29d4cbf513c285633841426579d80d5bd`|
|长期分支整合前的旧HEAD / 共享checkout基线|`6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`|

关键权重SHA256（不进Git）：
- 冻结BFM：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- M10部署Student / M11 prior：`1742ba74e9ec871abf8577eb9a276d7a1e614cd3184330a008dd9c4a7031bcd8`。
- M11初轮best / A、B共同起点：`bc305de6917103c5fbc64a6c6639f91bb517d3c7a48d689e7158739a6168cc8f`，原iteration60/46,080决策。
- continuation开发候选A40：`e979f44d60f16abfe3c3644d64eb410717c1717a3f0a4ebb879843f1313da601`；B40：`9c4196065365cfd5219caf6308e97ed21e0852eb9d99616bb511598bf5d84a10`。均未完成新封存终检，不推广为可靠新Expert。B114停止断点：`aca42857e291f023a52cdf90ce6d5a83bc4c2417365c3989c0856ad5e39d514f`。

## 4. 证据、运行路径与视频

服务器项目根记为 `P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`，SSH为`ssh pro5000`；项目环境`source "$P/env.sh"`，Python在`$P/.venv/bin/python`。不要使用共享Isaac环境。

- 旧M11 continuation固定runtime：`$P/bridge-v2-owned/trees/m11-throughput/repo`。旧`m11-privileged-expert/repo`及其结果保持不变。
- continuation：`$P/bridge-v2-owned/runs/m11-continuation-20261102/`。首先读`sequence-status.json`、`compute-ledger.json`；实际成功/运行attempt以ledger为准，不能硬编码`attempt-01`。训练`outputs/job/{launch,status,training,selection}`相应JSON/JSONL记录身份与进度。
- 初轮：`$P/bridge-v2-owned/runs/m11-privileged-expert-20261101/`；起点`train-attempt-01/outputs/job/best.pt`。完整索引见 [M11_RESULTS.md](M11_RESULTS.md)。M10见 [M10_RESULTS.md](M10_RESULTS.md)，旧M8/M9保留路径在Master/原报告。
- Mac已有M10五视频：`~/Projects/compliance-bfm-diagnostics/videos/m10-task-appropriate/INDEX.md`；M11初轮三视频：`~/Projects/compliance-bfm-diagnostics/videos/m11-privileged-expert/INDEX.md`。
- continuation未新增视频：用户提前结束，后续终检/渲染取消，有限等待脚本已退出。旧视频不变。
- [M11初轮完整结论](https://github.com/GaoE05/robot-research-bridge/issues/32#issuecomment-6567497953)；[续训协议及零均值结果](https://github.com/GaoE05/robot-research-bridge/issues/32#issuecomment-6568052677)；[续训工程失败/修复](https://github.com/GaoE05/robot-research-bridge/issues/32#issuecomment-6568170510)；[M10报告](https://github.com/GaoE05/robot-research-bridge/issues/31#issuecomment-6558767622)。

## 5. 新会话接力与边界

1. 先读本文件、纲领、Master和#33最新评论；区分M11初轮与continuation；当前两者均停止，旧运行中快照不是启动授权。
2. 只读核验服务器ledger、精确PID/启动命令、runtime SHA和工作区。进程在跑就不重启、不覆盖、不切分支；无进程也先读result/失败原因，不能默认重跑。
3. 当前授权与预算以第1节及CLEAN_EXPERT_PROTOCOL为准；旧窄域单GPU预算仅约束旧ledger，不限制本次明确授权的有界并行。真实BFM不因无GPU改CPU，共享盘总剩余不等于个人配额。
4. 共享BFM/PD/原权重、SoftSONIC和他人任务不动；只管理核验过的自有PID。代码/活动产物留个人CPFS目录，大资产`/mnt/data-oss/users/yi.gao/compliance_BFM/`；不自动下载大资产、启动Bridge/watch/真机或扩大预算。
5. 本次授权structured/direct Expert架构对照及有界训练；旧A/B不重启。新资源按独立ledger累计，普通工程修复自主处理，超出新权限或重大方法偏离才交接。

本次Mac关键视频：`~/Projects/compliance-bfm-diagnostics/videos/clean-contact-expert/INDEX.md`，三条分别是初始化、开发15/20批对比、新条件初始化/15批反例。均已同步并通过SHA、时长和完整解码验证。
