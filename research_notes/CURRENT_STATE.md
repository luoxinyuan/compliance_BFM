# CURRENT_STATE — 新会话先读

**最后核对：2026-11-02 11:26 CST（03:26 UTC）。这是带时间戳的快照，接力前必须刷新运行状态。**

长期目标及方法见 [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)。[Project Master #18](https://github.com/GaoE05/robot-research-bridge/issues/18)保留完整历史；[M11 #32](https://github.com/GaoE05/robot-research-bridge/issues/32)记录当前训练。Library中2026-10-28的PROJECT_STATE/SESSION_HANDOFF仅是历史快照。Bridge仓库`handbook/`先前副本不是第二个实时维护源；以本目录和最新Issue为准。

## 1. 当前状态：M11初轮完成，continuation正在训练

|对象|已经核实的状态|
|---|---|
|M11初轮|122,880 RL决策、783完成训练回合、18正式终检及24冻结诊断已结束；开发改善、均值部署严重失败，未推广新Expert或训练Student|
|M11 continuation已完成|零均值residual＋同std/seed的6条真实物理对照；续训best分数/权重身份修正；初始checkpoint NumPy分数序列化修复及测试|
|正在执行|固定探索A，`fixed-attempt-02`；核对时11批、8,448新增RL决策，状态running。序列PID1422760、训练PID1422772，启动11:10:37 CST；PID仅供核验，不是清理授权|
|已授权并排队，尚未完成|渐进减噪B；选定模型的预封终检、3组真实视频与汇总。A/B各目标新增122,880决策；不等于已完成或已收敛|
|本次维护|独立文档worktree，不动上述runtime，不启动/重启训练。未授权本次维护开展新实验、Student蒸馏或扩大资源|

A/B继承相同原best、optimizer和RNG；固定物理/reward/M10 prior/网络/任务分布。A保留约.08逐维std，B前20批保持，之后100批降到1/4，末40批约.02；外部std冻结于optimizer之外，只在采样批次边界改变。每批核验实际std、raw old logprob、解析KL、clip fraction、梯度及动作效果。每40批按预定均值/失败/尾部规则选择开发候选，采样结果保留；新封存6组合不用于重选。参数见 [M11_CONTINUATION.md](M11_CONTINUATION.md) 与 [protocol](m11_continuation_protocol.json)。

`fixed-attempt-01`在24个初始评估后安全读checkpoint失败，当时0次PPO更新；修正只转换分数类型，完整失败保留。A/B复用对应的初始评估，失败耗时计入预算。不能将此解释为原从头训练失败原因。

## 2. 可靠基线、实际结论与边界

- **M10是当前任务适配行为起点，不是万能专家。** 进入walking放手、下沉附带抬手、右转航向较M8/M9C改善，用户看视频确认；新直行改进专家5.643cm/3.296N、Student10.451cm/4.430N。但偏离状态接管、Y/Z负担、连续性仍不足。M8旧腕/躯干及M9B移动基线均保留，不以最新模型自动替换。
- M5–M8已验证有限接触技能的LPP、共享策略与访问状态聚合；M7同回合转接、M8躯干下沉有真实证据。M9有主动行走/有限转弯，不等于可靠全方向牵引。M1–M4的中间姿态泛化、恢复失败保留，不宣称general compliance已完成。
- **M11初轮**开发直行误差/力由M10的12.661cm/4.610N变为7.832cm/4.144N；新直行均值却由8.493cm/4.139N退至37.667cm/4.588N。左右跟随改善但力未降，不是全面成功。M11从M10部署Student先验学习，而非从最好的手写专家热启动。
- 同一冻结M11 checkpoint在失败条件完整采样9.329±1.167cm；仅记忆噪声未解决问题。force/GAE/raw logprob未发现对应硬错误；5N饱和压缩force惩罚，但已有误差/未限幅需求项把失败评得更差，没有force高斯sigma。
- continuation零均值＋同噪声18.119±14.218cm、4.302±.477N，对应学习＋噪声9.329±1.167cm、4.298±.269N（三执行seed，不是三次训练）。学习反馈对误差/尾部有贡献，但力几乎相同，尚无可靠均值改进证据。

**首要瓶颈：随机训练取得的反馈，能否在均值执行中可靠发挥作用，并真实降低交互负担。** 先完成既定A/B，分开判断Expert质量、合法BFM表达与Student差距。若均值差距改善但力不降，再依据分项物理证据选择下一目标；本轮不同时调reward/cap，也不默认更多数据或BFM无能力是答案。

## 3. 源码与模型身份（不要把文档HEAD当runtime）

长期集成/审查分支：[gaoe05/research](https://github.com/luoxinyuan/compliance_BFM/tree/gaoe05/research)。本次在严格快进后增加文档提交；文档提交身份以GitHub该文件的commit记录为准，避免自引用SHA。实验分支与运行源码保持不变：

|用途|精确commit|
|---|---|
|当前运行及本次整合的实验源码|`38c7b25dcb6a3a1124acd9cf44c4eb116364ac60`，`gaoe05/m11-privileged-expert`|
|M11初轮完整报告/源码|`466266385fab2fec9852253d661ecd24cd675c64`|
|M10完整报告/源码|`49639ed29d4cbf513c285633841426579d80d5bd`|
|长期分支整合前的旧HEAD / 共享checkout基线|`6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`|

关键权重SHA256（不进Git）：
- 冻结BFM：`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- M10部署Student / M11 prior：`1742ba74e9ec871abf8577eb9a276d7a1e614cd3184330a008dd9c4a7031bcd8`。
- M11初轮best / A、B共同起点：`bc305de6917103c5fbc64a6c6639f91bb517d3c7a48d689e7158739a6168cc8f`，原iteration60/46,080决策。
- continuation最终模型**尚未选定完成**；不要把滚动`best.pt`当正式交付。核对对应`selection.json`、evaluation SHA与训练量。

## 4. 证据、运行路径与视频

服务器项目根记为 `P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`，SSH为`ssh pro5000`；项目环境`source "$P/env.sh"`，Python在`$P/.venv/bin/python`。不要使用共享Isaac环境。

- 当前固定runtime：`$P/bridge-v2-owned/trees/m11-privileged-expert/repo`。
- continuation：`$P/bridge-v2-owned/runs/m11-continuation-20261102/`。首先读`sequence-status.json`、`compute-ledger.json`；实际成功/运行attempt以ledger为准，不能硬编码`attempt-01`。训练`outputs/job/{launch,status,training,selection}`相应JSON/JSONL记录身份与进度。
- 初轮：`$P/bridge-v2-owned/runs/m11-privileged-expert-20261101/`；起点`train-attempt-01/outputs/job/best.pt`。完整索引见 [M11_RESULTS.md](M11_RESULTS.md)。M10见 [M10_RESULTS.md](M10_RESULTS.md)，旧M8/M9保留路径在Master/原报告。
- Mac已有M10五视频：`~/Projects/compliance-bfm-diagnostics/videos/m10-task-appropriate/INDEX.md`；M11初轮三视频：`~/Projects/compliance-bfm-diagnostics/videos/m11-privileged-expert/INDEX.md`。
- continuation三视频**尚未交付**；目标为后者的`continuation/`子目录，预设有限传输脚本等待序列完成。最终需核非空、SHA、时长、完整解码和视觉内容，不能把服务器文件称已到Mac。
- [M11初轮完整结论](https://github.com/GaoE05/robot-research-bridge/issues/32#issuecomment-6567497953)；[续训协议及零均值结果](https://github.com/GaoE05/robot-research-bridge/issues/32#issuecomment-6568052677)；[续训工程失败/修复](https://github.com/GaoE05/robot-research-bridge/issues/32#issuecomment-6568170510)；[M10报告](https://github.com/GaoE05/robot-research-bridge/issues/31#issuecomment-6558767622)。

## 5. 新会话接力与边界

1. 先读本文件、纲领、Master和#32最新评论；区分旧结项与当前continuation，旧Library/Issue历史中的“停止”不是当前事实。
2. 只读核验服务器ledger、精确PID/启动命令、runtime SHA和工作区。进程在跑就不重启、不覆盖、不切分支；无进程也先读result/失败原因，不能默认重跑。
3. 当前授权：一个GPU顺序执行、8 CPU、32GiB owned RSS、8GiB owned VRAM、8GiB本轮输出、28,800s累计监督墙钟（含失败/评估/渲染）。核对时GPU1，未来不固定GPU5，也不擅自沿用旧空闲状态；真实BFM实验不因无GPU改CPU。共享盘总剩余不等于个人配额。
4. 共享BFM/PD/原权重、SoftSONIC和他人任务不动；只管理核验过的自有PID。代码/活动产物留个人CPFS目录，大资产`/mnt/data-oss/users/yi.gao/compliance_BFM/`；不自动下载大资产、启动Bridge/watch/真机或扩大预算。
5. 当前维护完成即结束，不触发新实验。训练阶段最终核验后更新本摘要、#32和Master，再推进获准源码到`gaoe05/research`。保留runtime commit、checkpoint SHA、完整失败和关键1×视频；不要以训练步数/高EV/代码已推送宣称科学成功。
