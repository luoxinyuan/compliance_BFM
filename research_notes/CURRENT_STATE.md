# CURRENT_STATE — 新会话先读

**最后核对：2026-11-02，窄域两轮及幅度诊断已结束。这是带时间戳的快照，接力前必须刷新运行状态。**

长期目标及方法见 [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)。[Project Master #18](https://github.com/GaoE05/robot-research-bridge/issues/18)保留完整历史；[窄域Expert #33](https://github.com/GaoE05/robot-research-bridge/issues/33)记录当前阶段，[M11 #32](https://github.com/GaoE05/robot-research-bridge/issues/32)保留旧训练。Library中2026-10-28的PROJECT_STATE/SESSION_HANDOFF仅是历史快照。Bridge仓库`handbook/`先前副本不是第二个实时维护源；以本目录和最新Issue为准。

## 1. 当前状态：两轮窄域训练及尺度诊断已完成，无训练运行

原独立Expert120批/92160决策未降力，完整保留。gain16修正80批/61440决策正常完成，runtime `51c653fa6b9cab0dd87a788394d842ddd9419d7a`，4452.97秒；选中80批mean12.368cm/4.703N，相对初始13.856cm/4.704N，仍未显著降力。checkpoint SHA256 `017ac2ada5677277baf4cb07a69fe201055cb969b64afa269aae579d4f68774b`。旧M11 A160/B114保持停止。

最新只做保存数据审计及冻结均值4x/16x/−4x真实物理对照，0训练更新。4x为11.296cm/4.459N/54.11%限幅；16x退至18.180cm/4.993N；−4x14.479cm/4.762N。三者完成、worker退出、GPU释放。4x尚不是可靠新Expert，存在接触前姿态变化和q_PD连续性代价，未做独立终检或Student。

首个明确问题：均值修正仍小（2.19deg latent、.330deg q_PD RMS），但未触及合法上限。XYZ显示总误差改善来自Y，保持阶段却向前超出anchor约9.5cm并受到向后近5N拉力；不是简单前进不足。主force reward在限幅段缺乏区分度，prior项并未压住收益。完整证据、推断边界及下一建议见[NARROW_RESIDUAL_FINDINGS.md](NARROW_RESIDUAL_FINDINGS.md)，阶段[Issue33](https://github.com/GaoE05/robot-research-bridge/issues/33)。

当前无训练排队。下一建议未执行：固定contact-window-only scale4对照，排除precontact重配置贡献，再决定均值/探索独立标定或有效反馈专家热启动。不得自动继续旧长训练。结果根为`bridge-v2-owned/runs/narrow-contact-expert-20261102`，训练attempt02/03，幅度诊断amplitude-attempt-02；runtime `fc650c2984c8022b4baacd1951defabdbc9c77c4`。首次幅度诊断因sparse checkout缺源文件失败，已修复并保留证据。

## 2. 可靠基线、实际结论与边界

- **M10是当前任务适配行为起点，不是万能专家。** 进入walking放手、下沉附带抬手、右转航向较M8/M9C改善，用户看视频确认；新直行改进专家5.643cm/3.296N、Student10.451cm/4.430N。但偏离状态接管、Y/Z负担、连续性仍不足。M8旧腕/躯干及M9B移动基线均保留，不以最新模型自动替换。
- M5–M8已验证有限接触技能的LPP、共享策略与访问状态聚合；M7同回合转接、M8躯干下沉有真实证据。M9有主动行走/有限转弯，不等于可靠全方向牵引。M1–M4的中间姿态泛化、恢复失败保留，不宣称general compliance已完成。
- **M11初轮**开发直行误差/力由M10的12.661cm/4.610N变为7.832cm/4.144N；新直行均值却由8.493cm/4.139N退至37.667cm/4.588N。左右跟随改善但力未降，不是全面成功。M11从M10部署Student先验学习，而非从最好的手写专家热启动。
- 同一冻结M11 checkpoint在失败条件完整采样9.329±1.167cm；仅记忆噪声未解决问题。force/GAE/raw logprob未发现对应硬错误；5N饱和压缩force惩罚，但已有误差/未限幅需求项把失败评得更差，没有force高斯sigma。
- continuation零均值＋同噪声18.119±14.218cm、4.302±.477N，对应学习＋噪声9.329±1.167cm、4.298±.269N（三执行seed，不是三次训练）。学习反馈对误差/尾部有贡献，但力几乎相同，尚无可靠均值改进证据。

**首要瓶颈：均值部署仍不可靠，实际交互力没有稳定降低。** 已停止宽任务续训；用户现已授权窄域独立Expert，按上方实时状态执行。不能把这次提前结束解释为BFM/LPP不可行或已经证明减噪无效。

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
3. 当前授权：一个GPU顺序执行、8 CPU、32GiB owned RSS、8GiB owned VRAM、8GiB本轮输出、28,800s累计监督墙钟（含失败/评估/渲染）。核对时GPU1，未来不固定GPU5，也不擅自沿用旧空闲状态；真实BFM实验不因无GPU改CPU。共享盘总剩余不等于个人配额。
4. 共享BFM/PD/原权重、SoftSONIC和他人任务不动；只管理核验过的自有PID。代码/活动产物留个人CPFS目录，大资产`/mnt/data-oss/users/yi.gao/compliance_BFM/`；不自动下载大资产、启动Bridge/watch/真机或扩大预算。
5. 本次授权仅窄域独立Expert及条件成立后的有限重复/邻域验证。旧A/B不重启；保留原结果。新资源使用按独立ledger累计，普通工程修复可自主处理，资源超限或核心架构改变须交接。
