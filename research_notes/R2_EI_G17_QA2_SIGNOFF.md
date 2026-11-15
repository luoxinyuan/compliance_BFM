# Issue65 QA2 v1.0 — Yi签署记录

批准日期：2026-11-15（Asia/Shanghai）。批准人：Yi。状态：**APPROVED**。

用户原话：

> 批准issue #65 那版 QA2合同为正式文本；#64 是历史稿、作废，别混用。整份合同一次批完——P0→P1→V1→V2 连续执行，P2 和 A0 按合同内各自的资格门条件触发，不用再逐段回来问我；先把已签合同逐字同步归档到执行分支并留下签署记录，再更新 Issue #65 和 Master #18 的状态，然后从 P0 开跑。

唯一生效正文：`R2_EI_G17_QA2_VALUE_TARGET_CONTRACT_V1.md`，源为 GaoE05/robot-research-bridge `734a669a:contracts/R2_EI_G17_QA2_VALUE_TARGET_CONTRACT_V1.md`；Git blob `7e4c5e14c4dfe08eba3731c759caad67a746f283`。包含Issue65追加的§9。正文逐字保留，包括签署之前的DRAFT措辞；本签署记录使其生效，不改写原始版本。

Issue64及其旧预算/旧合同为作废历史，不是本轮执行依据。QA1与原G17 FAIL、旧erratum、BFM/PD与既有资产全部保留，不重判。

执行顺序：P0→P1→V1→V2；P2仅按标签、模型与资源入口触发，不要求先取得只有P2才能产生的G1排序标签；A0须额外满足同策略价值资格，μ价值不能自动指导确定性mean Actor。Actor/U/I/Dev/Sealed默认不启动。

本轮硬帽：9,000 job秒、3GiB、38,000全流程5ms ticks；其中准备/失败/报告reserve1,200秒内工程修复总池≤800秒，每个稳定failure family最多修一次，须完整等价/恢复回归。各阶段子额沿正文，不挪用未触发阶段预算。继承R2累计17,244.117190454097/57,600秒及12GiB上限和项目240小时总账，单GPU0、共享free≥100GiB。

归档不等于实验完成。正式运行模型、数据、采样序列、来源、父组与成本分别记录在新隔离stage，运行前后均保留精确源码SHA。
