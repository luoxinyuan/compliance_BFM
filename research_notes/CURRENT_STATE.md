> 2026-11-04（北京时间）M38-C执行中：15因果对照＋9接管/回放，D0/D1/D2各3000更新及开发评估已完成（runtime e88f259）；已用满9000监督预算，无PPO/Student。接下来仅一次预封存评估与诊断/视频。旧模型不替换；下文保留M38-B历史。

# CURRENT_STATE — 新会话先读

最后核对2026-11-03。**M38-B授权对照已完成并停止；C2部分保旧改善，但未得到可靠双向Expert。** 所有owned GPU任务退出，未启动PPO、Student蒸馏或M39训练。不要重放旧任务。

入口：[研究纲领](RESEARCH_OVERVIEW.md) · [M38-B完整报告](M38B_RESULTS.md) · [Issue38](https://github.com/GaoE05/robot-research-bridge/issues/38) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。[M38原报告](M38_ROTATION_RESULTS.md)及所有M0–M37结果保留。

## 已有成果与本轮真实边界

- 冻结T30/S0仍是有限域直行/停止Teacher→ideal-force Student基线。用户已认可可执行行走及小幅上半身转动，不要求本轮迈步转身，不用毫米误差否定有效行为。ideal-force不是已确认G1真机传感能力。
- 旧torso双点平移共享Student成果保留，但旧腕退化；不直接替换S0。torso_link两点[0,±.13,.15]m，各2.5N/总5N，双向保持接触Tether，非单向按压。
- M38原生因果旋转源左右±15°有效，r1/r2有不同方向的正例；r2新左+13°却反向。历史保留，不按方向拼模型。
- **M38-B三组匹配各3000更新：**C0原终点q_PD+.1z；C1只改为真实上一z出发的5子步动作保持；C2同C1加固定20%有效恢复轨迹。原10回合1554标签一致；新增383标签来自3条精确接管后实际成功轨迹，不是全部r2离线查询。
- C2直行6.327cm/3.806N（T30 5.826/3.599），局部3.118/2.215（T30 4.121/3.107），晚停6.559/3.827（T30 6.083/3.667）；比C1在这些条件改善，但早停退化。双点平移仍8.171/4.504（T30 4.998/3.997）。行走仍可用，不是“什么都没学会”。
- **同一个C2：**开发左/右torso−8.22/−11.54°；新封存±11°左反向−8.68°、3.481cm/4.242N，右−11.12°、0.910/1.460。新左原生源+6.53°有效。C0/C1也没有修好左向；没有终检重选/补训。
- 源状态C2左右5子步动作RMS仅.039/.055°，自身左向访问状态对同状态专家误差却约1.239°，z角40.78°。单独替换上一z输入的离线干预仅把误差降到1.115°；自身记忆有影响，但不是已证实唯一原因，也不是物理恢复实验。
- 原数据117次完整输入逐值重复，没有不同目标；左右接触后可区分。未复发旧标签冲突，但不证明全域可辨识。真实访问域覆盖、记忆反馈与物理状态偏离仍需区分。
- 50完整回合未跌倒/记录到双足同时失去支撑。C2纯旋转q_PD最大约.4°/20ms；直行43.52°/20ms，但真实q为2.42°/5ms，不能混称。双点无力torso仍漂移约−10.76°，root仅约.56cm；空接触error=0不是无动作证明。释放后旧anchor不计任务失败。
- M39前置原生源探针：移动中±10°确有不同腰响应，但goal参与时root停在约30cm、伙伴继续前移；全接触误差14.02/16.42cm、力4.65/4.67N。能进出转腰不等于移动＋旋转协调合格；未继续调混合或训练M39。

## 首要瓶颈与唯一下一决策

当前不是“BFM完全不能转”，而是**静态监督的小动作误差不能稳定保持模型自身闭环历史与正确旋转方向**。5子步对齐及旧r2三条恢复轨迹已经实测，不能原样反复重训。

建议下一次有限实验在C2首次左向分叉前后获取实际接管证据，采用模型自身生成z历史的短序列监督对照，检验闭环历史一致性；与当前独立帧teacher-forcing比较。尚未授权/启动新更新，本轮9000已用满。保留T30/S0及C2有限收益，未把C2推广为统一Teacher，也没有蒸馏反向标签。M39原生组合功能冲突是另一个已知边界，不新增手工force-to-yaw控制器。

## 精确身份与复现入口

长期个人分支`gaoe05/research`，最终文档commit见Issue38 M38-B结项链接。实际训练/开发/终检/原生探针runtime **aa69aa7cbe344319b08cde65a326575f2b96b4cb**；诊断/渲染 **8768a0b3ee8ec8136afc01b1d6d13ffd5e70fe18**。

服务器根`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m38b-rotation-20261103/`。

|模型|相对本轮根路径|SHA256|
|---|---|---|
|C0|A-train-03/job/C0.pt|f8676d14df554dd8fccfbe903a2cacfdf643015782c4e60eb9cdbe1f55b1d45a|
|C1|B-train-05/job/C1.pt|5cdf95c4ac6c5aae2ba705efacc7a45cdfe969538c3ad56fbb47c17743aaaab5|
|C2|C-train-04/job/C2.pt|ff471b60c0bfc53c1d582a82fd3562416216c67c480e0e8039dee3d09277782c|

T30 SHA `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`，位于`bridge-v2-owned/runs/stop-sensing-20261103/B1-train-02/job/best.pt`；S0 SHA `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`，位于`direct-transfer-20261103/A-prepare-01/job/ppo15-student-r0.pt`；BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`，均未修改。

[结果](m38b_artifacts/results.json) · [分阶段物理](m38b_artifacts/physics.json) · [账本](m38b_artifacts/resources.json) · [恢复](m38b_artifacts/recovery.json) · [同状态诊断](m38b_artifacts/same-state.json) · [曲线](m38b_artifacts/learning-curves.png)。原始轨迹/权重仅服务器，Git不上传。

Mac：`~/Projects/compliance-bfm-diagnostics/videos/m38b-rotation-retention/`，仅3条MP4＋INDEX：左反向失败、右向保留、T30/C2直行。真实qpos，1×，16.08s，SHA/全解码/画面已核验。视频里的C2是特权Expert，不是新Student。

## 成本、安全与接力

累计2000.803受监督job秒，约1.31GiB，Expert9000更新、PPO0、Student0。首次尾部不足5步、第二次numpy.bool元数据序列化失败均保留；后者从已成功保存的3条物理轨迹恢复索引，没有重跑。模型/物理成功以退出码和证据判定，失败wrapper残留running字段不是正在训练。

GPU1/2/3已释放；GPU0/5/6他人任务未操作。共享checkout clean、main/共享BFM/PD/权重/环境不改，无新资产下载、无Bridge/watch、无真机。新会话先读本文件和报告，再核对Git/模型SHA/ledger/资源，不沿用空闲快照发车。Library 2026-10-28快照是历史；实时状态以这里、Issue38和Master为准。
