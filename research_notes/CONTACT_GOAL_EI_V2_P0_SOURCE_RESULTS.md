> 后续状态：Yi已追加授权自行修复源实现后继续；本文保留修复前失败事实，非最终停止。见[SOURCE_FIX_AUTHORIZATION_01](CONTACT_GOAL_EI_V2_SOURCE_FIX_AUTHORIZATION_01.md)。

# Contact-Goal E/I v2 — P0源祖先资格结果

2026-11-15。**BLOCKED_G01_DATA_INSUFFICIENT；没有开始正式训练。** Yi的整体批准与独立新增6,000 job秒已归档；[签署记录](CONTACT_GOAL_EI_V2_SIGNOFF_20261115.md)保持。此次按[获批合同§5.1/G01](CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_APPROVED.md)执行数据资格，发现新增arm-ready源衍生破坏了事前要求的train/dev源祖先隔离，故停止P0→P1。不是PPO学习失败，也不是BFM能力失败。

## 1. 首个阻断及责任边界

v2修订稿把train与dev各6个父组切成`goal_arm5`，但继承原standing源的`source_parent_id`及profile。实际`goal_arm5`生成器只读取`arm_fraction`；这12个源该字段全为1。标识中不同的pulse/phase字段没有进入这条源的实际控制。结果是**一条共同native轨迹在不同时刻截取，却仍带不同源父组ID**。这是本次修订稿源分组设计遗漏，不能把旧字符串去重当成新的物理资格。

证据代码：运行源码[dual_track_task.py L240–251](https://github.com/luoxinyuan/compliance_BFM/blob/6f1da47268a69d1bbdac0fc2f85f05eed0a8f9ef/research_notes/dual_track_task.py#L240)、[m8_stage.py L14–26](https://github.com/luoxinyuan/compliance_BFM/blob/6f1da47268a69d1bbdac0fc2f85f05eed0a8f9ef/research_notes/m8_stage.py#L14)、[native_env.py reset](https://github.com/luoxinyuan/compliance_BFM/blob/6f1da47268a69d1bbdac0fc2f85f05eed0a8f9ef/research_notes/native_env.py#L40)。reset实际seed0并校验原3秒standing；arm5使用固定smoothstep。**standing分支确实读取不同pulse，不能把此发现扩写成全部历史源都相同。**

## 2. 决定性真实物理验证（仅两个train前缀）

|项目|实际结果|
|---|---|
|短前缀train父组|`78a7f21fe0c2c1f5980f0d70a37f25e949024dd012198ebd75215c169b9600c1`，handoff4.1s|
|长前缀train父组|`fdde460420a47e3cb30e3a7b298277a9de8b2ed3de08ebd5547391339bbb24aa`，handoff4.6s，在真实4.1s截取|
|共同物理窗口|3.0–4.1s，双方各220条5ms记录|
|底层实际指令|双方各55条20ms action/z/q_PD/BFM观测，逐值相同|
|身体前缀|time/qpos/qvel/wrist/root/force/support，全部逐值相同|
|完整边界状态|11/11字段逐值相同：time、integration、有效MjData数组/有效域元数据、BFM观测、previous action/z/q_PD、Student许可历史、真实force、physics RNG|
|真实伙伴力|整个被检查源前缀为0，无active接触|
|模型修改|原BFM Actor哈希前后相同；优化器/训练更新0|

“11/11”是状态字段类别数，非11个独立样本。完整归档包含原始portable物理状态及控制/身体历史，不只qpos。有效域规则原样复用F1的既有no-island/Newton/dense证明，没有新增字段排除或放宽容差。两归档SHA/体积见[机器摘要](contact_goal_ei_v2_p0_results.json)。前缀比较源码[ei_v2_source_audit.py](../research/interaction_fb/ei_v2_source_audit.py)，recipe展开[ei_v2_manifest.py](../research/interaction_fb/ei_v2_manifest.py)。

## 3. 跨split证据及独立性分母

结构上仍是train34父组/44回合、dev16父组/20回合；source标签各6arm，其余standing；K80/K40的dev各8。完整44训练回合下界21,870条50Hz，未超过32k。**这些数量通过不代表实际独立性通过。**

以下5对train/dev源的实际生成配方及截取时间完全相同（K/请求不同不会在未接触native前缀中产生不同物理）：

|train父组前缀|dev父组前缀|共同source_until|
|---|---|---:|
| `cb29bf09014a` | `a3c5b59ce707` | 4.5s |
| `cb29bf09014a` | `a7540e92d5c9` | 4.5s |
| `4cfa2904af7c` | `5a13f004573c` | 4.4s |
| `fdde460420a4` | `5f8aa84d8a4f` | 4.6s |
| `fdde460420a4` | `80c1b0855c6e` | 4.6s |

证据等级区分：两train实际完整前缀相同是**实测已证实**；上述5对是**源码/输入配方与已证实确定性重建支持的跨split推导**，本轮未执行dev物理，不能写成5次dev重放已通过。12个arm记录同属一个native曲线祖先的原因由生成公式直接核实，不由哈希猜测。

因此dev的6个arm父组不能作为6个未见源。若10个standing源后续各自资格成立，dev也至多11个不同源族，其中arm1族已跨split共享；排除共享arm后，未见源上界至多10。**这只是上界，不是已完成10个源的物理资格。** partner任务仍不同，不等于所有联合任务重复；失败的是本合同明确要求的源祖先隔离/16独立dev父组口径。

Sealed案例文件未读、未执行；只用了原audit.json中已公布数量/hash摘要。不得从Sealed补源或把相邻相位换ID凑数。

## 4. 工程失败和一次等价修复

`P0-source-identity-01`因新stage缺失`prep/pointer_layout.so`在导入时失败：exit1、107.662970秒、1个初始化tick。原始log/sidecar完整保留。

按§9只补齐H0已验证的同一helper与json，不改任何源、状态比较或科学逻辑：helper SHA `8e7d74e1ffb947f1ccfc82b083dbf91b6433e2da847c29dc8e96ef1709dfb439`；json SHA `fa3c671ca8162ed1917bf0f7fbdcf2b701aa18f41342b202dbe5a554708df5ff`。MuJoCo3.3.3、安装头文件SHA、qpos/qvel ctypes指针ABI校验均通过，修复池记10秒。独立输出`P0-source-identity-02`退出0；实际134.017792秒/2341ticks。**程序运行成功与G01科学失败分别记录。**

## 5. 成本、身份与各门状态

- 批准归档`15fc96c573deba4f01972ccdeb85e8655428622e`；实际审计源码`6f1da47268a69d1bbdac0fc2f85f05eed0a8f9ef`。
- 原BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；未生成E/I训练checkpoint。
- P0准备保守450秒＋失败107.662970＋有效审计134.017792＝691.680762秒；reserve工程10秒；证据整理/归档/推送保守600秒，总新增**1301.680762 job秒**。两物理job嵌套子进程不重复计费；保守准备/报告上界与实测作业分别列。
- R2在此前21,537.928257基础上增至**22839.609019/57,600秒**，余额**34760.390981秒**；独立新批准6,000秒**尚未使用**。项目240h仍累计，这轮未建立跨所有旧ledger的非重叠精确总数，不能虚构总账余额已核清。
- 全流程真实/预留5ms ticks各2342；正式E/I50Hz=0/0，scratch训练=0，BC/Actor/U/FB更新=0，dev/sealed执行=0。
- 新接口/联合概率/奖励/新策略恢复/端到端吞吐及后续任务门全部UNMEASURED/NOT_REACHED；本轮只完成源身份检查与helper ABI修复，不借旧PASS补过。
- GPU0作业已全部结束；现场复核无显存占用。共享free末扫及输出完整体积见同期机器摘要，≥100GiB护栏未降低。

## 6. 唯一下一建议（未执行）

建议仅修订**arm-ready源生成与祖先manifest**：为train/dev各自预定真正不同的native准备前缀，并连续进入同一合法arm5，完整保留历史；先锁定实际使用字段与split祖先图，再有限资格，不按物理表现挑源。保持E/I网络、reward、采样、预算和原评价口径不动；无需再开标量Q考试或H0.1。

这涉及已签source分布/manifest，超出§9纯路径修复权限，须明确审阅相应v2.1补丁后才可执行。不能静默加入pulse、删除arm、把16改10、放宽祖先隔离或从Sealed取样。当前已停止，无训练队列；保留获批v2正文、旧H0正结果及全部历史负例不重判。
