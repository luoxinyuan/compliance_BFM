本轮资产资格核查有实质进展：**官方确实发布了当前权重对应目录下的原始 replay buffer，之前只是本地没有下载；但没有找到当前权重对应的预导出 reward latent。现成的 `reward_locomotion.pkl` 位于另一套权重的发布目录，不能混用。** 本轮尚未获得通过来源资格的两个 reward 命令，因此没有进行 MuJoCo 进入—执行—返回验证，也不能宣布已获得局部/全身技能对。

**资产与来源证据**

官方 Hugging Face 固定快照：`LeCAR-Lab/BFM-Zero@62b4206d68e026de5e5dc7efb1529bccfb95164c`。本轮成功取得递归文件清单，共47项（含目录），解决了上轮发布端访问失败的问题。以下 SHA 均来自该固定快照的 LFS 元数据；当前本地权重另作完整重算，结果一致。

| 资产 | 实际发现 | 资格判断 |
|---|---|---|
| `new_model_for_training_code_inference/checkpoint/model/model.safetensors` | 3,384,968,272 bytes；SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361` | 与当前冻结BFM完整文件重算一致；根config、model config、init kwargs也与官方文件逐字节一致 |
| 同目录 `checkpoint/buffers/train/buffer.hdf5` | **26,675,212,288 bytes（24.84 GiB）**；发布SHA256 `46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b` | **真实发布且可访问**；HTTP 206读取前8192 bytes，Content-Range总长一致、HDF5 magic正确。未下载完整文件，未本地核验全文件SHA或逐条transition内容 |
| 同目录 `checkpoint/buffers/train/config.json` | 548 bytes；SHA256 `5f6ba56ad7c4ec281c75f8bc488f02b7d702fdfa5cf2224aee5f6c94c4d38c20` | 已读取/核对Git blob；原始trajectory buffer元信息，非LaFAN reference |
| 同目录 `reward_inference/reward_locomotion.pkl` / `checkpoint/buffers/train_reduced` | 当前发布清单未包含，本地也未找到 | 尚无可直接执行的这两个reward命令；没有把清单缺失扩大为“作者/团队任何地方都没有” |
| `model/reward_inference/reward_locomotion.pkl` | 541,950 bytes；发布SHA256 `1b974e7278ab6183d809d11fbd77cca817f3714a4808737cd69f96606ab77718` | 所在`model/`权重SHA为`d8d07d7b83ad030286f44d55a48899dc42db4b48000132fffad063653a6eb49e`，与当前不同；不下载/加载/混用，也不以相同256D或norm16推断兼容 |
| 当前模型目录 `exported/zs_25.pkl`、`zs_7.pkl` | 分别13,377,777 / 6,730,993 bytes；发布SHA见`qualification.json` | **不能再说官方没有任何tracking导出。** 但它们是按motion相位播放的序列，不是这两个固定reward命令；未核验内部内容或从共同standing进入/退出，本轮没有改用整段reference-reset来替代所需资格验证 |

来源：[固定快照目录](https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/62b4206d68e026de5e5dc7efb1529bccfb95164c/new_model_for_training_code_inference)、[原始buffer目录](https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/62b4206d68e026de5e5dc7efb1529bccfb95164c/new_model_for_training_code_inference/checkpoint/buffers/train)、[另一套模型目录](https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/62b4206d68e026de5e5dc7efb1529bccfb95164c/model)。同发布快照/模型子目录是buffer与checkpoint的来源关联证据；不是已经完成buffer内容语义验证。HTTP ETag是Xet对象标识，未把它当作文件SHA256。

**预处理、生成与执行合同**

- 原buffer配置：`TrajectoryDictBufferMultiDim`，capacity=5000、n_dim=2、seq_length=1、cursor=1250、full=true，嵌套键分隔符`-`。原loader会读取全部HDF5 arrays；不是只把26.7GB文件放到路径下就成为轻量检查。配置里的旧Python模块路径被当前显式class loader剔除，不应据此先判不兼容。
- 当前reward脚本从有效trajectory边界采样，使用 **action(t) + next qpos/qvel + next observation**，原reward重标注，再通过当前模型的backward/`reward_wr_inference`得到z。必须保留环形buffer游标、终止边界、quaternion/关节顺序和模型自己的观测预处理；不能拿LaFAN姿态替代、错配t/t+1，或复用另一模型缓存的B。当前默认150,000样本；本轮没有执行抽样、生成新buffer或推断z。
- 预期命令文件格式由当前源码确认：`{task_name: [z, ...]}`，每个z为`[1,256]`，非零时norm16。目标两个key为`raisearms-l-m`、`move-ego-0-0.3`。这是**所需文件合同**，不是声称已读取旧pkl验证其内容。执行仍用当前actor均值、原50Hz/200Hz actor-PD/物理节拍；固定reward z本身不依赖逐帧未来motion。
- `raisearms-l-m`目标是左腕60–80cm、右腕至少100cm及站立/低速度，**并不等于右手局部前伸**；`move-ego-0-0.3`是身体前向COM速度目标0.3m/s，**并不等于已经安全行走/停止**。两者语义和真实动作范围仍须物理验证，当前全部未通过资格。

**作者答复解决了两个复现疑点**

作者说明新checkpoint使用更硬的腰部，需要匹配PD；reward z会多次subsampling择优，部分reward推断存在塌缩限制。因此，不应期待任意一个样本批次精确复现已发布旧z，也不能把其择优后的演示当作本模型单次推断保证。[作者说明](https://github.com/LeCAR-Lab/BFM-Zero/issues/26#issuecomment-4966553366)

已用保存的真实运行`action-contract.json`核对：当前硬腰Kp=300、Kd=5、normalize/action rescale=5，腰yaw有效q_PD scale约0.366667rad、roll/pitch约0.208333rad，与作者针对新模型给出的关键修正相符。原checkpoint config也指定`g1_29dof_hard_waist`。本轮没改PD；这不代替完整部署接口的逐项数值等价验证。当前deploy分支默认腰部参数仍不同，不能直接照抄覆盖已验证运行链。[作者配置](https://github.com/LeCAR-Lab/BFM-Zero/issues/17#issuecomment-4035650943)

**有界核查范围与实际验证量**

- 官方`main@05ad6119a25b06c341c191a72ccf936e987f59cf`、`deploy`、`minimal_inference`递归树分别324/122/94项，均未截断；main的model目录是示例媒体，deploy/minimal下载脚本指向HF旧`model/`，不是另一份当前权重的reward导出。没有下载整个模型库或任意爬取其它项目。
- 团队`luoxinyuan/compliance_BFM`的main递归树339项、无截断，README与资产下载器仍指向上述官方新模型；远端分支列表为main和gaoe05/research。本机共享checkout仍为`6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`且干净，没有合并远端变化。
- 本轮重新检查个人项目OSS资产目录及项目HF cache，深度上限7，无扫描错误/截断，未找到reward pkl或buffer；上轮已经检查的个人runs/outputs缺失结果直接引用，没有扫描其他用户目录。团队其它机器或未发布私人资产不在此证据范围。
- 当前模型全SHA重新验证；5份下载配置的Git blob与官方清单一致，其中3份当前模型本地配置与官方内容一致。HDF5只做8KiB头部检查。模型前向 **0**、MuJoCo物理步 **0**、策略更新 **0**。因此腕/肩/腰/root/足运动、稳定性和退出连续性均为**未测**，没有虚构进入—执行—返回结果。查看了全部7张GPU，未分配GPU、未停止其他任务；停止原因是候选资产资格，不是只看GPU5。

**首个边界与唯一下一步**

首个边界已经从“官方资源未知”推进到“**原始buffer已定位，但当前checkpoint的两个reward z尚未取得或重算、未完成物理资格**”。现有证据既不支持BFM没有这些能力，也不支持它们已经可用。依据本轮有界资产检查范围，未擅自整包下载24.84GiB或启动全默认24-worker推断；也未从旧pkl偷换命令。

**唯一推荐下一步：用已定位的同发布buffer补齐这两个固定reward z，再作共同standing的进入—执行—返回资格验证，仍不训练Student。** 预先固定抽样seed/次数和资源预算，记录每次推断与所有物理失败，不根据末次有利rollout隐去其它结果；先明确原生行为是否满足局部/身体参与语义，再谈接触触发。当前不提出已可执行的接触切换实验。

向Xinyuan/作者可直接索取的最小交接内容（本轮未对外发信）：针对权重SHA `33f410…69e361` 的上述两个z（最好只导出这两个小张量）、文件checksum与checkpoint绑定说明；生成代码commit、reward版本、原buffer SHA、抽样seed/样本数/是否best-of-N及选择记录；如没有预导出z，则确认已发布`46eac5…93b546b` buffer的生成预处理和建议推断设置，并提供任何已有standing进入/返回记录。**无需再泛问“有没有原buffer”——它的位置现在已确认。** 这些资料能避免混权重并减少不必要的大文件传输；并不声称必须等作者答复才能从官方原buffer重算。

归档：`P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM`；证据目录`P/bridge-v2-owned/runs/issue21-native-assets-20261030A/`，含`qualification.json`、`local-assets.json`、官方/团队完整目录元数据、配置校验及8KiB header记录。个人分支`gaoe05/bfm-native-asset-qualification`，仅新增只读审计脚本/小型manifest/本报告；无BFM、教师、Student或共享源码改动。原S0/S1、C/D及历史结果保留，不push、不启动Bridge。到此停止。
