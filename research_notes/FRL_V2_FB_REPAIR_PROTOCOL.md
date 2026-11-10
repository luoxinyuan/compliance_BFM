# Issue58 — F-RL-v2 / FB修复：新阶段预登记

状态：新授权，准备与工程验证中。事实父提交8dd48a0；旧55cfe85与8716f28、全部结项和erratum不改。物理搜索退出主训练数据生产，仅可独立测量/抽验。本阶段不含Student、正式FB1M或5M。无结果视为未知，不把smoke当科学模型。

## 资源、成本与停机

仅GPU0 RTX PRO5000，重任务串行，原项目进程锁不变。P0-A≤14400s、P0-B≤43200s、F-RL-v2两臂合计≤172800s；首先达到各自额度即保存停止。总240job小时继续分账，旧双轨16.056h、后续D1/D2保守3226.80s、三包2271.82s等继续计费；精确汇总以原账本去重为准，未完成汇总不得宣称新额度来自旧未用样本。新数据/日志/失败/独立评价/准备/IO/恢复全计费，不把子进程评价重复计算或漏计。每包独立job sidecar、runtime和计数，所有旧资产不删不覆盖。

共享盘自由空间保护100GiB，不降低；启动/采样块/存盘前检查。初次仅43.6GiB被阻塞，用户清理后实测约499GiB；OSS容量占位不能当配额。不修改main、共享BFM/PD/权重、SoftSONIC或他人任务。

## P0-A：恢复更新，固定数学分量累加

新runtime中仅重绑discriminator方法：expert BCE、unlabeled BCE、原权重WGAN-GP各自autograd.grad，按expert→unlabeled→GP顺序逐参数FP32累加。全部有效梯度保留，原loss/GP系数/六Adam/target更新语义不变；旧runtime不改。先完整实际采样→更新→采集中停止→不可覆盖原子包→新进程恢复与继续采样→完整更新；真实mini-batch、RNG、模型/六Adam、replay/历史/solver/实际last-z仍按原逐值断言。记录每个optimizer的每个有效梯度SHA，不用全模型均值掩盖差异。

优先eager逐值一致；compiled独立验证，不要求二模式相同。若eager精确失败，先做不中断重复数值基线再另锁浮点更新的逐张量误差界，不预先采用容差，也不放宽状态/数据/RNG。新修复通过不改判旧D2门。后续还需固定输入16更新及同状态action/真实物理回合核查，未完成不发FB长跑资格。

## P0-B：独立安全执行合同（物理运行前须补齐量测依据）

保持原真实q越界.02rad、10N/K80D8、控制频率、reward、97D和robot-only discriminator。拟以原模型关节范围推导左右ankle roll q_PD安全区，经独立动力学检查确定有界映射；原始提议、变换后实际action/q_PD、触发与行为来源逐步记录，replay/F/critic与执行一致。固定交接.35s（不按测试调）；只在真实原生无接触域保持新旧实际动作，不约束接触域。三源各10独立真实前缀，旧step0随机、原生均值、新安全actor配对。工程≤8192真实50Hz/16updates每1024，保存0/早期/用户停止包。

P0-B尚未发车。变换/无接触保持权重和退化界必须在该包运行前进一步锁定，不能用本段计划冒充已实现合同。若破坏原生运动则停、不放宽边界。该包不阻塞F-RL-v2。

## F-RL-v2：共同合同与唯一机制差异

standing原生真实前缀，右腕前向；near .04–.12m /平均.03–.09m/s，mid .16–.45m /平均.06–.15m/s；stop也从mid生成。onset为handoff+.5–1.2s；减速.5–1s，hold1.5–3s，release后4s。BC500+零初始化特权支路、原normalizer，原BFM/PD冻结，fresh critic/Adam，std .008只采样一次，jointKL .02/原logprob3e-4/事务Adam回滚。gamma .99/100ms（native=.99**.2），原积分reward及GAE .95。actor3e-5、critic3e-4，6epochs/256minibatch，首批100value-only，768真实on-policy决策/批，128批/臂，总196608。无搜索/Teacher覆盖数据。

### 课程与独立组

episode边际near/mid/stop/zero/none=.30/.30/.20/.10/.10，stop父组包含停止与继续子回合；continue计mid。宏组选择near .375、独立mid .125、stop-pair .25、zero .125、none .125，故完整宏组的episode边际符合上述比例；有限预算末尾残余及真实完整回合/决策/阶段另报，不拿理论概率当实际经历。stop与continue共享源/请求/初态和分叉前随机流，actor不得读取未来距离或时钟。

训练seed5801；dev58031，每near/mid/stop至少8父组，zero/none各4；封存58061，每near/mid/stop至少16父组，zero/none各4。实际源前缀/父组hash独立性另验，不把新seed叫新物理组。旧#56全部条件已暴露，仅回归；62.07cm/.22m/s为独立暴露退化哨兵，不进主训练/封存成功统计。

### B唯一增加：接触前函数保持

A系数0；B初始lambda=.1，目标均值D=.01。D=sum256(mu-mu_init)^2/(2*.008^2)，不除latent维数。每个PPO minibatch同时在固定参考小批量与本批合法precontact输入上计算等权保持，lambda在完整PPO batch结束按lambda=clip(lambda+.1*(meanD/.01-1),0,100)更新；频率一次/批，完全固定，不据开发/封存调参。lambda及参考池索引/RNG随checkpoint保存，拒绝的Adam候选不能改变它们。

参考分布：真实原生源交接后contact onset前状态、当前策略实际访问的合法无实际外力状态。释放后的迟到追赶不进入保持池；连接零需求若已经active也不自动当precontact。训练mask使用真实当前/既往接触记录，仅loss可见；未来onset、family/stage标签不进入actor。原始输入x相同；init网络eval冻结、不维护虚构独立history。报告均值/p95 D、z方向、五子步实际action/q_PD及真实前缀分叉。接触后按原任务reward自由学。

### Reward前置及选模

先对已有真实匹配源/需求/时序轨迹复算实际优化reward及物理时间折扣return：force/error/饱和、滑移、连续性、终止尾部、启动/推进/保持/停止分别报告。明确更好且无新质量退化的反馈必须被正确排序；证据不足不宣称正确，明确结构性错误则停正式RL、不静默改reward。必要新配对评价属于48h且不产搜索训练标签。

dev节点0/16/32/64/96/128，独立进程与独立sim/RNG，父训练状态逐值不变。原物理保护与P5容差：原终止排除该候选；族中位误差不可恶化>max(2cm,20%初始)，力不可增>.75N，loaded slip峰不可增>max(.20m/s,50%初始)，release yaw不可恶化>10度。接触成本F_RMS+.03*error_cm+cap_fraction；near/mid/stop等权（zero/none另列控制），不按任务拼模型。

在保护内选等权成本最低的固定checkpoint；封存一次。主要行为共同收益：至少两族成本相对初始化下降≥5%，第三族成本恶化≤5%，同时逐条件/尾部质量无严重代价转移；绝对参考线全部报告，不将小样本完整质量门用作方向一票否决。若机制有用但行为无用，下一建议探索/信用分配，不能增加保持强度找补。机制不成立则关闭该实现。非正确性质量暂差不自动停128批，正确性/预算/空间失败必须保存停止。
