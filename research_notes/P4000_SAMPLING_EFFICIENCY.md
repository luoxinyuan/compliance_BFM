# Issue52：采样提效与学习质量分开验证

2026-11-08用户授权尽量跑得更快、学得更好。原两seed继续，不改变任何运行中源码/进程；critic配对科学变量仍仅为独立value拟合。当前没有实测新提速或学习改善结论。

## 已核查的当前瓶颈范围

当前3采样worker、768决策约110–124秒，actor优化约.01–.04秒；GPU利用率约96–97%、显存约3.4GiB，CPU20核额度中约3核持续采样，RAM余量约181GiB，CPU/I/O pressure在核验时为0。利用率不等于算力占用率，遥测不证明所有算力饱和。耗时集中在采样，但现有计时没有完整分开MuJoCo、BFM、history与同步，不能擅自宣布单一瓶颈。旧OmegaConf缓存优化已经启用，不作为本次新收益。

## 本次只尝试一项低风险优化

clean_env.Task.step每个50Hz子步除实际latent动作，还调用同一冻结BFM、同一观测下的standing动作，用于nominal_q_PD及qPD_vs_prior_rms_deg诊断。当前soft_prior=0，奖励、actual q_PD、PD、下一观测与策略均不读取该nominal动作。BFM.act(mean=True)不采样，normalizer在eval_mode，不更新统计。删除这项重复诊断有明确源码依据；最终仍以真实物理对照验证。

默认BFM_RECORD_NOMINAL_DIAGNOSTIC=1，仍计算全部旧诊断。仅新私有运行树支持0；这时nominal_q_PD标None、nominal_diagnostic_recorded=false，与standing动作差的指标省略，绝不伪造0或沿用旧值。实际BFM观测/已执行z/action/q_PD、力、支撑、q/dq、动作连续性及每8决策的同状态五子步mean-vs-sample诊断继续保留。必要时可从保存的BFM观测离线重算省略的standing比较，但不能当已测值。

## 有界测试和采用规则

在原两seed及原封存评价完全退出后、两个critic修复seed启动前，若24000累计job秒余量能同时保障两组完整32批和终检/分析，允许一项至多600job秒的GPU/MuJoCo测试；失败与诊断成本照计。否则跳过，不牺牲critic共同预算。无新增RL训练决策，无新GPU或并行抢占。

固定冻结P4000，四完整条件：normal72均值、short08采样.008、stop66均值、none采样.008；AB/BA交替，各自相同物理初态/随机序列。比较全部200Hz物理行的canonical hash、每100ms完整MuJoCo积分/solver状态、BFM观测/历史、1088D输入、上一z、实际action/q_PD、reward及真实动作。全为0差才算工程等价。

每回合前16个真实决策进行cProfile；它们全部参与数值一致性比较，但不参与吞吐比。其余Task.step用CUDA同步后的wall计时。分别报告reset与常规step；profile inclusive/exclusive分开，不累加重叠inclusive，不把CPU dispatch计时当CUDA计算时间。统计真实BFM act次数。单sampler测量不等于3worker端到端收益，后者从实际训练batch采样秒数/决策率继续核对。

全部数值一致且aggregate未profile步提速>=1.05才在两个新critic seed共同启用；否则保留完整诊断。采用只看吞吐和零差异，不看任务评分、终检或新臂谁更好。原控制程序和PPO概率/梯度不变。无需也不允许通过扩大网络、batch、worker数、缩短历史或改reward制造数字。

## 后续条件化判断

若有效，先记录当前端到端增益并完成critic配对；若无实质收益，保留负结果和cProfile，下一轮再考虑跨环境BFM批处理。增加worker会改变片段长度/任务覆盖，须另做有解释力的工程与概率比较，当前不静默采用。当前不建设central inference框架或迁移GPU仿真平台。

额外保护：两个新critic seed的首批实际768决策，在任何PPO更新前与原同seed首批保存数组逐值比较obs/next_obs/raw action/reward/discount/done；若不一致保存并停止，避免工程差异混入critic归因。该检查不要求训练后的轨迹继续相同。
