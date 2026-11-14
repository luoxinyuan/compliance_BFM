# Q-only入口释放参考重入修复

E-four-blocks-02（runtime e2de0d00a22ff992cd7f6083fb6e339be8085652）完成500条真实50Hz、8组Q更新、0 Actor/U更新，round0/1/2完整包保留。第三块释放参考因same_state_reference调用env.collect_call，又进入新GuardedEnvironment参考包装，重复mkdir退出。没有真实物理门失败，不重判既有负例。耗261.916016秒/5341物理tick；未提交块内采样不冒充完整checkpoint计数。

修复：参考分支明确调用基础Environment.collect_call，原Task安全/监测callback仍执行，参考后按同状态恢复和相对保护判断；不递归进入外层参考生成。CPU合成重入拒绝测试独立标注，不冒充真实物理验证。

从已提交round2新进程读回，先通过原全物理/历史/RNG/Adam精确恢复，再补round3/4；旧包以不可变硬链接依赖保留，新包独立目录。round4继续采样/更新产生比对证据，另一个新进程再验证同样续跑。未通过不得称恢复资格通过。不改变数据、样本总额、reward、噪声、Q与Actor节奏。
