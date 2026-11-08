# Issue52 — P4000热启动特权Expert真实反馈PPO

[Issue52](https://github.com/GaoE05/robot-research-bridge/issues/52) · [预登记协议](p4000_protocol.json) · [实时状态](CURRENT_STATE.md)。源码推送不代表训练成功。

P4000 actor/center/scale从冻结SHA9f560903072b355e6d2af537af725b9e61210ecea87a56151aa5d59dff45d164原样载入，critic用seed+17新初始化，Adam fresh；不使用BC中的critic，不在线调用旧I/C0/C32/搜索动作。两个seed5201/5202先32×768；开发证据及资源均满足协议时共同延至64。当前reward/网络/动作/物理/历史不变。

预检：均值短请求2.510971cm/2.075716N、较快长牵引13.635801/7.299315，与Issue51相同；原始每维std.008采样短请求2.414217/2.019794、正常牵引8.823717/6.476469，无安全终止，因此没有启动预登记.004修正。FP32 batch-one/batched联合logprob最大差.00012207<原.0003，未使用PPU路径/放宽断言。少量完全同状态第一命令干预后运行P4000反馈1.2秒，5.2s随机动作略好、搜索首动作更差，6.2s各分支接近；搜索完整反馈有效不等于任意单个首动作可独立复制收益。预检不是PPO样本。

raw Gaussian在1088D当前特权输入上输出256D raw，采样一次，然后原direct_command归一为norm16 z；真实10Hz高层、5个50Hz BFM子步、200Hz物理和原PD。没有小residual硬半径。std固定，概率空间是实际raw；保留原logprob<3e-4、clip±.15、joint KL≤.02及参数/Adam双重回退。无额外BC/replay约束。

第一批768真实样本用于100次critic-only更新，不额外采样，actor/normalizer函数SHA必须不变；用更新后critic重新计算每个worker的current/next value和GAE，再以原actor旧logprob进行PPO。有限回合自然结束为terminal；batch截断的done=False使用next-value，GAE先按worker时间序列计算再展平。元数据严格按[time,worker]展平对应，避免按任务统计advantages时错位。采样时保存next_obs/raw/logprob/reward/done/discount和物理统计，不能把expert覆盖动作算on-policy。

训练分布near/mid/long/stop/zero/none/arm，权重及范围见协议；伙伴需求真实作用于物理，onset时一次性实际对齐，之后不追随机器人。当前特权actor可见当前error/relative velocity/support/force等，不能见最终距离、任务ID、未来/停止时钟/搜索phase。source标记仅环境账本；goal_arm5源为真实前缀，保留solver/BFM历史/真实last-z，不qpos-only重置。

开发7条件/每8批确定性评价，0/16/32/48/64另存采样执行；选择以物理绝对单位代价加最差退化项，不用近零基线比值。明显安全/任务崩溃按共同预登记规则恢复best+Adam并降低actorLR，真实sampler状态保留，下一批重新实际采样。保留init/best/candidates/last；新终检仅一次且不重选。收益来自训练时以任务/阶段实际覆盖、正负优势概率变化、均值z变化、KL回退和物理曲线交叉判断，不能只看EV。

资源：单卡既有NVIDIA环境，3采样worker、两seed顺序，不自动增卡。stage experiments/p4000-feedback-ppo-20261108；18,000累计job秒/10GiB/98,304RL上限；所有失败、预检、开发/终检/渲染均计入。runtime93383a406e4b4923418af0b47aca2077ca68e40c，预检b6fcbab4f15799436b6ea8e489067baf6ffcdf18；文档HEAD非物理runtime。有限p4000_sequence只执行本协议、完成后停止，无后台长期watch/新Bridge。当前仍训练中，无PPO成功结论。
