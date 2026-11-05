# Issue46 真实反馈局部—行走共存协议

预登记2026-11-05，机器配置与精确任务见coexist_protocol.json。I actor完整SHA72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a，原BFM/PD/10Hz→50Hz→200Hz不变。10N主任务、5N回归，不改reward/观测尺度/std=.008/网络/绝对z动作空间。

## 前置证据与启动

已保存同条件轨迹独立复算geometry奖励，原先每control legacy terms不是PPO使用的完整reward；逐physics tick重算force/tracking/relative velocity/heading/uncapped，结合未修改的slip/tilt/continuity/idle，和rl_stats逐步对齐。C0局部9cm回报−1.265优于I−3.452；I长65cm−12.024优于C0−47.692。故本轮无需经济性reward修改。

20个实际I访问状态，每状态256个std=.008样本，比较I/C0同状态完整五子步q_PD与合法z；只作方向覆盖证据，不将同状态C0查询当物理恢复真值，也不把z距离当概率。I随机完整9/65cm物理误差4.422/7.166cm，长距随机比均值差，保留该边界。

## 两seed训练与保持

4601/4602独立seed，均从I actor及normalization/std出发，新critic和Adam。每批3workers×256=768高层决策，首段32批/24576决策每seed。每8批完整多条件均值开发，初始/候选/最佳/末轮保存。保持仅限真实I长距起步4–7.5s以及3个已确认有效恢复起点的短段，使用原I同状态raw均值函数保持、权重.002；不对小请求强加walking标签。记录保持/PPO梯度比例与夹角、jointKL、clipfraction、raw/z变化、actual coverage与分项成本。

沿用已验证raw logprob、gamma=.99、GAE=.95、transactional Adam+parameter回滚；不把法向投影后z当高斯概率坐标。可靠域raw保持不等于真实物理保旧，另用完整开发与终检测量。

开发选择：0.5×最差相对原物理cost +平均相对cost；出现安全边界或任一cost>初始1.75倍视为开发破坏，回到开发best参数及Adam状态，记录并降低actor lr。该标准不是声称1.75倍退化可接受，仍逐任务公布退化。不得按终检选择。

两seed均32批后，只有两者开发best共同满足：score改善≥5%、local7/9平均error改善≥10%、long65/80最差相对cost≤1.15且无boundary，才在总预算内按同一规则延长到64。否则不自动续训；最终测试只在此共同决策后读取。

## 训练单位与权限

近/中/远、连续扩大需求、同前缀stop/continue、自然arm-ready、零需求/无接触、早晚停止及左右转腰。回合概率不等于实际决策数，记录各family/phase实际计数和完整回合。eval打断的非终止采样段按原bootstrap处理，不能计作完整训练回合。

arm-ready只在接管前由原生goal真实执行生成；保留真实q/dq、BFM/力/实际上一z及solver历史，执行策略不读取源名称或时间。特权actor可读当前误差/相对速度/方向需求，不能读最终距离、未来轨迹或变化时刻。当前阶段是Teacher，不把输入权限转给未来Student。

开发7/9/18/22/65/80cm等均已暴露。新6/24/72cm、onset4.93、分叉7.37、decel.55以及ready_progress.55的组合预先封存，不按其调参。

## 评价与停止

同checkpoint逐任务/最差/Pareto结果、实际XYZ力/限幅、腕相对肩/躯干、root及足路径、支撑/滑移、q_PD与实际q连续性。接触前后/移动/减速/保持/真正释放分开；释放后不考核旧anchor。保留I/C0/C1/F/R，禁止按方向或距离拼模型。

总≤98304新增RL决策、18000累计job秒、12GiB、最多3实时空闲GPU。不训练Student、不运行reward prompting或SF planner。普通启动脚本引号故障在运行前修正，旧文件/日志保留，尚无RL样本因此不称训练失败。

若缺乏共同收益，按reward排序、实际探索、更新和闭环阶段定位，不原样续训。下一阶段若证据支持探索来源不足，优先有界BFM reward prompting；具体来源研究见BFM_REWARD_REUSE_ROADMAP.md，当前不执行。
