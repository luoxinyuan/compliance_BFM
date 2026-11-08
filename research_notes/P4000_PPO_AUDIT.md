# Issue52 PPO实现审查：学习更新、critic耦合与奖励信号

2026-11-08中途只读审查，不是阶段结项或因果修复实验。[Issue52](https://github.com/GaoE05/robot-research-bridge/issues/52)。实际训练runtime `93383a406e4b4923418af0b47aca2077ca68e40c`；审查只读源码及seed5201的32批/24576实际RL决策。没有改active worktree、物理、reward、std、KL、优化器或队列；seed5201已完成32批，seed5202按原计划独立运行。

## 核心控制合同与未发现的问题

P4000 actor/normalizer → 1088D当前特权观测 → raw Gaussian采样一次 → norm16合法256D z → 5个真实50Hz BFM子步/原PD → 200Hz MuJoCo/Tether → 实际reward → PPO。没有在线I/C0/T30/M10/搜索动作覆盖，没有force-to-motion规则、固定技能菜单、小residual半径或锁腿。stand仅norm<1e-6的数值fallback，不是默认参与上限。

BFM/PD冻结是研究合同。actor不含未来/任务ID；critic与actor网络参数独立，但当前读取同一观测。raw-logprob原阈值3e-4未放宽。逐worker重算GAE后，与记录的家族归一化advantage均值最大差6.18e-7（排除首次value预热前保存值），未见worker/time错配或GAE算术符号错误。这不是所有实现均无bug的证明。

## 已证实的更新耦合：actor KL同时停止独立critic

`clean_policy.py:39-68`中同一inner loop计算pg+vl，Adam同时更新actor/critic；actor全批joint Gaussian KL拒绝时，两者参数与Adam状态一并回滚。KL>.015时退出整个循环，而不只是actor。

32批名义6 epochs×3 minibatches=576个更新机会，实际仅68个接受步；83个候选回滚；32批全部提前停止。critic除第1批100次value-only预热外，也只获得这68步。**提前停止actor是正常PPO保护；将其同时绑定到独立critic并非概率正确性要求。** 这属于可解除的工程耦合，不是已经证明唯一根因的数学bug。

首轮value预热拟合的是含新critic bootstrap的GAE目标；高初次EV不等于独立完整回报已经可靠估计。已保存数据重算显示分阶段value/advantage存在系统差异，值得检查，但bootstrap MC尾部仍取保存critic，且不含未知未来的观测本来就有任务不确定性。因此不能由“无接触advantage偏正/接触前偏负”直接宣布优势方向错误或critic失效。

**最小下一对照：保持actor概率空间、KL阈值、物理、reward、std、网络完全不变，将critic拟合与actor KL停止/回滚分离；相同真实RL决策下比较value校准、优势及均值物理曲线。** 当前两seed不插入该变更；另行登记才执行。

## 已证实的有效更新尺度；尚未证明阈值应放宽

固定raw std=.008，平均采样相对均值z夹角3.573°；每批接受更新后均值z变化平均0.03489°。joint KL末值均在.01513–.01990之间，平均clip fraction .444。actor nominal lr每批重设3e-5后多次回退，说明信赖域实际活跃。

这是“小幅均值改变”证据，不是“小residual动作空间”证据。固定小raw噪声下joint KL限制raw均值变化，且raw范数会影响投影后角尺度。不能将256D总采样角度当成某个有利方向的探索幅度；不能凭这些数字证明必须增大std/KL。梯度裁剪比例约.00097也不能直接当Adam参数步长缩小1000倍。

## Reward与力限幅：不是force权重明显过小

实际integrated cost合计中force约89.3%（2718.284），uncapped162.912、tracking56.691、heading56.796、relative velocity19.737，其余更小。当前实际力项为平方成本，没有待调的Gaussian force sigma。

实际接触时间25.80%达到10N限幅；移动28.21%，减速49.69%。力项在10N以上的需求区不再区分大小，但uncapped与tracking仍有较弱反馈。因此“整回合reward正确排序好坏”不等于“每个小改善方向均有强信号”。约74%接触时间未饱和，不能将全部失败归因cap，更不能自动增力或同时改reward。

另一个设计假设：腕任务heading权重使用spec.distance，不是仅当前反馈。它没有作为actor未来输入；当前同前缀停止/继续的distance相同，但跨最终距离回合的相同前缀可能有不同heading权重。这不是已证实主因，后续如研究纯因果统一目标应单独审查，不在当前实验静默修改。

## 覆盖、时间信用与输入裁剪

24576决策中：移动6218、减速343（合计26.7%）；释放后8685、无实际接触2342（合计44.9%）；接触前2587、保持4401。episode采样比例不是决策/梯度比例；关键过渡覆盖可能比想象少。释放恢复和无接触仍必要，不能简单删掉。

10Hz下gamma=.99、lambda=.95，GAE直接trace权重在1/3/5秒约.541/.159/.0466；长期过渡更依赖value bootstrap。这不是频率错误，也不能单凭trace数值认定lambda不合理。

对24576×1088保存输入做center/scale纯算术检查：非有限值0；归一化clamp截掉0.0654%的元素，15.1%的决策至少一个维度受截。原始最大归一化幅值68.81。没有大面积裁掉所有观测的证据；关键维度的物理含义和敏感性尚未因果验证，不能因此宣布完全无影响。未运行任何CPU替代BFM或新的GPU实验。

## 当前物理结果与下一判断

seed5201完成32批；开发正式best仍batch8，score4.90009对初始4.91678，仅很小改善。batch32不是新推广基线，正常72cm接触期物理3D误差12.306→12.943cm、力7.939→8.287N；中距亦退化。不得将训练reward波动解释为稳定收益。

建议优先级：①解除独立critic与actor KL的非必要耦合并控制比较；②在value/优势更可信后判断有利动作能否保留进均值，再选择探索或更新尺度对照；③若有用动作已采到但饱和区难区分，再做单独局部reward灵敏度对照。当前没有证据支持同时增力、放开KL、换动作空间、改reward或盲目延长PPO。

## 可复算与证据边界

[p4000_update_audit.json](p4000_update_audit.json)仅小型统计，无权重/原轨迹；[脚本](p4000_update_audit.py)只读现有stage buffers，在项目venv运行即可，不创建物理或学习任务。normalizer结果为已保存buffer算术，未进行网络前向。完整私有原始证据在stage/audit-readonly-20261108及runs/seed5201-32，训练runtime与文档commit分别记录。当前无新的修复训练、无Student、无新视频成功宣称。
