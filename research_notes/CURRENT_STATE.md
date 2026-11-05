# CURRENT_STATE — 新会话先读

最后核对：2026-11-05 CST。**Issue46两seed真实反馈PPO及封存评估完成并停止。无训练、Student迁移、原生reward inference或下一阶段任务排队。**

[完整结果](COEXIST_RESULTS.md) · [预登记协议](COEXIST_PROTOCOL.md) · [Issue46](https://github.com/GaoE05/robot-research-bridge/issues/46) · [Project Master18](https://github.com/GaoE05/robot-research-bridge/issues/18) · [BFM原生复用最小路线](BFM_REWARD_REUSE_ROADMAP.md)。RESEARCH_OVERVIEW是长期纲领；ChatGPT Library2026-10-28是历史快照。

## 科研目标与可靠基线

Contact-conditioned whole-body behavior selection/modulation/transition：依据真实接触与身体状态，合理组织局部手臂、躯干/root及支撑响应；不预设先手后脚，不以root走远替代任务合适性。冻结BFM/PD。有限直行Teacher→ideal-force Student、原生局部伸臂、有限双向转腰均已成立；没有验证真机精确外力感知。

- BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。
- 实际行走基线I `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`。
- T30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`；S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`。
- D2 `9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5`。
- C0局部有效但长距失败；C1恢复walking却失去局部优势。Issue45 F/R均未稳定共存，冻结I参数不等于闭环保旧。全部旧权重/失败保留；不按任务拼模型。

## Issue46实际结果

1. 原reward同条件复算认可C0局部和I长距有效解，未改reward/网络/历史/绝对z/BFM/PD。10N主物理、K80/D8、原观测尺度；5N回归。两个seed同I actor，独立fresh critic/Adam，std=.008，各32×768；无新增监督初始化/Student。
2. 两组固定开发规则均选择iteration0。末轮分1.619/1.612，I1.500；预登记联合延长条件不成立，未追加训练。best两个文件不同critic SHA，但均值actor/归一化与I相同，最终评估去重，不能算新学习成功。
3. **部分正结果保留**：同一4601末轮在新6/24/72cm误差4.190/4.617/5.478cm，I4.452/5.290/7.632cm；72cm真实力5.877→4.511N。4602长距改善但新6cm退化4.790cm。不是两个seed稳定共存。
4. 停止取舍：新共同前缀停止4601为4.956cm/4.038N，I4.387/3.602；旧早停止6.062/4.967 vs4.707/3.889。退化发生在仍接触的移动/保持阶段，不能用释放后自然收步解释。共同前缀7.37s前实际physics/命令一致，分叉后可反馈调整，但不是最优行为分配。
5. 动作质量限制：新6cm4601足部路径.33/.75cm，远少于I20.80/.80cm，却伴随root降低6.77cm；世界腕相对肩X仅+.90cm，C0+4.32cm且误差仅2.293cm。减少脚步不等于自然伸臂已经成功。继续保留XYZ、相对运动、滑移和连续性。
6. 242完整审查回合无安全终止；seed4602中距有共10个无支撑标记tick（单段10–15ms），未隐去。最大q_PD20ms跳变68.29°不等于实际q5ms3.60°。无接触稳定，释放后不考核旧anchor跟随。
7. 探索噪声能改变同状态BFM动作，未证明std太小；但单步方向覆盖不等于持续有效反馈。raw logprob复现≤.0001221，联合KL≤.01996，critic EV提高不能代替actor成功。每seed208完整采样回合，减速决策仅136/213；覆盖及优化共同限制尚无唯一根因。

## 下一步唯一建议（尚未执行）

不原样无界续训或再次重复F/R监督。下一轮做一个小型**同权重BFM原生reward-prompting局部行为来源实验**：固定buffer/seed/采样预算和状态reward，取得少量局部伸臂/站立稳定候选，在同起点近/远任务进入、保持、退出中与I/C0物理比较，检验能否提供比独立高维噪声更连贯的有效行为来源。

这不是宣称当前任意force/anchor reward可零样本prompt；原B/回放不含外部伙伴信息。reward composition/F-SF候选后果排序已列路线，未运行；force-history0/1/2/4/8/16也未展开。若新来源不足，不扫描大量z或恢复手写分类器。需要用户下一阶段授权。

## 身份、成本与接力

训练runtime `f6ca5c5f6c4de205b3457bef4a024345e2105009`；最终评估/渲染 `057f5c6bd302185097770c3c1ed4d10dfb2ce109`。长期个人分支`gaoe05/research`；最终文档commit见Issue46链接，勿混作训练runtime。

- 4601 last32 `8fa4759d3766568dee8fbba0367e772a97977694d19a503868708efa6d8c9228`。
- 4602 last32 `e9a1972166b25e3b915b8d75e97d3377e540583e215d91f269e8dba4550310c9`。
- [完整模型清单与服务器路径](coexist_artifacts/final-frozen-models.json)；[资源账本](coexist_artifacts/resource-closeout.json)。
- Run：`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105`。初始/每8批候选/best/last、原始rollout、所有失败均保存。
- 总49,152高层RL决策、194实际Adam更新；0新增Student/监督更新。7711.53 GPU-job秒＋165.02分析秒＋180秒保守CPU余量，约4.24GiB，低于授权18,000秒/12GiB/98,304决策。训练两独立worktree未修改。
- Mac：`~/Projects/compliance-bfm-diagnostics/videos/feedback-coexistence/`，仅3个1×真实qpos MP4＋INDEX；6cm、72cm四模型对照，以及同4601停止/继续。SHA/时长/完整解码/视图已核对；未挑seed重跑。
- owned进程均结束，GPU2/5/6释放；共享checkout仍`6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`干净。未改BFM/PD/原权重/main、SoftSONIC/他人任务，未做真机。

新会话先读本文件、RESEARCH_OVERVIEW、COEXIST_RESULTS及Issue46；核验实际运行/模型SHA与新授权，再检查GPU/CPU/RAM/I/O，不复用旧空闲快照或PID。历史Issue44/45、M41仍有效，不能用这轮局部正例抹去停止与跨状态失败。
