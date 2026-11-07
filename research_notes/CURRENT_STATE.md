# CURRENT_STATE — Issue50原生候选＋短程滚动反馈进行中

核对2026-11-07。[Issue50](https://github.com/GaoE05/robot-research-bridge/issues/50)为本次新授权；Issue49正式结项，不续旧PPO、AR网格或C0专项。

## 可靠基线与本轮定位

旧I/C0/T30/S0、有限域Teacher→ideal-force Student及所有失败保留。Issue48/49原生独立高层PPO已跑通，但未获得及时行走，不改写其开发选择或终检。当前检验原生来源＋短程真实物理优化能否获得有效反馈，不训练新PPO/Student。

冻结BFM SHA256 33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361、原PD、Direct256D、10/50/200Hz、K80/D8、总10N。I/C0仅冻结参照；实际滚动搜索不调用任何旧高层actor，连C32也只作对照。

## 当前实际完成

- 原生初始化d399174c、C32 39cd0db5、I72877017、C0ffe0eb06及goal/motion来源已核SHA。三速度平均.09/.15/.22m/s均为72cm，smoothstep峰值1.5倍；接触瞬间完整standing状态一致。
- 同权重原replay完整SHA46eac53f...3b546b已重新校验。原采样器seed5001同150000有效successor，3固定机器人状态reward×普通/softmax加权得到6个合法z；保存未投影embedding，和原API最大差0。不是外部force/anchor零样本prompt。
- 18个匹配来源物理回合＋2个无力对照已完成。主.15：Standing47.105cm/9.169N，native motion25.195/9.171，I10.265/6.682；原生goal局部7cm任务2.548/2.098。motion在.22误差19.256cm，.09约60.523cm，但真实三轴负担/侧漂仍大，不叫合格接触专家。
- 新ordinary15提示主.15约28.752cm/9.396N，weighted约47.540/9.424；前者负载中有足部重排，无力时移动段root仅约1.35cm、足路程约1.40/.50cm。区分负载反馈/受外力影响与可靠自主walking，不宣称学成新行走Teacher。
- 四状态有限搜索有预测改善；预测不是实际闭环成功。完整机械/solver/BFM/实际z/力历史恢复已测试，修复预测期onset对齐可能修改闭包原点的边界问题；正式滚动版本同时保存并校验该原点。

## 正在执行与权限

实际滚动runtime32b3502262399409dd9091b067c89ba619d20873：7原生提议、2.4s窗口、每100ms只执行首指令并重规划；常速度预测只取当前伙伴状态，删除最终距离/停止/释放元数据。两条同前缀停止/继续将完整执行并保留释放恢复。当前尚未结项，不能说MPC已优于基线或已有新Student。

H100级后端发生外部重启和挂载变化；旧隔离目录暂不可见。新挂载现有Torch2.7.1+cu128/MuJoCo3.3.3差异检查、BFM/PD实际执行、精确backend内前缀和12概率回归通过（最大.00019837<.0003）。没有依赖安装或共享环境修改；不将先前cu118/3.3.7输出混成严格因果对照。

总上限仍18000job秒/10GiB，无新RL/Student。旧已知673.3848s加中断作业850s保守预留已计入新账本；没有重置预算。只使用实时核验空闲的单H100，保留所有工程失败和外部中断。

## 源码与接力

长期个人分支gaoe05/research；本轮运行树独立固定，文档在独立worktree维护。代码/文档发布是否成功以远程核验为准，源码提交不等于实验成功。最新证据先看Issue50；Master18保留历史，阶段结束集中更新。

接力读取本页/OVERVIEW/Issue50，核对实际账本、boot/GPU身份、PID/start_ticks、runtime与checkpoint，不能见旧running标记就重启。新原生prompt是运动候选，不是反馈专家；预测分支不作on-policy样本，理想仿真力不等于G1实机传感。

阶段结束交付2–4条真实qpos1×关键视频至Mac并写精简INDEX；当前没有完成本轮新视频交付声明。不改main/共享BFM/PD/权重/SoftSONIC/他人任务，不做真机。
