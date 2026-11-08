# 定向代码审计：冻结历史实现，区分物理复现与调度限制

审计基准 `19b60df8433877859e8d5bc1918671e470cbf5de`；原训练 runtime `b1751077a1a5d59bb08a1a2386cbc7db3e8b20fa`。本轮只读原实现，未修复或重构共享/历史代码。原始证据在独立stage `runs/C-readonly/audit.json`，补充物理复核见A。

|路径|结论|验证及影响范围|
|---|---|---|
|CrossResetState物理/回调对象图|已核：现有固定任务分支恢复有效|绑定源tether/spec/history_handler/live MuJoCo data、apply/measure/physics callback与原始回调；保存mjSTATE_INTEGRATION及可写solver数组、BFM和动作/力/last-z历史。复读#53三个跨reset不干预检查：normal-pre、short-post、normal-post的全部physics/control行与不中断执行完全一致。A四模型同初态完整前缀亦精确一致。|
|快照的任务采样RNG|确认覆盖缺口；不扩大解释|`rolling_snapshot.clone/state_of`跳过NumPy Generator。最小复现：保存含rng的对象，抽一次，put恢复后rng并未回退。Torch CPU/CUDA RNG有保存，NumPy/Python及task/e.rng并未全面保存。当前固定需求接管不调用sample_spec，这不证明34有效标签失效；若用于跨随机任务恢复必须另行修复/测试。本轮没有修改它。|
|其他restore入口|已核当前pilot；历史入口有范围限制|`interactive_pilot.py`已导入CrossResetState；`native_rolling.py`仍用SameReset BranchState但只在同reset搜索分支使用。旧BranchState跨reset负例保存在snapshot_check中，未重新作为正确产物使用。全目录入口清单另存restore-entrypoints-full.json：p4000_preflight/mean_diagnosis也引用旧同reset快照，历史M0–M10初始化工具还直接调用mj_setState；这些并未自动获得CrossResetState的跨reset保证，本轮没有重跑或宣称全项目所有历史入口均已安全。|
|checkpoint/normalizer|已核无误|BC500、5401b4/b7、5402b4文件SHA逐一相符；1639D模型strict按键load_state_dict，center/scale逐值相同。A重评旧三条件数值复现，5402无接触检查补齐。不能以文件名替代此核验。|
|旧开发评价与训练隔离|确认不具备完整隔离|旧evaluate复用同一Task；节点4评价后`task.reset(...)`重新抽训练需求并丢弃原未完回合延续。真实Gaussian采样有独立Generator，但评价时机仍可能改变后续课程/状态流。它不使原确定性物理指标或raw logprob自动无效；不能把改变评价频率当成只增加日志。新A1按本轮明确要求用独立进程评价，并逐次断言训练RNG及物理状态未改变；旧实现未改。|
|评价定义与选模|已核并限制结论|旧#53开发0/4/7、固定末轮7；本轮全部旧模型评价标retrospective。新5403按事前两层规则选候选，独立报告best与末轮。释放后旧anchor误差不作主动跟随成功指标。|
|F(s,a,z)输入|已核源码合同一致，价值含义有限|`m.forward_map(e.obs,z,action)`使用当前真实BFM观测、实际previous-z到首候选的20%插值z、同状态`m.act(...mean=True)`低层action。公共actor/F各自对原观测做一次normalizer；没有把q_PD或高层raw当低层action。|
|B/reward投影及F评分|已核源码；不能称完整任务值|reward_inference先对next-state B作reward加权求和，再project_z到norm16；影子F打分用未投影forward015 embedding除150000、再取ensemble保守值。它是固定机器人状态reward代理，不能覆盖外部伙伴接触reward或整段时变z序列。候选逐行z/action没有显式保存，B审计已说明可重建范围；本轮没有补算或校准。|
|35%混合索引|已核保存资产|1000×128批次与原old_data_only索引一致，后83个旧索引逐值保留，前45替换（实际35.15625%）；34唯一fingerprint，两个窗口14/20，1000批新抽样各22500次。总更新数、唯一标签数、相关后继与独立回合数分开。|
|去重/标签质量|已核有限域，不能过度概括|34=2个Student起点+32个实际Teacher恢复后继；旧40受影响标签保持隔离。没有把去重后的34当34次独立Student偏离，也没有因一次接管有效宣称整域Teacher有效。|

## 对既往结论的影响

本轮没有发现需要撤回BC500或两份batch4固定物理结果的新数值缺陷；本轮重评复现这些结果。快照RNG与旧评价调度是明确的使用边界：不能把“完整固定物理恢复”泛化为“任意随机任务生成器也恢复”，不能把原评价当作不影响后续训练状态的旁路。原始结果与runtime保留，不将这两点宣布为batch7退化的已证明唯一原因。

当前不实施快照RNG修复；若下一轮需要修复，应在独立小轮增加随机任务恢复回归，并复算一个归档物理评价。新A1的独立评价实现属于用户本轮明确要求，未修改历史模型、PPO数值合同、reward或物理。

## 审计产物的后续修复合同

本轮没有把存疑项顺手改成新算法。两项明确限制均保留最小复现或源代码位置：随机采样器不在快照覆盖内；旧评价改变训练Task的生命周期。若未来修复原helper/runner，必须另开小轮，加入对应回归，并复算一个原归档物理条件。当前A1新runner仅按本轮指定的评价隔离要求实现，不将其收益归因成已证明修复了batch7退化。

## 可定位的源码/复现入口

- [CrossResetState](interactive_snapshot.py)、[原同reset快照](rolling_snapshot.py)、[原跨reset正反检查](interactive_snapshot_check.py)。最小RNG复现与模型/索引审计脚本见[retention_code_audit.py](retention_code_audit.py)。
- [原评价/训练生命周期](interactive_increment_ppo.py)；本轮隔离评价由[retention_ppo.py](retention_ppo.py)实现，评价前后RNG及物理指纹证据单独保存。
- [F影子评分](interactive_pilot.py)、[冻结FB公共接口](../humanoidverse/agents/fb/model.py)；完整配对资格见[ranking_eligibility.py](ranking_eligibility.py)。
- 审计原实现的commit为19b60df；本轮新增报告/审计脚本不改变那个历史runtime。

## 新A1汇总字段的记账说明

`retention_ppo.py` 返回的旧式 `student_optimizer_updates=0` 字段不能被解释为“PPO没有更新”。本轮监督BC更新确实为0，但实际actor接受步必须由逐批 `learning.json/optimizer_updates` 求和，RL交互由保存的rollout统计计数。保留原runtime/原输出，结项evidence明确另列真实actor接受步、critic更新和RL决策；不通过这个命名含混字段判断训练量或学习收益。该问题不改变动作、概率、物理或选模结果。
