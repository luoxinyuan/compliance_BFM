# Issue66 — P1优化合同只读审计（2026-11-15）

本轮只读存档与源码、形成待签稿。**无新模型前向／反向、Adam提议、rollout、物理或优化规则执行**。权威独立审查：[6893703131](https://github.com/GaoE05/robot-research-bridge/issues/66#issuecomment-6893703131)。源码基准 `02da99fcf26e638fd42d6516fd5aff2bdda9d821`。原v2正文、P1失败、梯度erratum和所有工程账本不改。

## 1. 实际资产资格与新增只读统计

完整读取并校验逻辑checkpoint的delta链和全部数据segment的SHA/长度：E/block4与I/block3；解码真实模型、Adam、RNG/物理历史与4,096/3,072行。未实例化优化器或Task。完整原ledger的32项费用重新求和7807.703623851936秒，与原总数及P0/P1/P6/reserve分账逐值相同；原账未改。两个CPU读包内部用时24.930084/22.541928秒，包含在本次400秒整体保守准备账中，不重复相加。初次未发布的统计脚本把none的inactive错标release，第二次按真实episode spec修正；两个私有原件保留，以下与公开JSON采用第二次结果。

|实际记录|E|I|
|---|---:|---:|
|真实工程条目|4096|3072|
|访问父组／回合|7／8|5／6|
|完整回合|7|5|
|实际option随机抽样|83|41|
|P3 HOLD抽样：standing/arm5/arm15/motion|54/0/1/1|37/0/0/1|
|P3 FOLLOW抽样：standing/arm5/arm15/motion|2/4/19/2|0/0/3/0|
|tail / FiLM / selector非空Adam张量|0 /0 /0|0 /0 /0|
|Goal encoder / predictor / decoder非空Adam张量|0 /6 /不适用|4 /6 /4|
|value非空Adam张量|6|6|

不同帧、stop/continue兄弟不增加独立父组。I尚未覆盖第二P3块与第六个完整回合。旧1024失败条目没有完整raw包，仍入8192总帽，不能从这些现存包重建它。

按真实spec与physics索引的阶段行数：E none480、precontact255、moving1155、stopping75、holding765、released-recovery1366；I分别480/195/825/35/550/987。它们是50Hz行数，不是独立状态数或完整成功段数。连接zero保留在相应真实物理时窗，none不算release。

只读以FP64公式复算全部7168个原Gaussian采样logprob，超过原3e-4断言的行数为0；不是新GPU batch-one资格。每块原实际联合logprob误差都是0。详见[机器摘要](contact_goal_ei_v2_optimizer_archive_audit.json)，其manifest/report SHA可定位私有原包，不公开原始状态或权重。

## 2. 概率合同：有依据的结论

[research/interaction_fb/ei_v2_ppo.py:27](https://github.com/luoxinyuan/compliance_BFM/blob/02da99fcf26e638fd42d6516fd5aff2bdda9d821/research/interaction_fb/ei_v2_ppo.py#L27)、[research/interaction_fb/ei_v2_ppo.py:59](https://github.com/luoxinyuan/compliance_BFM/blob/02da99fcf26e638fd42d6516fd5aff2bdda9d821/research/interaction_fb/ei_v2_ppo.py#L59)、[research/interaction_fb/ei_v2_ppo.py:74](https://github.com/luoxinyuan/compliance_BFM/blob/02da99fcf26e638fd42d6516fd5aff2bdda9d821/research/interaction_fb/ei_v2_ppo.py#L74)：实际每个100ms段的logprob为真实抽样边界的categorical一项（多数段为0）加最多五个29D raw Gaussian的和；29维已在`agent.logp`中求和，外层`.sum()`并未再乘29。只按记录的old历史、option与插值z重算；不把新策略另跑的状态当同状态。Gaussian raw采样一次，执行clamp，score-function使用实际raw随机变量；**不是clamp动作的普通Gaussian密度**。

KL以各段条件Gaussian解析KL之和＋真实draw处categorical解析KL，最后除实际100ms段数；不是每维平均、不是每20ms帽，也不是每个option单独.02。每块205/206段，行政块可能切开命令，短段按真实子步数算，不能强补五步。actual categorical conditional-KL还应另除真实draw数报告，以免大量hold槽稀释；gate分母不改。

历史E/block4第一提议：Gaussian .18824948258336308、categorical 2.7811622260839427e-9、joint .1882494853645253、sample old−new .1515213838299212。真实joint帽的9.412474倍几乎全来自Gaussian。全56次Actor提议均正确回滚，不能称更新小但已学会。现存P3 categorical FP32求和偶有约−1.7e-8舍入残差；只记录并在未来诊断附FP64核算，不用负估计抵消Gaussian或偷偷修改gate。

满五步近似解释：KL=.02对应均值差RMS `.01*sqrt(2*.02/145)=.0001661`，该提议约.0005096 normalized-action。**仅为维度尺度解释，不证明α²规律、动作安全或具体α能通过**。全部候选实际KL需重算。理论上缩小步长有望降低KL，尚未测得成功α。

## 3. 信用、value与“动作变化”

[research/interaction_fb/ei_v2_ppo.py:35](https://github.com/luoxinyuan/compliance_BFM/blob/02da99fcf26e638fd42d6516fd5aff2bdda9d821/research/interaction_fb/ei_v2_ppo.py#L35)：现实现先将20ms已积分reward按gamma20累计成100ms R，再在段起点/段末V上做宏观GAE；完整段gamma=.99、lambda=.95。真实terminated不bootstrap；truncation/bootstrap与episode重置分开。保存GAE与源码一致；不是对5个native动作分别算20ms advantage。一个段内五个动作和该段抽中的option共用同一标准化advantage。

宏观GAE允许后续段reward经trace影响option，不是“option只看到100ms”。但段内早晚低层动作信用无法分别分配；native GAE在lambda<1时一般不等于宏观GAE。500ms option的后果跨段，段起点V又没有显式held-option/age输入（deploy有真实previous-z），可能增加bootstrap/基线误差。**这些是结构性风险，不是已证实的信用失败；新20ms V/GAE及完整行为干预尚未执行。** 任意未选动作的反事实回报无法从这些rollout确定。

value确实被训练而Actor未动：E各块raw advantage中正值数5/205、56/206、77/206、71/206；I前三块相同。正号并不等于干净有利动作；标准化后更不能拿正号当物理成功。critic质量不能只靠这些数下结论。当前日志没有原生20ms value序列、分模块梯度或提议后的完整surrogate，因此“20ms信用一定更好”“selector梯度冲突”均未证。

[research/interaction_fb/ei_v2_ppo.py:120](https://github.com/luoxinyuan/compliance_BFM/blob/02da99fcf26e638fd42d6516fd5aff2bdda9d821/research/interaction_fb/ei_v2_ppo.py#L120)：当前`policy_loss/clip_fraction`在`Adam.step`前算；所有拒步恢复到old策略，故历史clip_fraction全0。它不代表被拒绝提议的ratio温和。缺少proposal后的surrogate/ratio/clip与逐父组动作数据，是下一次有限诊断必须补的记录，不补造历史。

[research/interaction_fb/ei_v2_agent.py:57](https://github.com/luoxinyuan/compliance_BFM/blob/02da99fcf26e638fd42d6516fd5aff2bdda9d821/research/interaction_fb/ei_v2_agent.py#L57)：tail/FiLM/selector仍是初始张量。FiLM和selector末层全零，辅助头虽然学习，当前无法经这两个零末层改变执行函数。因此**已接受的均值动作/q_PD变化为0**；“原提议相同状态的输出变化”存在但未执行，也不能当真实q变化。逐proposal未保存逐关节变化，无法从存档确定其关节归因。

## 4. selector为何尚未学习，什么尚未证明

[research/interaction_fb/ei_v2_agent.py:67](https://github.com/luoxinyuan/compliance_BFM/blob/02da99fcf26e638fd42d6516fd5aff2bdda9d821/research/interaction_fb/ei_v2_agent.py#L67)：HOLD先验(.97,.01,.01,.01)，FOLLOW(.10,.10,.70,.10)，四项始终可选；并非永久动作mask。deterministic argmax要换离主先验，HOLD需约log97=4.575的相对logit改变，FOLLOW约log7=1.946。随机执行不要求跨这个argmax阈值。

已证实有三重限制：(1) 同一次事务Gaussian超KL时把selector一起回滚，实际selector0步；(2) I只有3个FOLLOW draw，替代option观测几乎没有；(3) 当前cat项是整个100ms分母下的很小提议变化。**不能由cat KL接近0证明联合KL本质禁止selector、或先验不可克服**。同样，uniform缩全部模块可能使高层更慢，但当前没有其已接受步的实测。

下一次须区分：模块梯度是否为0；Adam原提议的logit变化；alpha缩后概率变化；真实选项覆盖；确定性选择是否变化；真实stop后切换延迟（自然伙伴停止没有直接作为actor阶段答案）。不增加entropy、不改prior，不将目标“选项必须切换”当任务成功。

## 5. 裁决

- **已证实：** 优化提议与原joint-KL预算不匹配；Actor全回滚；I工程资格缺格；原恢复只验证已有head/value Adam；初始mean/sample与有限reward资格有效。
- **相容：** 宏观信用粗、prior导致替代动作少、共同Adam回溯可能把selector压得更慢；均需有限真实检验。
- **未证：** 本任务没有可行动作、reward必错、E/I方法失败、B_I无价值、α=.25已通过、分层PPO一定解决。

建议仅签一次[局部优化修订](CONTACT_GOAL_EI_V2_OPTIMIZER_AMENDMENT_OR1_DRAFT.md)：先两个归档起点、固定五α回溯；有实际执行变化才给一轮有限新on-policy资格。原32k/臂不自动启动。若仍无可用步，直接交20ms低层＋真实option时长的分层替代合同，不再追加缩放网格、Teacher或新reward。
