"""Final narrative from completed evidence, not an experiment controller."""
import argparse,json,statistics,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--publication',type=Path,required=True);a=p.parse_args();R=a.publication/'research_notes';read=lambda n:json.loads((R/n).read_text())
s=read('f1_results_summary.json');c=read('f1_cost_ledger.json');pairs=s['L2']['pairs'];cp=sum(x['CP']['good'] for x in pairs);cs=sum(x['CS']['good'] for x in pairs);parents={x['parent'] for x in pairs};allgood={p:all(x['CP']['good'] for x in pairs if x['parent']==p) for p in parents};p90=sorted(s['L2']['C_P_query_wall_s'])[math.ceil(.9*len(s['L2']['C_P_query_wall_s']))-1]
estimate=120*p90+2*132+900
decision=f'''# F1中立决策表：停于L2，L3待明确确认

|问题|已证实|与证据相容|未证/不能推出|
|---|---|---|---|
|D1 冻结能力|L1 11/24状态、7/12父组在有限池存在3s参考合格控制；L2 C-P {cp}/8窗口合格，{sum(allgood.values())}/{len(parents)}父组全部已测窗口合格|冻结能力在一部分真实状态可被因果反馈调用|不是一个已训练共享checkpoint，不是完整回合/泛化成功；尾部未发现合格解不等于所有z不可达|
|D2 定价/学习|在3/11个有合格候选状态，锁定v3评分仍选中未同时合格者；旧/v3同候选排序有明确取舍|当前加权成本与硬参考不完全一致，可能妨碍学习获得共同质量|没有新增Gaussian采样/GAE/更新干预，不能给没采到/没记住/没保留因果比例；不证明唯一reward bug|
|D3 信息权限|C-S固定BC500；C-P当前位置/速度/active/yaw常速预测；未来spec干预与真实共同前缀验证|更好的当前特权优化可在部分状态减轻负担|C-S planner缺格；没有构造控制相关饱和别名；L1/L2池与时域不同，不能证明未来oracle必要|
|D4 学习位置|尾部存在误差/力改善却slip/yaw不合格；参考合格控制仍可能有大q_PD峰值|接触状态下的连续控制组织、候选覆盖、预测及低层质量均值得区分|没有修改BFM的匹配反事实，不证明必须解冻，也不据“尚未证伪”自动续大规模PPO|

保持BC500为冻结基线。小请求C-P仅因腕进度参考差约2.36mm未过，同时误差/力、root/足部运动有所减小；这个边界与尾部明显质量代价分开，不改变原判据。所有真正硬物理边界、绝对参考和旧相对风险坐标分别报告。

## 若另获批准，L3只验证一个新增假设

**假设：尾部剩余失配有可分辨的一部分来自常速度伙伴预测，而不是仅候选池/执行能力。**

在已暴露的`new_tail-sustained-5c67db5a4802`与`new_tail-hold-97ff10c4dcd9`两个父组状态，匹配同一8候选、2s预测、100ms执行、3s窗口、同一选择公式与质量参考，对照C-P常速预测与明确标记O-F的真实未来伙伴程序。候选、控制节拍、成本参数、包括heading的因果metadata口径保持相同；只改变预测伙伴轨迹权限。优先复用完全同合同C-P档案，否则成对重跑，不能只多给O-F候选或时间。

最多120次查询（两状态×两权限×30；能合法复用C-P则只新增60），不训练、不新封存、不扩候选、不测alias。按本轮p90={p90:.2f}s/query、两次132s冷启动及900s恢复/IO/报告余量，保守投影约{estimate/3600:.2f} job小时，建议独立硬顶1.5h且绝不超过L3原3h。

若O-F在同预算下仍不能同时改善任务/质量，不再把这两个状态的主要差距简单归因常速预测；若明显改善，也只支持当前预测/未来依赖，不证明可部署Student应模仿oracle。它是可判别的有限诊断，不是下一轮正式训练许可。

**此处仅提出建议，代码与队列没有启动L3。须Yi明确确认；3h额度保持完整。** 不确认也可保留本次L0–L2结论，不自动续命。
'''
(R/'F1_DECISION_TABLE.md').write_text(decision)
results=f'''# F1 L0–L2结果：有限可执行性成立，尾部任务/质量仍取舍

2026-11-12。已完成授权L1/L2并停止，L3未执行。完整正式确认点：[F1_L2_GATE_REPORT](F1_L2_GATE_REPORT.md)。

- **已证实：** 12父组24状态、384条连续3s分支中，11状态/7父组存在任务/推进/slip/yaw参考合格候选；v3评分实际选中8/24。不是新训练或统一网络胜率。
- **已证实：** 六父组八个L2窗口，因果特权重规划参考合格{cp}/8，固定BC500为{cs}/8。P1-13误差/力7.373cm/5.594N→6.954cm/5.081N，slip .318→.312m/s，但yaw17.82°→19.56°；保留收益和代价。
- **已证实：** 尾部可出现大幅减力却不合格：hold-97ff接触窗17.046cm/9.210N→3.183cm/2.028N，窗口slip .179→.762m/s、末端abs yaw9.34°→28.37°。释放后旧anchor不计跟随，阶段表单列。
- **边界：** near-5f62误差/力略好且root/足部运动较少，但腕进度低于原参考约2.36mm，保留未过判定；不把边界小差异称严重失效。q_PD目标跳变与真实q分开，窗口合格不等于动作质量完美。

奖励、滑移三分与完整分段坐标见[F1_PRICING_AND_SLIP_RESULTS](F1_PRICING_AND_SLIP_RESULTS.md)。同状态候选排序与硬质量参考存在取舍，不等于实现bug；本次没有新增学习更新，所以探索/信用/保留的因果比例仍未识别。

## 信息与恢复证据

C-S只用原BC500受限ideal-force输入，C-P不读真实未来；C-S planner仍缺格，别名未构造。L1为O-F 16候选/3s，L2为C-P 8候选/2s，不是严格信息单因素对照。未来依赖仍未证，见[实际方法](F1_L1_L2_METHODS.md)和[逐query审计](f1_info_access_audit.jsonl)。

L0原manifest不变。两次L1恢复失败及allocated efc_JT保守恢复、三秒干预、辅助滑移统计更正均单列[erratum](F1_ERRATUM.md)，未扩大mask、放宽容差、修改科学判据。原始失败和已资格分支分开保存，合并384条不重复。

## 身份、资产与成本

模型全部冻结，主L1补全runtime `f42fb177fe3679d0a162ace03b22acb5196279c4`；L2 runtime `ad93901e9b75791ae918dd37664b7d6f8b8e86bb`；早期有效分支另存f03/e31身份，发布HEAD不是执行版本。模型与原生来源SHA见[f1_model_identities](f1_model_identities.json)。

含L0、所有失败、CPU复算、IO及明确准备估计累计{c['total_charged_h']:.4f} job小时；L3支出0，训练决策/优化更新0。详见[f1_cost_ledger](f1_cost_ledger.json)。源/预测/实际执行分开，原始完整状态和大轨迹仅在个人stage保存；公开仅源码、汇总和哈希。

实际轨迹视频索引见[F1_VIDEO_INDEX](F1_VIDEO_INDEX.md)，为完整3s诊断窗，不是完整回合或预测动画。下一单一可判别假设及成本见[F1_DECISION_TABLE](F1_DECISION_TABLE.md)。BC500仍为基线，不反选旧模型、不读封存、不自动PPO/Student/FB/L3。
'''
(R/'F1_RESULTS.md').write_text(results)
current=f'''# CURRENT_STATE — Issue60 F1停于L2正式报告点

最后核对2026-11-12。[L2 gate报告](F1_L2_GATE_REPORT.md) · [汇总](F1_RESULTS.md) · [方法/权限](F1_L1_L2_METHODS.md) · [成本](f1_cost_ledger.json) · [Issue60](https://github.com/GaoE05/robot-research-bridge/issues/60) · [Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。

## 当前执行与授权

用户授权的L1/L2已完成并停止；没有L3或训练自动队列。L3仍需Yi明确确认，其3h未使用。原BFM/PD/BC500及所有历史模型不改，未读取任何封存集。GPU占用/资源必须下次实时核对，不能沿用本页快照。

## F1实际证据

L0：当前MuJoCo3.3.3 Newton dense、island禁用下25个NULL数组经156条3s C指针干预通过；27状态有效恢复/测速完成。模型与状态manifest锁不变，规则仅适用于已验证配置。[逐数组审计](F1_L0_ARRAY_AUDIT.md)和[完整L0](F1_PREFLIGHT.md)保留。

L1：12已暴露父组24状态，384连续3s候选；11状态/7父组存在任务/推进/slip/yaw参考合格候选，v3实际选合格8/24。有限存在性，不是共享checkpoint或泛化成功。L2：六父组七初态＋一个相关continue分支，八窗口；C-P参考合格{cp}/8，固定BC500{cs}/8。部分因果反馈降低误差/力，但尾部仍明显转移到滑移/偏航。near的2.36mm进度参考差与尾部严重代价分别解释，未改判据。

L1 O-F与L2 C-P候选数/时域不同，未来oracle依赖未证。C-S只有原ideal-force BC500对照，受限planner缺格，饱和别名未构造。没有训练，没有新的探索→GAE→更新因果比例。[定价/滑移/分段表](F1_PRICING_AND_SLIP_RESULTS.md)与[中立决策](F1_DECISION_TABLE.md)给出证据等级。

## 失败、身份与资产

两次L1完整恢复检查失败原判保留；对allocated efc_JT显式全量恢复并做3s前后物理干预，未新增mask/容差。早期辅助滑移时长重复计点错误已从原接触行重算，不影响原物理/reward/选择。[F1_ERRATUM](F1_ERRATUM.md)。

BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；BC500原SHA `a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e`。F1读取BC500零支路扩展及B64/B128/v3-32/128，全部独立冻结。精确路径/模型SHA见[f1_model_identities](f1_model_identities.json)。L1主要runtime f42fb17，L2 ad93901；发布commit不代替实际runtime。

服务器项目相对stage `experiments/f1-l1-l2-20261111`；L1原始l1-01/02/03及合并索引，L2 l2-01，所有失败/预测/真实执行/状态快照保留。含L0与准备估计总{c['total_charged_h']:.4f}job小时，L3=0，仍计入原240h而非重置。视频见[F1_VIDEO_INDEX](F1_VIDEO_INDEX.md)。

## 不可丢失的旧结论

F-RL-v2接触前保持机制有效，但A/B均未晋级；v3部分风险坐标改善同时有尾部和旧停止退化，BC500仍是基线。正式科学候选v3-32与末轮128身份不混淆。[v2](ISSUE58_EXECUTION_RESULTS.md) · [v3](FRL_V3_RESULTS.md)。有限直行Teacher→ideal-force Student历史成果保留，不宣称已具备G1真实腕力感知。

FB eager更新一致性、portable状态覆盖、compiled资格、踝交接/运动质量是不同层级。L0/F1仅补portable有效域/恢复证据，没有新FB update或长跑，不追溯改判旧失败。P0-B质量负例保留，不自动重启1M/5M。旧erratum、5403b7未晋级、T30/S0/M38等历史结果保留。

## 下一会话接力

先读本页、OVERVIEW、F1_L2_GATE_REPORT及Issue60/Master。当前唯一新增建议是L3的匹配预测权限诊断，尚未获准；不恢复旧PPO或自行占用3h。下一步先核真实进程、磁盘≥100GiB、runtime/模型/状态身份与剩余额度。只改个人research，main/共享资产/其他任务不动，无真机。Library旧PROJECT_STATE/SESSION_HANDOFF仍为历史快照，不用其覆盖实时GitHub状态。
'''
(R/'CURRENT_STATE.md').write_text(current)
f=R/'RESEARCH_OVERVIEW.md';text=f.read_text();text=text.replace('L1已完成，L2因果重规划进行中，结果未预判。','L1/L2已完成并停于正式gate：有限状态存在合格连续控制，因果重规划也有部分任务收益，但尾部仍有明显滑移/偏航代价；不能把有限搜索失败当全空间不可达。');text=text.replace('信息合同见F1_L1_L2_METHODS；L3仍必须另行用户确认。','信息合同见F1_L1_L2_METHODS，实际结果见F1_L2_GATE_REPORT；候选/时域不同使未来依赖仍未识别。L3仍必须另行用户确认，不自动扩大训练。');f.write_text(text)
