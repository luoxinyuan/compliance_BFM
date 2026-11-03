"""Build an auditable result table from completed outputs; does not launch experiments."""
from pathlib import Path
import json,hashlib,shutil
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/t30-transfer-20261103';D=P/'maintenance/research-handoff-20261102/research_notes';A=R/'A-prepare-01/job';C=R/'C-final-02/job';artifact=D/'t30_artifacts';artifact.mkdir(exist_ok=True)
assert json.loads((A/'result.json').read_text())['status']=='completed'
rows=json.loads((R/'analysis/comparison.json').read_text());by={x['label']:x for x in rows};recs=json.loads((R/'analysis/recovery-index.json').read_text());ledger=json.loads((R/'compute-ledger.json').read_text());models=[]
for j in [A,C]:
 for f in j.glob('*-training.json'):
  x=json.loads(f.read_text());name=f.name.removesuffix('-training.json');pt=j/(name+'.pt');assert hashlib.sha256(pt.read_bytes()).hexdigest()==x['sha256'];models.append(dict(name=name,path=str(pt),**x))
selected={k:by[k] for k in by if not k.startswith('filter')};packet=dict(models=models,results=selected,recovery=recs,ledger=ledger,aggregation=json.loads((A/'aggregation-scope.json').read_text()))
(artifact/'results.json').write_text(json.dumps(packet,indent=2))
for name in ['coverage-and-action-errors.json','normal-separation.json','early_stop-separation.json','late_stop-separation.json','sealed_early-separation.json','sealed_late-separation.json']:
 if (R/'analysis'/name).exists():shutil.copyfile(R/'analysis'/name,artifact/name)
shutil.copyfile(R/'D-final-03/job/jump-decomposition.json',artifact/'jump-decomposition.json')
def table(cond):
 s='| 条件 | Teacher30 | S0 r0 | S1matched | S2 | S1warm (4k) | N30 (4k) |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for c in cond:
  cells=[]
  for lab in ['Teacher30','S0','S1matched','S2','S1warm','N30']:
   x=by.get(lab+'_'+c) or (by.get('source_'+c) if lab=='Teacher30' else None);cells.append(f"{x['error_cm']:.3f} / {x['force_rms']:.3f}" if x else '未运行')
  s+='| '+c+' | '+' | '.join(cells)+' |\n'
 return s
text='''# Teacher30 transfer / recovery / motion quality — 2026-11-03

本阶段不训练Teacher/BFM，不修改PD；真实Tether上限5N。冻结Teacher30自身反馈改善的历史结论保留。本报告区分Teacher自身执行、Student状态接管、标签拟合和Student物理迁移，不能相互替代。所有模型是仿真研究模型，G1精确腕部XYZ外力硬件未知。

## 固定实验与身份

主运行 `8af5a87cf6684469d28a99a11e156ecdb0ba9575`；独立质量对照 `23309461185f0b10cdc6dfc291c8665f6213fa80`。协议[t30_protocol.json](t30_protocol.json)、源码[t30_stage.py](t30_stage.py)。阶段[Issue35](https://github.com/GaoE05/robot-research-bridge/issues/35)。全数据目录：`'''+str(R)+'''`。

Teacher30 `c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33`；旧PPO15 `92692be758d04386c32bea3108b3b155d4c37b0c2b70679e78cf849f593112fe`；S0 `0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12`；BFM `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`。由实际文件SHA检查，不凭best文件名认定。

Student 1025维：465本体/已执行动作历史、300实际历史力、256自身合法z、4固定腕部安装编码。没有anchor、伙伴速度、任务时钟、active真值或Teacher隐藏状态；BFM observation只含state/last_action/history_actor，其history仅角速度、重力投影、关节位置/速度/动作。N30物理删除力列及对应归一化/网络输入，725维，重新监督训练；没有外力矩估计或接触真值替代。200Hz物理、50Hz原BFM/PD、10Hz高层命令及原5子步slerp不变。

Teacher30六条独立回合943标签；旧PPO15数据824标签保留正常/无接触/局部行为。每条标签同一10Hz决策时刻的真实输入与合法z，没有重复近恒等投影。固定seed805、网络、归一化、Adam3e-4、batch128。S1warm4000更新，随后分叉相同模型和新Adam：S1matched追加3000原新混合，S2追加3000并将每batch32条Teacher-own换成恢复域Student标签。两者总学习7000更新；实际共享warm计算仅记一次。N30与S1warm同4000更新/数据/seed。固定末轮，不按封存评估挑checkpoint；单训练seed不支持统计普遍性。

新封存：82cm/ramp7.8，早onset4.31/stop7.65/decel0.9/hold3.1，晚onset4.16/stop9.65/decel0.7/hold3.3，释放后6s；其余为已暴露开发/回归。伙伴轨迹真实改变施力，三轴力自然产生。

## 物理结果

单元为全接触3D RMS误差cm / 真实力RMS N；无接触的0不是主动跟随成功。完整XYZ、阶段、饱和、质量和失败见[t30_artifacts/results.json](t30_artifacts/results.json)。

'''+table(['normal','early_stop','late_stop','local','no_contact','sealed_early','sealed_late'])
text+='\n## 接管与聚合范围\n\n完整Student前缀重放到接管时刻，保留BFM历史、实际动作、自身z和真实物理状态。所有接管前qpos/qvel/force逐点比较；没有只恢复qpos或虚构Teacher历史。接管判据为剩余接触force RMS+3×error(m)较低，释放root路程不比继续Student增加超过3cm；释放后用root路程+腕速描述性组合。此操作性筛选不是通用恢复真值，必须同时看分项取舍。\n\n'
for r in recs:
 records=r['records'];text+=f"- `{Path(r['path']).name}`：{sum(x['accepted'] for x in records)}/{len(records)}端点满足预定筛选；完整前缀最大差{max((x.get('prefix_max_diff',0) for x in records),default=0)}；接受区间{r['accepted_windows']}。\n"
a=packet['aggregation'];text+=f"\n实际聚合{a['labels']}条标签，来自{len(a['unique_episodes'])}条Student回合：{a['unique_episodes']}。仅取两端接管都成立的0.4s窗口；这些区间之间或其他状态没有恢复保证。少量标签重复采样的风险完整保留，不将端点的轻微改善夸大为大范围恢复。\n"
text+='\n## 阶段指标与恢复\n\n| 模型/条件 | 阶段 | 3D误差cm | 力N | 饱和% | root位移XYZ cm |\n|---|---|---:|---:|---:|---|\n'
for label in ['Teacher30_sealed_early','S0_sealed_early','S1matched_sealed_early','S2_sealed_early','Teacher30_sealed_late','S0_sealed_late','S1matched_sealed_late','S2_sealed_late','N30_sealed_late']:
 if label not in by:continue
 for ph,x in by[label]['phases'].items():
  if ph in ['pre','recover']:continue
  text+=f"|{label}|{ph}|{x['error_cm']:.3f}|{x['force_rms']:.3f}|{100*x['cap_fraction']:.1f}|{[round(100*v,2) for v in x['root_displacement']]}|\n"
text+='\n释放后的几何anchor误差及force=0不计主动跟随收益。完整释放路程、腕速、支撑变化见逐阶段质量数据；支撑布尔翻转不等于准确步数。无接触也检查姿态/位移，不以force=0代替保持资格。\n\n## 目标与真实动作质量\n\n旧Teacher30四个保存条件的大目标跳变都发生于移动期右踝pitch，41.75–53.92°。同状态计算分解：只换z为0.44–1.54°，只换BFM观测为41.38–53.47°；重构保存q_PD误差0。它指向原BFM状态反馈输出，不证明是独立物理因果或BFM实现bug。实际前一个20ms关节变化2.42–3.75°，不能称关节瞬间旋转50°。完整分析另记录后一个20ms变化、q/dq/目标差、力矩/限幅、滑移、root/腕加速度。\n\n固定0.1s高层因果z平滑（原PD不变）的有限物理对照没有整体收益，不采用、不追调tau：\n\n|条件|原误差/力|平滑误差/力|\n|---|---:|---:|\n'
for c in ['normal','early_stop','late_stop']:
 x,y=by['source_'+c],by['filter01_'+c];text+=f"|{c}|{x['error_cm']:.3f}/{x['force_rms']:.3f}|{y['error_cm']:.3f}/{y['force_rms']:.3f}|\n"
text+='\n晚停止hold有改善但整体移动误差变大，正常条件hold及释放变差；不能只取局部改善宣布连续性修复。\n\n## 新模型身份\n\n'
for m in models:text+=f"- {m['name']}: `{m['sha256']}`，`{m['path']}`，本次fit {m['updates']}更新。\n"
text+='\n## 资源、复现与限制\n\n账本`compute-ledger.json`记录真实卡号、GPU/内存峰值、固定源码、进程、墙钟及退出。解析/汇总脚本为`t30_analyze.py`、`t30_label_audit.py`、`t30_report.py`；物理数据及全部失败不入Git，公开的是精简数值。不要重复运行已消费的stage覆盖目录。视频来自真实保存qpos，双固定世界/俯视、1×，含真实force/anchor；从ready t=3s到完整释放结束，省略的是共同无交互预初始化，不剪掉失败。\n\n本阶段只有单训练seed、少量窗口恢复证据；近似接触信息与Teacher特权信息差异、有限覆盖及闭环分布偏移尚未被独立完全分离。不能由新标签未全面获益证明无力不可行或BFM没有能力。\n\n平面下一候选仍为已有同权重M10缓弯/heading，已有standing进入证据，不等于Teacher30在当前停止偏离状态可安全切换；高度候选已有下沉/回升，属于保持连接Tether，不是已验证单向按压。此轮不新增大范围技能训练，见STOP_CAPABILITY_REVIEW。\n'
(D/'T30_TRANSFER_RESULTS.md').write_text(text)
print('report',len(text),'models',len(models),'aggregate',a['labels'])
