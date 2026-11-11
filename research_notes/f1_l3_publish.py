"""Assemble small final F1 reports from completed, audited records. No simulation."""
import json,hashlib,shutil
from pathlib import Path
S=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/f1-l1-l2-20261111');R=S/'publication-l3/research_notes';D=S/'runs/l3-02'
x=json.loads((S/'reports/f1_l3_results.json').read_text());assert x['status']=='completed' and x['queries']==90 and len(x['paired'])==3
jobs=[]
for p in sorted((S/'jobs').glob('l3-*.json')):
 row=json.loads(p.read_text());assert row.get('status')!='running';jobs.append(dict(name=p.stem,**row))
measured=sum(r.get('elapsed_s',0) for r in jobs);charged=1200+measured;assert charged<=5400
cost=dict(stage='F1-L3',jobs=jobs,preparation_reporting_estimate_s=1200,measured_s=measured,total_charged_s=charged,cap_s=5400,prior_F1_s=11372.572671389,total_F1_s=11372.572671389+charged,F1_cap_s=43200,project_cap_s=864000,accounting='additive; no reset of original240h; supervisor wall includes all child compute/IO; no nested double counting')
historical=[
 dict(scope='approved dual-track ledger including carried benchmark/preparation',seconds=57800.07173384906,evidence='dual-track-formal-20261109/ledger.json'),
 dict(scope='D1/D2',seconds=3226.804518200908,evidence='research_notes/dual_track_diagnostic_artifacts/runtime-cost.json'),
 dict(scope='FB/native/offline three packages',seconds=2271.820629119873,evidence='fb-native-audit-20261110/ledger.json'),
 dict(scope='Issue58 sidecars and declared preparation/CPU',seconds=42942.23947739601+6300+83.0034,evidence='frl-v2-fb-repair-20261110/jobs/*.json; ISSUE58_EXECUTION_RESULTS; rounded CPU figures'),
 dict(scope='v2 archive diagnosis upper bound',seconds=2259,evidence='FRL_V2_ARCHIVE_DIAGNOSTICS.md:129'),
 dict(scope='v3 P1/P2 sidecars plus preparation and CPU',seconds=27151.606599092484+1800+81.63+1200+120,evidence='frl-v3-risk-20261111/jobs/*.json; FRL_V3_RESULTS.md:81'),
 dict(scope='F1 L0-L2 inclusive',seconds=11372.572671389,evidence='f1_cost_ledger.json at a72cb8a')]
cost['project_rollup']=dict(prior_stage_entries=historical,prior_charged_s=sum(z['seconds'] for z in historical),including_L3_s=sum(z['seconds'] for z in historical)+charged,precision='includes stated estimates and report-rounded CPU, not all-exact kernel time',dedup='prior benchmark already carried by dual formal; F1 L0 not counted again; supervised parent contains child wall')
assert cost['project_rollup']['including_L3_s']<=864000
(R/'f1_l3_cost_ledger.json').write_text(json.dumps(cost,indent=2).replace('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/','<project>/'))
shutil.copy2(S/'reports/f1_l3_results.json',R/'f1_l3_results.json');shutil.copy2(D/'checks.json',R/'f1_l3_checks.json')
rows=json.loads((D/'candidate-scores.json').read_text());queries=json.loads((D/'query-ledger.json').read_text())
# Aggregates only. Full states, raw actions, proposals and actual qpos remain private.
rows=[{k:v for k,v in r.items() if k not in ('proposal','prediction')} for r in rows]
shards=[]
for i in range(0,len(rows),200):
 name=f'f1_l3_candidate_scores_{i//200:02}.json';(R/name).write_text(json.dumps(rows[i:i+200],indent=2));shards.append(name)
(R/'f1_l3_candidate_scores.json').write_text(json.dumps(dict(rows=len(rows),shards=shards,raw_states_and_trajectories='private; referenced by sha256'),indent=2))
(R/'f1_l3_query_ledger.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
lines=['|窗口（两个父组）|C-P误差/力|O-F误差/力|slip峰 C-P→O-F|末yaw C-P→O-F|参考合格 C-P→O-F|','|---|---|---|---|---|---|']
phase=['|窗口/层|阶段|误差cm XYZ RMS|力N XYZ RMS|root X推进cm|腕 X推进cm|slip峰m/s|末pelvis yaw°|','|---|---|---|---|---|---|---|---|']
slip=['|窗口/层|原超限行/tick|积分后回落行|前后均超限行|超限最小法向力N|最长连续前后超限s|','|---|---|---|---|---|---|']
fmt=lambda v:'/'.join(f'{z:.3f}' for z in v)
for r in x['paired']:
 a=r['arms']['C-P'];b=r['arms']['O-F'];am=a['metrics'];bm=b['metrics'];sid=r['state_id']
 lines.append(f"|{sid}|{am['error_cm']:.3f}cm / {am['force_rms']:.3f}N|{bm['error_cm']:.3f}cm / {bm['force_rms']:.3f}N|{am['loaded_slip_peak']:.3f}→{bm['loaded_slip_peak']:.3f}|{a['window']['pelvis_yaw_final_deg']:.2f}→{b['window']['pelvis_yaw_final_deg']:.2f}|{am['low_slip_task_success']}→{bm['low_slip_task_success']}|")
 for tier,d in r['arms'].items():
  for k,v in d['phases'].items():phase.append(f"|{sid}/{tier}|{k}|{fmt(v['error_xyz_rms_cm']) if v['error_xyz_rms_cm'] is not None else '释放不计'}|{fmt(v['force_xyz_rms_N'])}|{v['root_delta_cm'][0]:.3f}|{v['wrist_delta_xyz_cm'][0]:.3f}|{v['loaded_slip_peak_mps']:.3f}|{v['pelvis_yaw_final_deg']:.2f}|")
  v=d['slip'];slip.append(f"|{sid}/{tier}|{v['old_loaded_overlimit_contact_rows']}/{v['old_loaded_overlimit_ticks']}|{v['loaded_landing_transient_rows']}|{v['persistent_pre_post_rows']}|{v['event_normal_N_min']}|{max(v['max_consecutive_persistent_s'].values(),default=0):.3f}|")
report='''# F1 L3 最终匹配预测权限实验

状态：完成并停止；无训练、封存读取或下一搜索。授权为Issue60追加评论6801908284，替代此前两尾部状态建议。

## 固定比较

stop/continue为同一父组，加一个tail-hold父组，共三个3s窗口、两个独立父组。C-P旧档案复用先验证完整初态、当前input/last-z、八候选描述、首次查询八条完整预测轨迹/评分、全部实际决策输入和三秒physics/control逐值一致。不得当作三个独立样本。

O-F与C-P均8候选、2s预测、100ms首动作/重规划、同一v3评分和原质量参考。只改变伙伴预测轨迹与未来active/release来源；heading成本distance标度仍按当前anchor和2s当前速度推算。O-F不是可部署或因果控制；其共同前缀允许提前产生不同动作，这正是未来信息干预。候选生成仍只读当前真实状态。

'''+ '\n'.join(lines)+'''

## 权限与真实执行

90次O-F选中raw均与实际执行精确一致；首BFM子步bfm_obs/z/action/q_PD及首100ms物理q/qdot/force/anchor/support/feet全部逐值对齐（见机器表）。预测分支与真实执行独立标记，未作为PPO或模仿数据。
C-P原未来spec不变性与stop/continue100tick/5command共同前缀验证保留。本轮全实际轨迹精确重放，未改变其权限结论。C-S只有固定BC500，C-S planner和控制相关饱和别名仍缺格。

## 分段真实物理

释放后原anchor误差不再计任务；三秒窗口不是完整回合。表中yaw为有符号pelvis末角，不以转腰代替脚步转向。完整q/qdot、逐点force、cap、torque、支撑和足路径在JSON。

'''+ '\n'.join(phase)+'''

## 滑移三分（原事件全保留）

原加载>1e-6N、速度>.45m/s保持；前/后速度沿同一约束Jacobian计算。持续支持运动还须看接触点/足路径，不因前后超限只持续单tick就宣称无滑移。q_PD目标峰值与真实q的5ms变化分别报告。

'''+ '\n'.join(slip)+f'''

## 失败、身份与成本

首运行1773f8f失败115.830369s：为缩短重建跳过前序reset，导致base奖励统计/episode计数与部分数组不等。没有生成科学比较结果。修复61b4371恢复原重建顺序，而非屏蔽字段；原L0 NULL规则与完整状态断言未变。旧错误保留，详见F1_ERRATUM。
实际L3 runtime `{x['source_runtime']}`；BFM/BC500及原L2模型身份不变。逐query/状态/执行文件SHA保留，发布commit不代替运行commit。

实测job {measured:.3f}s＋明确准备/报告估计1200s＝{charged:.3f}s（{charged/3600:.4f}h）≤5400s。F1原11372.572671389s继续累计，总{(11372.572671389+charged)/3600:.4f}h≤12h，计入原240h；不借剩余L3。实际90 oracle查询、24首次C-P复验候选，全部失败/IO/验证计账。完整成本与数据分层见[f1_l3_cost_ledger](f1_l3_cost_ledger.json)、[f1_l3_results](f1_l3_results.json)、[query ledger](f1_l3_query_ledger.jsonl)。

任何改善只证明当前有限搜索对预测权限敏感；任何失败都不证明所有合法z不可达。BC500不变。F1收官，后继仅S0接口审查获授权；S1需用户确认，PPO/FB/Student不重启。
'''
(R/'F1_L3_RESULTS.md').write_text(report)
print(json.dumps(dict(charged=charged,rows=len(rows),table=lines),ensure_ascii=False))
