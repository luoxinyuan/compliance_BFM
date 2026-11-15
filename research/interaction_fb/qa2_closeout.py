"""QA2 closeout: append costs and public evidence, never modify old verdicts."""
import collections,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/r2-qa2-value-20261115';R=S/'runtime-value';N=R/'research_notes'
sys.path[:0]=[str(R),str(N)]
from research.interaction_fb.supervision import tree_bytes

def write(p,x):
    tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)

def main():
    p0=json.loads((S/'prep/p0-p1/p0.json').read_text());p1=json.loads((S/'prep/p0-p1/p1.json').read_text());p2=json.loads((S/'prep/p2-adjudication.json').read_text());physical=json.loads((S/'runs/P2-prospective-08/p2-result.json').read_text());counter=json.loads((S/'runs/P2-prospective-08/physics-counter.json').read_text());reward=json.loads((S/'prep/reward-reaudit.json').read_text());fixtures=json.loads((S/'prep/target-fixtures.json').read_text())
    assert counter['actual']==counter['reserved']==15301 and physical['status']=='P2_COMPLETE'
    assert all(v['G1']=='FAIL' and v['joint']!='PASS' for v in p2['models'].values())
    assert hashlib.sha256((N/'R2_EI_G17_QA2_VALUE_TARGET_CONTRACT_V1.md').read_bytes()).hexdigest()=='df94c3e44dd537d855a3cbefeb158180865d702e76d3bd6bf7f2bd2e642e927a'
    ledger=json.loads((S/'ledger.json').read_text());oldpath=P/'experiments/r2-v11-qentry-20261114/ledger.json';old=json.loads(oldpath.read_text())
    additions=[dict(name='QA2-readonly-reward-reaudit-cold-and-IO',phase='P0',wall_s=60,cost_type='conservative_upper_bound',payload_wall_s=reward['wall_s'],physics_ticks=0,optimizer_updates=0),dict(name='QA2-final-independent-audit-report-publication',phase='reserve',wall_s=300,cost_type='conservative_upper_bound',includes='final grouped result audit, final filesystem scan, report and public JSON generation, commit/push/Issue65/Master publication',physics_ticks=0,optimizer_updates=0)]
    for ent in additions:
        assert not any(j['name']==ent['name'] for j in ledger['jobs']);ledger['jobs'].append(ent);ledger['charged_upper_s']+=ent['wall_s'];old['job_entries'].append(dict(ent,category='Issue65-QA2'));old['total_charged_upper_s']+=ent['wall_s']
    phases=collections.defaultdict(float)
    for j in ledger['jobs']:phases[j['phase']]+=j['wall_s']
    assert phases['P0']<=900 and phases['P1']<=900 and phases['V1']<=1500 and phases['V2']<=1700 and phases['P2']<=1800 and phases['reserve']<=1200
    assert ledger['charged_upper_s']<=9000 and ledger['repair_used_s']<=800 and old['total_charged_upper_s']<=57600
    totals=[]
    for name in p2['models']:
        q=json.loads((S/'runs'/name/'result.json').read_text());totals.append(dict(name=name,groups=q['groups'],fit=q['metrics'],sha256=q['model']['sha256'],formal=p2['models'][name]))
    assert sum(t['groups'] for t in totals)==5120
    # Read prior residuals as retrospective evidence, never as new finite-Q labels.
    prior=json.loads((P/'experiments/r2-g17-qa1-20261115/runs/fixed-replay-02/node-56.json').read_text());prior_ix={(s['round'],s['candidate']):s for s in prior['scores']}
    sat=[]
    for b in p0['archived_pi_mean_branches']:
        k=(b['round'],b['candidate']);row=prior_ix[k]
        sat.append(dict(b,Q56_one_step_TD_residual=row['TD_residual'],Q56_pi_mean_1s_nstep_residual=row['nstep_residual'],retrospective_only=True))
    sat_groups={k:len({b['parent'] for b in sat if b['stratum']==k}) for k in ('inactive','low','mixed','high')}
    qualities=[]
    for state in physical['states']:
        qualities.append(dict(parent=state['parent'],family=state['family'],phase=state['phase'],time=state['time'],branches=[{k:b[k] for k in ('candidate','crn','return_1s','quality','force_rms','error_rms','complete_50_steps','prefix_restore_exact','failure')} for b in state['branches']]))
    public=dict(status='QA2_COMPLETE_G1_FAIL_NO_ACTOR',approval_issue=65,obsolete_draft_issue=64,contract_sha256=ledger['contract_sha256'],original_G17='FAIL_UNCHANGED',QA1='FAIL_UNCHANGED',source_manifest_sha256='2068e10d41e6cace8dfc5bef840e444562ad395af2767b06d996fb911bcc21de',source_replay_sha256='befe8ca2574600769f9737c329dab4ac48d118197d4d1516bf187e8dd3541e36',BFM_sha256='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361',runtime_Q_training='5a95b65dfc2d9713b4e1272374e4a1bf821b6452',runtime_P2='72e5b73e02f283de3e6a4db08fcb5893ece1fd1c',runtime_adjudication='72a1661316409333a7d7e1dfb7d22b944b851b56',P0=dict(main=p0['main'],archived=sat,stratum_independent_parents=sat_groups,within_state_cross_threshold_pairs='NONE: same-state first-action variants remained in same saturation stratum; no causal saturation attribution'),P1={k:p1[k] for k in ('source_rows','independent_parents','fit_parents','heldout_parents','counts','next_observation_exact','schedule_digest','data_logical_digest')},model_results=totals,P2=p2,physical_quality=qualities,reward_reaudit={k:v for k,v in reward.items() if k!='branches'},new_process_restore=json.loads((S/'prep/new-process-value-restore.json').read_text()),MC_n50_exact=fixtures['MC_vs_n50_model_target_Adam_exact'],new_head_Adam_groups=5120,new_training_online_rows=0,prospective_evaluation_native_rows=2000,actor_updates=0,U_updates=0,I_updates=0,Dev_Sealed=0,A0='NOT_AUTHORIZED_G1_FAIL_AND_NO_PI_MEAN_QUALIFICATION',QA2_job_s=ledger['charged_upper_s'],R2_job_s=old['total_charged_upper_s'],phase_job_s=dict(phases),repair_job_s=ledger['repair_used_s'],physics_ticks=counter,branch_ticks=8000,source_reset_prefix_ticks=7301,job_summary=[{k:j[k] for k in ('name','phase','runtime','wall_s','exit_code','cost_type') if k in j} for j in ledger['jobs']])
    write(N/'r2_ei_qa2_value_results.json',public)
    lines=['# QA2_VALUE_DECISION — 同策略有限价值可拟合，动作方向尚不可信','',
        '2026-11-15。Yi已签Issue65全文含§9；Issue64作废历史不混用。P0/P1/V1/V2及条件P2全部完成；**五个头的新父组G1均FAIL，G2仅1个低成本独立父组而UNDERPOWERED，A0不触发，阶段完成即停。**没有新Actor/U/I、Dev/Sealed或32k行为训练。旧G17、QA1及全部erratum原判保留。',
        '', '## 1. 先把成本量级与动作排序分开', '',
        '|固定终点|训练高成本1s Macro-NMAE|新P2高成本Macro-NMAE|P2确定对正确/34|含5个不确定对的准确率界|G1|G2|', '|---|---:|---:|---:|---:|---|---|']
    for t in totals:
        f=t['formal'];lo,hi=f['conservative_accuracy_interval'];lines.append(f"|{t['name']}|{t['fit']['50']['fit']['high_macro_nmae']:.2%}|{f['calibration']['high_macro_nmae']:.2%}|{f['certain_correct']}/34|{lo:.2%}–{hi:.2%}|{f['G1']}|{f['G2']}|")
    lines+=['', '每臂fresh同一541,441参数H条件head、Adam1e-4、1024组×64；抽样父组/phase/remaining-H完全同序列，H固定1s上限。不是旧无限时域双Q的直接算法胜负。MC和n50的model/target/Adam逐值相同，属于数学等价重复检查，不算两次独立成功。',
        '', 'n5在P2的高成本数值bounds通过：Macro-NMAE21.04%，对更好常数/phase基线改善57.70%，但动作确定排序仅4/34。G2因低成本独立父组不足仍不通过。这个对照说明“成本量级拟合”不足以保证可用动作方向，不能用更低MSE给Actor发车。n25近距确定对9/9、中距1/10、停止0/9、zero4/6；不能按任务拼价值头。',
        '', '## 2. P0：实际力、reward与目标身份', '',
        f"主轨14,000 tick中，actual-active {p0['main']['active_ticks']}，饱和{p0['main']['saturated_ticks']}（{p0['main']['saturation_fraction']:.3%}）；逐点raw/applied与K80/D8/10N公式最大差均0。旧40条真实分支2,000个20ms reward按原dtype量化后逐值一致，已存成本分项重算最大差0。首控制的continuity保留实际归档项，没有伪造前序q_PD；后49控制逐项重算。",'',
        '|旧暴露mean状态|饱和层/active tick比例|真实1s G（π_mean）|Q56（旧∞Q）|原始力峰N|', '|---|---|---:|---:|---:|']
    for b in sat:
        if b['candidate']=='mean':lines.append(f"|round{b['round']} {b['family']}|{b['stratum']}/{b['saturation_fraction'] if b['saturation_fraction'] is not None else 'NA'}|{b['G1s_pi_mean']:.6f}|{b['Q56_infinite']:.6f}|{b['raw_norm_max']:.3f}|")
    lines+=['', f"旧分支独立父组按层计：{sat_groups}。同状态五动作没有实际跨层，故不能据此证明饱和因果效应；mixed层无独立组，记UNDERPOWERED。mid入口未饱和仍错排、mid与stop均高饱和却排序不同；力方向、raw/tracking/relative-velocity仍有信号，10N饱和不是已证实唯一原因。原TD/1s nstep residual逐状态/动作保留在机器摘要。旧∞Q与π_mean回报不是新有限Q_mu的正式校准标签。",'',
        'Q输入为当前特权、本体/真实动作历史、previous/target z与slot；不含未来伙伴spec或全局clock。原1968个mean控制target action/z逐值一致的QA1证据只读继承；3500行为是随机μ，旧40分支是首动作干预后mean。两者标签身份严格分开。严格相同过去、不同未来导致相反最优动作：UNMEASURED，不宣称已经证明POMDP根因。',
        '', '## 3. P1：合法连续后继和独立性', '',
        '|H步数/时长|fit完整窗/4父组|heldout完整窗/2父组|删失fit/heldout|','|---|---:|---:|---:|']
    for h in (5,25,50):
        ff=next(x for x in p1['counts'] if x['H']==h and x['split']=='fit');hh=next(x for x in p1['counts'] if x['H']==h and x['split']=='heldout');lines.append(f"|{h}/{h*.02:.1f}s|{ff['windows']}|{hh['windows']}|{ff['censored']}/{hh['censored']}|")
    lines+=['', '3500行、6独立父组。fit=none/首near/mid/stop-continue四组；heldout=zero/partial-near两组。stop/continue同父组但不同episode不可拼接，actual next observation逐字段相同；所有实际terminated为false，time-limit短窗删失，release不是终止。旧heldout没有高成本父组：不能在这里宣称完整G2通过。完整h1..50仅支撑V2剩余H查询，主H报告5/25/50，未扩预测时域。',
        '', '## 4. P2：四新train父组、同状态/同μ/两CRN', '',
        '模型、父组、时间与seed在拟合前锁定，预测在每组真实候选执行前存盘。40条1s分支全部完成，恢复effective MuJoCo/solver、Task/本体/力/z/动作历史及前缀RNG逐值一致；首实际action/z/q_PD与预测合同一致。首20ms为明确干预，随后49步由冻结随机μ实际采样。所有新结果仅评价，不回流训练。',
        '', '|新状态|两CRN×五动作平均G|五动作真实平均G极差|最大同动作CRN差|MC预测均值|n5预测均值|n25预测均值|','|---|---:|---:|---:|---:|---:|---:|']
    for state in physical['states']:
        rr=[x for x in p2['records'] if x['parent']==state['parent']];avg=lambda key:sum(x[key] for x in rr)/5;pred=lambda key:sum(x['predictions'][key] for x in rr)/5
        lines.append(f"|{state['family']} {state['phase']}|{avg('G_mu_mean'):.6f}|{max(x['G_mu_mean'] for x in rr)-min(x['G_mu_mean'] for x in rr):.6f}|{max(x['CRN_return_difference'] for x in rr):.6f}|{pred('V1-MC'):.6f}|{pred('V2-n5'):.6f}|{pred('V2-n25'):.6f}|")
    lines+=['', '总40个无序动作对：39个均值差≥1e-4，其中5对两个CRN的收益差符号反转，全部保留UNCERTAIN；另1对低于门槛。四父组都贡献确定可比对。表中准确率界以全部39对作分母：确定正确/39至(确定正确+5)/39，不删除不确定对凑高准确率；即使最乐观上界，所有模型仍低于70%。这是真实G1 FAIL，而非仅因样本不足暂不能判。',
        '', '三个active组都属于|G|≥1的高成本；near入口并未实现预想的低成本，zero只有一个低成本独立父组。因此G2标UNDERPOWERED，未换near时刻、补seed或替换父组。n5数值上满足已有三高/一低bounds不等于完整G2通过。',
        '', '## 5. 正确性、预算及失败', '',
        '两次纯工程故障及各一次修复见[Engineering Errata](R2_EI_QA2_ENGINEERING_ERRATA.md)：缺既有pointer依赖、NumPy标量JSON封装。没有数学/数据/门修订；原包和失败保留。五新模型/targets/Adam独立新进程逐值读回；source完整物理及replay以不可变引用保存。G2通过/拒绝与常数基线fixture在正式P2裁决前通过；V1=n50完整终点一致。',
        '', f"实际QA2累计{ledger['charged_upper_s']:.6f}/9000 job秒；工程池{ledger['repair_used_s']:.6f}/800秒（包含已记P1失败，不在总账重复加算）；R2累计{old['total_charged_upper_s']:.6f}/57600秒。分阶段秒数{dict(phases)}。P2作业636.948487秒；actual/reserved={counter['actual']}/{counter['reserved']}，包含8000分支tick＋7301冷启动/source/reset/prefix tick，reserved不与actual相加。新增head Adam组5120，真实训练行新增0，物理评价行2000，Actor/U/I/Dev/Sealed全0。",'',
        '冻结原BFM SHA `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；source manifest `2068e10d41e6cace8dfc5bef840e444562ad395af2767b06d996fb911bcc21de`。head训练runtime5a95b65、P2 runtime72e5b73；每头完整SHA、原始分组、全部对与物理质量在[机器摘要](r2_ei_qa2_value_results.json)。',
        '', '## 6. 证据分级和单一下一干预建议', '',
        '**已证实：**已核范围内真实力/reward/连续后继正确，随机μ与mean标签已分开；固定预算多步/MC能改善训练高成本拟合；新父组动作方向仍严重不可信，故不更新Actor。n5虽有较合理成本量级，却不能给出可用动作改善方向。',
        '', '**相容：**总回报拟合误差大于同状态首动作带来的细小收益；仅四fit父组、每状态仅一个实际采样动作的当前回归，可能拟合了状态成本与偶然相关性，没有学稳因果动作增益。一步自举传播在此预算下更慢，但它不是当前已证实唯一根因。',
        '', '**未证：**隐藏未来信息是否导致相反最优动作、是否本质不可观测、是否增加网络容量/更新就可解决、可信Q能否带来Actor收益，以及I的交互表示价值，均未在本轮得到证明。',
        '', '下一项最可判别干预建议：单独设计有限、分父组的同状态动作差分ΔG监督/校准，直接核验改善方向，保持本物理reward与明确后续策略；继续用独立父组评价。它针对“知道状态贵不贵，却不知道动作往哪改”的现有证据，不再仅追加同轨迹MC/TD更新。新P2仍是评价集，不得转训练或自动启动该建议。',
        '', '**A0不执行：**所有头G1 FAIL，并且本轮没有Q_pi_mean的同策略价值资格；μ价值不能自动指导确定性mean Actor。QA2至此STOP，原QA1/G17 FAIL、BC500与原资产不改。']
    lines+=['','## 附：前瞻物理窗口，不是新策略成功','','|状态族|1s实际误差RMS均值cm|实际力RMS均值N|loaded-slip峰最大m/s|yaw excursion最大°|硬关节/失支撑|','|---|---:|---:|---:|---:|---|']
    for state in physical['states']:
        bb=state['branches'];lines.append(f"|{state['family']}|{100*sum(b['error_rms'] or 0 for b in bb)/10:.3f}|{sum(b['force_rms'] for b in bb)/10:.3f}|{max(b['quality']['slip_peak'] for b in bb):.6f}|{max(b['quality']['yaw_excursion_deg'] for b in bb):.3f}|{max(b['quality']['max_joint_excess_rad'] for b in bb):.3f}rad/{sum(b['quality']['unsupported_ticks'] for b in bb)}tick|")
    lines+=['','上述只有1s冻结μ动作评价；mid仍高负担，不冒充合格反馈Expert或完整释放恢复。完整任务与新行为收益未训练/未验收，原机械门未放宽。']
    (N/'QA2_VALUE_DECISION.md').write_text('\n'.join(lines)+'\n')
    head=f'''# 最新：Issue65 QA2完成；新父组动作排序全部FAIL

2026-11-15。Yi签署Issue65全文含§9；Issue64作废历史。[QA2最终报告](QA2_VALUE_DECISION.md) · [逐字合同](R2_EI_G17_QA2_VALUE_TARGET_CONTRACT_V1.md) · [机器摘要](r2_ei_qa2_value_results.json)。P0/P1/V1/V2/P2均已完成；五头各1024组、40条新train-only随机μ分支、四独立父组。训练高成本拟合n1 62.09%→n25 8.53% Macro-NMAE；但新排序最好的n25确定对14/34，含5不确定对上界48.72%仍低于70%。五头G1 FAIL；G2因三高/一低独立父组UNDERPOWERED。没有补组、看封存、重选或补训。

A0不触发：G1失败，且无Q_pi_mean匹配资格。Actor/U/I/Dev/Sealed均0，原QA1/G17 FAIL保持。QA2累计{ledger['charged_upper_s']:.6f}/9000job秒，15301 actual/reserved全流程tick；R2累计{old['total_charged_upper_s']:.6f}/57600秒。GPU0作业已退出；阶段完成即停，不自动新训。两次工程故障及一次修复/家族原记录保留。以下为历史时点。

---

'''
    cs=N/'CURRENT_STATE.md';cs.write_text(head+cs.read_text())
    overview=N/'RESEARCH_OVERVIEW.md';text=overview.read_text();note='''## 最新：Issue65 QA2有限价值与动作方向分离

同策略μ、显式有限H的MC/多步监督已完成；固定数据训练高成本拟合改善，但四新train父组40物理分支上动作排序全部FAIL。成本量级可学不等于可用于Actor的梯度，n5较合理量级仍仅4/34确定对正确；n25近距有局部排序正证据，不能拼头替代统一资格。G2三高/一低父组不足，未补数据找补。Actor/U/I/Dev/Sealed仍停，原QA1/G17结论保留。报告见[QA2_VALUE_DECISION](QA2_VALUE_DECISION.md)。下一项建议为独立同状态ΔG方向校准研究，未自动启动。

''';overview.write_text(text.replace('## 科学目标',note+'## 科学目标',1))
    output=tree_bytes([S]);alloutput=tree_bytes(old['output_roots']);assert output<=3*1024**3 and alloutput<=12*1024**3
    ledger.update(status='QA2_COMPLETE_G1_FAIL_A0_NOT_AUTHORIZED',physics_actual=15301,physics_reserved=15301,output_bytes=output,R2_output_bytes=alloutput,new_head_Adam_groups=5120,new_actor_updates=0,sealed_executed=False,pending_jobs=[],phase_charged_s=dict(phases))
    old.update(status='ISSUE65_QA2_COMPLETE_G1_FAIL',g17_qa2=dict(new_head_Adam_groups=5120,actor_updates=0,U_updates=0,I_updates=0,new_main_online_rows=0,P2_branches=40,P2_physics_ticks=15301,A0='NOT_AUTHORIZED',original_G17='FAIL_UNCHANGED',QA1='FAIL_UNCHANGED'))
    write(S/'ledger.json',ledger);write(oldpath,old)
    public.update(QA2_output_bytes=output,R2_output_bytes=alloutput);write(N/'r2_ei_qa2_value_results.json',public)
    with (N/'QA2_VALUE_DECISION.md').open('a') as f:f.write(f'\n最终输出实扫：QA2 {output/1024**3:.6f}/3GiB，全部R2 {alloutput/1024**3:.6f}/12GiB；源/运行目录与小型报告均计入。\n')
    print(json.dumps(dict(status=ledger['status'],QA2_job_s=ledger['charged_upper_s'],R2_job_s=old['total_charged_upper_s'],output_bytes=output,R2_output_bytes=alloutput,phase_s=dict(phases)),indent=2),flush=True)

if __name__=='__main__':main()
