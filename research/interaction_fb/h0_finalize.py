"""Read-only final H0 evidence audit; no physics, optimizer, dev or sealed."""
import json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import torch,numpy as np
from research.interaction_fb.h0_common import *
from research.interaction_fb.journal import read
from research.interaction_fb.h0_learn import read_fit
from f1_mjdata_contract import exact
from research.interaction_fb.supervision import tree_bytes
torch.set_num_threads(1)
def main():
 t=time.monotonic();N=ROOT/'research_notes';pub=json.loads((N/'h0_results.json').read_text());out=S/'runs/evaluation-06';ev=json.loads((out/'evaluation.json').read_text());rr,X,Y=read_fit();constant=Y.mean(0).numpy();units=[];same=[];source=[]
 for c in ev['results']:
  pp=c['parent'][:12];packet=read(out/(pp+'-predictions.ptz'),c['predictions']);d={};seen=set();u=[]
  for r in c['rows']:
   a=read(out/f"{pp}-{r['method']}-{r['goal']}.ptz",r['archive']);i=r['command']
   if i in d:
    assert exact(a,d[i]),'same state + same actual plan changed full control/physics'
    same.append(dict(parent=c['parent'],command=i,full_physics_control_exact=True))
   else:d[i]=a
   if i in seen:continue
   seen.add(i);actual=np.array(r['y']);row=dict(command=i)
   for method in ['E','I']:
    pred=np.array(packet['predictions'][method][i])*SCALES
    row[method]=dict(wrist_delta_RMSE_cm=float(100*np.sqrt(np.mean((pred[:3]-actual[:3])**2))),root_delta_RMSE_cm=float(100*np.sqrt(np.mean((pred[3:6]-actual[3:6])**2))),mean_force_XYZ_RMSE_N=float(np.sqrt(np.mean((pred[9:12]-actual[9:12])**2))))
   u.append(row)
  units.append(dict(parent=c['parent'],K=c['K'],rows=u));source.append(dict(parent=c['parent'],h_digest=hashlib.sha256(packet['h'].numpy().tobytes()).hexdigest()))
 assert len(set(a['h_digest'] for a in source))==4
 macro={name:float(np.mean([np.mean([r[name] for r in x['rows']]) for x in pub['prospective_prediction']])) for name in ['E_MSE','I_MSE','constant_MSE']}
 counts={x.parent.name:json.loads(x.read_text()) for x in (S/'runs').glob('*/physics-counter.json')};ticks={k:sum(v[k] for v in counts.values()) for k in ['actual','reserved']};assert ticks['actual']==ticks['reserved']==25164
 assert all(r['boundary'] is None and r['quality']['max_joint_excess_rad']==0 and r['quality']['unsupported_ticks']==0 for c in ev['results'] for r in c['rows'])
 assert len(rr)==24 and len(ev['results'])==4 and sum(len(c['rows']) for c in ev['results'])==16
 # One full live scan; retained code/worktrees count as new bytes, hardlinks once.
 old=json.loads((P/'experiments/r2-v11-qentry-20261114/ledger.json').read_text());nb=tree_bytes([S]);allb=tree_bytes(old['output_roots']);assert nb<2*2**30 and allb<12*2**30
 audit=dict(status='PASS_FINAL_READONLY_AUDIT',future_unit_errors=units,parent_macro_MSE=macro,matched_duplicate_branch_exact=same,unique_test_h=source,physics_by_job=counts,physics_ticks=ticks,new_output_bytes=nb,R2_output_bytes=allb,head_manifest=json.loads((S/'runs/learn-04/head-manifest.json').read_text()),scientific_runtime='b77ed34f06fcfcf6f68de6f0793f4f8b4b22fb22',audit_wall_s=time.monotonic()-t)
 write(S/'prep/final-audit.json',audit);pub['final_audit']=audit;write(N/'h0_results.json',pub)
 text=(N/'H0_REPRESENTATION_AND_CAUSAL_ADDRESS_RESULTS.md').read_text()
 pre='''# H0结项摘要：公开Goal可寻址；B_I增量未证

2026-11-15，Issue66批准的一次性H0已按锁完成，未启动正式E/I、PPO或Student。

- **已证实：**六fit源父组×四持续命令有真实后果差异；四独立评价父组/16真实分支，P-E 4/4、P-I 3/4满足≥2cm且质量合格的局部Goal效应。P-E的FOLLOW−HOLD为5.09/6.08/5.07/5.93cm；I在stop仅1.69cm，保留未过门。关节硬越界/失支撑均0；最大loaded-slip约0.004m/s、yaw excursion约6°，精确值见表。
- **已证实：**同进程及新进程完整物理/历史/RNG/solver恢复与真实2s执行逐值一致；同命令重复的完整控制/物理亦逐值一致。K40与原K80每分支raw/applied公式最大差0，原BFM/PD未更新。
- **B_I独立价值未证：**三组与P-E选相同命令、物理结果相同；stop选更小动作，进展不够2cm但误差/力可能有局部取舍，不能称P-E全指标支配I。不能凭B谱/低训练loss宣称I胜出。
- **相容而未唯一证明：**结果编码会改变寻址取舍；当前候选和Goal范围很窄，16D表示有效秩约3.14。未证明低有效秩就是因果失败原因，也未证明冻结底座缺能力。
- **未证：**P-E四组均为HOLD→standing、FOLLOW→arm15，未证明其选择真正依赖不同接触状态；I有一次状态相关改变但没有额外行为收益。更合适的持续行走、高层每100ms重规划、完整接触释放恢复、随机鲁棒性、generalization及正式Actor适配仍未验证。K40中距FOLLOW仍9.64cm，不能称完整中距Expert达标。HOLD的几何/力成绩不是主动跟随成功。

**决策：H0到此完成，不产生H0.1/新Q通行证。进入匹配可训练Actor的E/I合同设计；raw Goal/outcome是强基线，B_I只能凭下一轮实际增量保留。以下完整数据与协议边界。**

'''
 text=text.replace('# H0 — finite interaction outcome prediction and causal Goal addressing','## Detailed evidence: finite outcome prediction and addressing',1)
 text+= '\n## R1b physical-unit audit and restore evidence\n\n'
 text+=f"Parent-balanced selected-command standardized MSE: E={macro['E_MSE']:.7f}, I={macro['I_MSE']:.7f}, constant={macro['constant_MSE']:.7f}. Both forecast selected outcomes better than this constant on these four parents; no test-time refit or unexecuted-candidate ranking claim. Detailed wrist/root cm and mean-force N errors are in JSON. The same command duplicated across methods/Goals passed full physics/control exactness on {len(same)} comparisons.\n"
 (N/'H0_REPRESENTATION_AND_CAUSAL_ADDRESS_RESULTS.md').write_text(pre+text)
 (N/'H0_INFO_ACCESS_AND_IDENTITIES.md').write_text('''# H0 permissions and exact identities

Scientific collection, predictor fitting and16 physical Goal branches all used runtime b77ed34f06fcfcf6f68de6f0793f4f8b4b22fb22. Read-only reporting source0894fb4; final integration is recorded separately. BFM SHA33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361. Small E/I full-model/Adam/RNG/normalizer/sampler packet hash is in h0_results.json; original shared models unchanged. Sources kept private and hash-addressed.

|Path|Actual permission/evidence|
|---|---|
|clean_env.Task.prepare → m9_learning.inputs|assert keys exactly state,last_action,history_actor; frozen per-key normalizer, first465 own-state features. No privileged_state used in this path.|
|geometry_env.Task.student_features/contact_features(False)|465D proprio +256D actuallastz +100-frame measured force and known mounting slots=1639D; no anchor error/partner velocity/active task truth.|
|h0_physical.source and vectors|h is exact student_features; proposal phase is nearest original8-grid to own current actuallastz.20×256 plan is the robot's own proposal, not future partner motion.|
|h0_learn vectors/Models/select|Only restricted h, proposed command plan and publicGoal; no spec/clock/phase/final-distance fields enter either fitted predictor or T. Method/Goal choices saved before future physics.|
|h0_common.result|24D future actual robot/contact outcomes only for training/evaluation; includes stability truth, not future taskID/phase/clock. No online prompt is produced from hidden anchor or future truth.|
|Environment/Task/reward|Does read true partner/current geometry to generate real force and assess consequence; this process-level access is not controller permission. Real future script used only by environment, same across allGoal/method branches.|
|Stored F1 data|Historically exposed dev FIT records explicitly approved for reuse; actual native command branches, not old Actor output labels. Historical prefix may be visited by BC500/B128/v3, but none of those actors or hidden features is loaded online.|
|Negative controls|Raw/phase/random fit-only retrieval; goal shuffle reuses actual opposite-Goal branch; successor-zero selects tied standing. Physical successor-zero evidence only where standing branch actually exists, missing cells marked; no new physics fabricated.|

No R2 Dev/Sealed execution; publicGoal is a command, not hidden future task identity. Ideal-force permission remains a simulation sensing assumption, not hardware validation. Full source/solver/history restore proven in separate new process, model/Adam/output/RNG/sampler restored exactly; no change to old strict gates.
''')
 (N/'H0_ENGINEERING_ERRATA.md').write_text('''# H0 engineering/prelock record

No physical/learning/restore runtime failed in this stage; all committed jobs succeeded. Two prelock checks caught ordinary archival/preparation issues before any physical or optimizer run: (1) an over-specific textual occurrence-count assertion rejected the proposed K edit before writing it; corrected to three exact unique source replacements, then256-case K80 raw/applied parity and per-branch formulas passed; (2) Markdown patching introduced one extra final LF into the copied signed contract, detected by Git blob mismatch154d701…; rewrote exact fetched bytes, verified approved blobac5224640ec52a1fa30f66ccde40aeeaba805c04 before the first commit. Both are included in450s inclusive preparation; no additional/double-count repair charge. Original remote scripts/logs/source hashes retained; no scientific thresholds, reward, seed, candidate pool or past verdict changed. No repair reserve was consumed by a runtime restart.

Final audit is observational; physical-unit statistics and successor-disable missing cells do not alter the immutable method lock or trained weights. Historical QA1/QA2/D2/Envelope/Return negative results and errata remain unchanged.
''')
 print(json.dumps(dict(status=audit['status'],new_output_bytes=nb,R2_output_bytes=allb,macro=macro,audit_s=audit['audit_wall_s'])),flush=True)
if __name__=='__main__':main()
