"""H0 immutable endpoint aggregation and publication-safe summary. No learning/physics."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import numpy as np,torch
from research.interaction_fb.h0_common import *
from research.interaction_fb.journal import read
from research.interaction_fb.h0_learn import load_models,read_fit
def main():
 out=S/'runs/evaluation-06';ev=json.loads((out/'evaluation.json').read_text());learn=json.loads((S/'runs/learn-04/learning.json').read_text());models,p=load_models();rr,X,Y=read_fit();constant=Y.mean(0).numpy();prediction=[]
 for case in ev['results']:
  pp=case['parent'][:12];a=read(out/(pp+'-predictions.ptz'),case['predictions']);seen=set();vals=[]
  for row in case['rows']:
   if row['command'] in seen or row['y'] is None:continue
   seen.add(row['command']);actual=np.array(row['y'])/SCALES
   vals.append(dict(command=row['command'],E_MSE=float(np.mean((np.array(a['predictions']['E'][row['command']])-actual)**2)),I_MSE=float(np.mean((np.array(a['predictions']['I'][row['command']])-actual)**2)),constant_MSE=float(np.mean((constant-actual)**2))))
  prediction.append(dict(parent=case['parent'],K=case['K'],unique_executed_commands=len(vals),rows=vals))
 physical=[];ablations=[]
 for c in ev['results']:
  cr=[];pp=c['parent'][:12]
  for r in c['rows']:
   a=read(out/f"{pp}-{r['method']}-{r['goal']}.ptz",r['archive']);ph=a['physics'];co=a['controls'];ac=[x for x in ph if x['active']]
   qpd=np.array([x['q_PD'] for x in co]);q=np.array([x['qpos'][7:] for x in ph]);feet=np.array([x['feet_xyz'] for x in ph]);F=np.array([x['force'] for x in ac]);err=np.array([np.array(x['anchor'])-x['wrist'] for x in ac])
   details=dict(force_xyz_rms_N=np.sqrt(np.mean(F*F,axis=0)).tolist() if ac else [0.]*3,error_xyz_rms_cm=(100*np.sqrt(np.mean(err*err,axis=0))).tolist() if ac else None,saturation_fraction=float(np.mean(np.linalg.norm(F,axis=1)>=9.999)) if ac else 0.,q_PD_internal20ms_jump_max_deg=float(np.degrees(np.max(abs(np.diff(qpd,axis=0))))),actual_q_internal5ms_jump_max_deg=float(np.degrees(np.max(abs(np.diff(q,axis=0))))),actual_joint_dq_abs_max_rad_s=float(np.max(abs(np.array([x['qvel'][6:] for x in ph])))),foot_paths_m=np.linalg.norm(np.diff(feet,axis=0),axis=2).sum(0).tolist(),tilt_peak_deg=max(x['tilt_deg'] for x in ph),raw_force_norm_max_N=max(np.linalg.norm(x['contacts'][0]['raw_force']) for x in ph),force_formula=a['formula'])
   cr.append({**{k:v for k,v in r.items() if k!='archive'},**details})
  physical.append(dict(parent=c['parent'],K=c['K'],rows=cr))
  a=read(out/(pp+'-predictions.ptz'),c['predictions'])
  for goal in ('HOLD','FOLLOW'):
   g=torch.tensor(a['public_goals'][goal])/torch.tensor(SCALES)
   # Disable successor features: all candidate F vectors are the same zero vector.
   scores=(torch.zeros_like(a['preds']['I'])-models.B(g[None])).square().mean(-1);selected=int(scores.argmin());assert selected==0
   observed=next((x for x in cr if x['command']==selected),None)
   ablations.append(dict(parent=c['parent'],goal=goal,intervention='I_successor_features_zero_frozen_offline',selected_command=selected,physical_reuse=observed,physical_status='MATCHED_ACTUAL_BRANCH_REUSED' if observed is not None else 'UNMEASURED_NO_EXTRA_PHYSICS_AUTHORIZED'))
 summary=dict(status=ev['status'],scope='H0 one finite-outcome mechanism probe, frozen BFM/PD; no formal E/I Actor training',fit=learn,prospective_prediction=prediction,causal_effects=[dict(parent=c['parent'],K=c['K'],effects=c.get('effects',[])) for c in ev['results']],physical=physical,successor_disable=ablations,historical_G17_QA1_QA2='FAIL_UNCHANGED',Actor_updates=0,U_updates=0,Dev_Sealed_executed=False)
 write(ROOT/'research_notes/h0_results.json',summary)
 lines=['# H0 — finite interaction outcome prediction and causal Goal addressing','', 'Issue66 signed H0 v0.1; old G17/QA1/QA2 FAIL remains unchanged. This is a single bounded frozen-BFM probe, not formal E/I behavior training or full FB.','', '## Actual identity and data','', 'Three F1 exposed source/partner ancestors ×4 native command plans, actual first2s reused; three new train-only K40 ancestors ×4 real commands. Four independent source+partner ancestors reserved for physical evaluation, K80/K40 balanced. Whole ancestors kept apart; source prefixes actually executed; no pose-only restore. Original command/source manifests retained privately. Old F1 historically exposed dev is explicitly FIT, not a new blind claim; current R2 Dev/Sealed never opened.','', 'P-E predicts24 standardized raw future outcomes. P-I uses16D B/F with outcome reconstruction and fixed variance/covariance. Same current1639D restricted input, full proposed20×256 legal plan, hidden64/32,1024 groups×24, same parent-balanced sampler. Parameters E435512/I440024; additional I capacity is disclosed, no exact capacity-match claim. T is0-parameter finite selection, not arbitrary continuous latent generalization.','', 'K40/D8/10N is an approved new domain; no cross-K comparison is attributed to method. K80 unchanged calculation checked256 cases; every new branch has raw/applied force audit. Original PD, BFM and physical reward unchanged. No old high actor loaded; historical F1 visited prefixes can originate from old policies, while fit labels are actual native frozen-BFM commands.','', '## Goal intervention results','', '| Parent | K | Method | FOLLOW−HOLD wrist progress cm | error worsening cm | Quality pass | Local effect pass |','|---|---:|---|---:|---:|---|---|']
 for c in ev['results']:
  for a in c.get('effects',[]):lines.append(f"|{c['parent'][:12]}|{c['K']}|{a['method']}|{100*a['follow_minus_hold_m']:.4f}|{100*a['error_worsening_m']:.4f}|{a['quality_pass']}|{a['effect_pass']}|")
 passes={m:sum(a['effect_pass'] for c in ev['results'] for a in c.get('effects',[]) if a['method']==m) for m in ['E','I']}
 lines+=['',f"Independent local effects: P-E {passes['E']}/4, P-I {passes['I']}/4. Need at least2 independent ancestors, ≥2cm progress, ≤.5cm follow-error worsening and retained physical quality. This threshold is an engineering effect criterion, not statistical significance.", '', '## Future prediction on actually executed heldout commands','', '| Parent | K | distinct executed commands | E MSE | I MSE | constant MSE |','|---|---:|---:|---:|---:|---:|']
 for c in prediction:
  rows=c['rows'];v=[np.mean([r[k] for r in rows]) for k in ['E_MSE','I_MSE','constant_MSE']] if rows else [float('nan')]*3
  lines.append(f"|{c['parent'][:12]}|{c['K']}|{len(rows)}|{v[0]:.6f}|{v[1]:.6f}|{v[2]:.6f}|")
 lines+=['', 'Selected-command observations alone do not identify the full unexecuted candidate ranking. Duplicate method/Goal commands are deduplicated for prediction error. Full2s commands are real interventions; no predictive branch is called Student on-policy data.','', '## Representation diagnostics','', 'Fit-only B variance, spectrum/effective rank, same-phase cross-parent raw/random retrieval agreement, phase-only predictor and per-parent actual-command spread are in the machine summary. Raw self-nearest agreement is definitional, not evidence for a learned representation. Small fit-only retrieval or loss cannot establish causal control. Goal-shuffle reported by reusing frozen opposite-Goal choices is an algebraic matched-branch control, not additional physical trials or statistical replication.','', '## Evidence levels and scope','', f"Established: actual fixed-endpoint models and real Goal branches above; count satisfying local effects P-E={passes['E']},P-I={passes['I']}. Compatible mechanisms and unproven hypotheses must be separated below. No universal contact policy, general intent inference, Actor adaptation benefit or I-over-E significance is claimed. Four heldout parents are prospective for this fit, train-designated, not sealed.", '', 'H0 ends once. Next is a matched trainable-Actor contact/Goal E/I design, not H0.1, a new Q ranking exam, candidate sweep or Teacher repair chain. Any future training needs separate approval.']
 lines+=['','## Physical burden and control quality','', '| Parent | K | Method/Goal | cmd | error cm | force N | slip peak/p95 m/s | yaw excursion deg | q_PD jump deg | real q jump deg |', '|---|---:|---|---:|---:|---:|---|---:|---:|---:|']
 for c in physical:
  for r in c['rows']:
   q=r['quality'];lines.append(f"|{c['parent'][:12]}|{c['K']}|{r['method']}/{r['goal']}|{r['command']}|{100*r['error_rms_m']:.3f}|{r['force_rms_N']:.3f}|{q['slip_peak']:.6f}/{q['slip_p95']:.6f}|{q['yaw_excursion_deg']:.3f}|{r['q_PD_internal20ms_jump_max_deg']:.3f}|{r['actual_q_internal5ms_jump_max_deg']:.3f}|")
 lines+=['','All jumps in this table are internal to the saved window:20ms q_PD target changes differ from5ms actual joint motion. They do not imply the joint executed the target jump instantly. Per-axis force/error, applied cap fraction, raw demand, foot paths, support, root and actual dq are in the JSON.','',f"Successor-feature disable: {sum(x['physical_status']=='MATCHED_ACTUAL_BRANCH_REUSED' for x in ablations)}/8 goals have an already executed matching standing branch; remaining physical entries explicitly UNMEASURED. Offline candidate tie selects standing, not a demonstrated new physical outcome. Actual native standing baseline exists in every fit group; no extra independent-evaluation baseline branches were silently added.",'','The main intervention is a single current-state choice of a sustained2s plan with native50Hz body feedback. It is not a demonstrated high-level controller that replans at every100ms, not full-episode contact/actual-release recovery validation, and not stochastic deployment robustness.']
 (ROOT/'research_notes/H0_REPRESENTATION_AND_CAUSAL_ADDRESS_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(effects=passes,evaluation=ev['status'],head_manifest=p['manifest'])),flush=True)
if __name__=='__main__':main()
