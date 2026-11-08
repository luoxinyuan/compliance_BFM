"""Validity-aware Issue53 saved-artifact closeout. No model inference, physics or updates."""
import argparse,copy,gzip,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent

def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(x):return dict(n=len(x),mean=float(np.mean(x)),p50=float(np.median(x)),p90=float(np.quantile(x,.9)),minimum=float(min(x)),maximum=float(max(x)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);start=time.monotonic()
 import torch
 D=S/'runs/diagnose-01';C=S/'runs/repair-short-01';E=S/'runs/bc-eval-01';queries=read(D/'searches.json');extra=read(C/'searches.json')+read(S/'runs/moving-cost-01/searches.json');shadow=[q for q in queries if 'shadow' in q];allq=queries+extra
 err=read(S/'recovery-validity-erratum.json');correct=read(C/'corrected-recovery.json');validation=read(S/'runs/snapshot-check-01/validation.json')
 assert all(r['fixed_cross_reset']['all_physics_rows_equal'] and r['fixed_cross_reset']['all_control_rows_equal'] for r in validation)
 labels=torch.load(D/'interactive-labels.pt',map_location='cpu',weights_only=True)['rows'];good=[r for r in labels if r['valid'] and r['episode'].endswith('normal72_4.7')]
 if correct['qualified']:good+=torch.load(C/'repaired-labels.pt',map_location='cpu',weights_only=True)['rows']
 unique={}
 for r in good:unique.setdefault(r['state_fingerprint'],r)
 new=list(unique.values());torch.save(dict(rows=new,scope='corrected verified actual recovery windows only; NOT trained in Issue53',bad_original_postonset_rows_excluded=True,old_data_source='Issue51 valid train311, separate original dataset',early_labels_continuous_replay_all_errors_zero=read(S/'runs/early-replay-01/verification.json')['errors'],erratum=err),a.output/'corrected-new-labels.pt')
 fb=[]
 for q in shadow:
  sh=q['shadow'];cs=q['candidates'];fs=sh['F']['all_F_scores'][:7];ps=[r['score'] for r in cs[:7]]
  def rank(v):
   v=np.array(v);return np.array([np.sum(v<x)+.5*(np.sum(v==x)-1) for x in v])
  rf,rp=rank(fs),rank(ps);rho=float(np.corrcoef(rf,rp)[0,1]) if rf.std()>0 and rp.std()>0 else None
  fb.append(dict(label=q['label'],time=q['time'],replacement_gain=sh['replacement_gain'],nominee_wins=sh['nominee_wins'],F_top4_retains_top7=sh['F_top4_retains_physical_top7'],F_top4_score=max(ps[j] for j in sh['F_rank7'][:4]),same_count_fixed4_score=max(ps[j] for j in [0,1,4,5]),rank_correlation=rho,nominated=sh['nominated'],F_scores=sh['F']['all_F_scores'],physical_scores=[c['score'] for c in cs],physical_xyz_error_force=[{k:c['physical'][k] for k in ['error_cm','force_rms','force_xyz_rms','cap_fraction','pelvis_yaw_change_deg','min_height','max_tilt']} for c in cs]))
 counts={}
 for folder in [D,C,S/'runs/moving-cost-01',S/'runs/snapshot-check-01']:
  for k,v in read(folder/'counts.json').items():counts[k]=counts.get(k,0)+v
 counts['supervised_updates']=2000;counts['Student_PPO_decisions']=0
 counts['actual_early_replay_physics_steps']=read(S/'runs/early-replay-01/verification.json')['actual_physics_steps']
 counts['early_replay_reset_physics_steps']=read(S/'runs/early-replay-01/verification.json')['reset_physics_steps']
 counts['actual_valid_control_eval_physics_steps']=read(S/'runs/valid-control-01/result.json')['study']['actual_evaluation_physics_steps']
 counts['valid_control_reset_physics_steps']=read(S/'runs/valid-control-01/result.json')['study']['reset_physics_steps']
 assert counts['full_queries']==96 and counts['FB_extra_sequences']==40
 cases=[]
 for row in read(D/'summaries.json'):
  r=copy.deepcopy(row);ep=read(D/(r['label']+'.json.gz'));r['condition']=ep['condition']
  r['validity']='valid actual source Student trajectory' if r['label'].startswith('S4000_') else ('valid early intervention, relative onset_actual contract' if r['label'].endswith('normal72_4.7') else 'protocol affected cross-reset anchor; NOT qualification or source-domain outcome')
  cases.append(r)
 cases_correct=read(C/'summaries.json');bc=read(E/'summaries.json');models={}
 for f in (S/'runs/bc-fit-01').glob('*.pt'):models[str(f.relative_to(S))]=sha(f)
 ranks=[r['rank_correlation'] for r in fb if r['rank_correlation'] is not None];base=[q for q in allq if 'shadow' not in q]
 ledger=read(S/'ledger.json');ledgers=[j for j in ledger['jobs'] if j['name']!='final-analysis-01'];spent=sum(j.get('elapsed_s',0) for j in ledgers)+ledger.get('auxiliary_CPU_analysis_reserve_s',0)
 # A formal current-gate file supersedes original diagnostic gate rather than editing old evidence.
 gate=dict(stage_status='stopped_at_validity_and_BC_budget_boundary',original3of3_claim_retracted=True,valid_early_source_start=True,corrected_postonset=correct['qualified'],unresolved_late6s=True,Teacher_validity='two limited windows, not a universal rate',unique_valid_new_states=len(new),independent_Student_start_states=2 if correct['qualified'] else 1,original_mixed_training_has40_affected_states=True,interactive_learning_advantage='not established',Student_PPO_started=False,new_sealed_evaluations='clean frozen S4000/old-only500 only; affected interactive model not evaluated',extra_BC_allowed=False,next='freeze valid old-data BC500 as new limited S-init; next separately authorized frozen-BC vs own-Student-PPO test, with sparse validated Teacher recovery; do not require another mixed BC perfect gate')
 (a.output/'current-gate.json').write_text(json.dumps(gate,indent=2))
 evidence=dict(status='closed_with_partial_positive_evidence_and_repaired_implementation_failure',counts=counts,validity_erratum=err,current_gate=gate,snapshot_validation=validation,first_recovery=next(r for r in read(D/'recoveries.json') if r['time']==4.7),original_full_actual_results=cases,corrected_postonset=correct,corrected_full_actual=cases_correct,BC_partial_real=bc,BC_training=read(S/'runs/bc-fit-01/training.json'),BC_partial_selection=read(E/'selection.json'),new_sealed_files=[str(p) for p in E.glob('*FINAL*')],Student_PPO_decisions=0,valid_old_BC_control_final=read(S/'runs/valid-control-01/summaries.json'),early_label_verification=read(S/'runs/early-replay-01/verification.json'),model_shas=models,valid_new_labels=dict(path=str(a.output/'corrected-new-labels.pt'),sha256=sha(a.output/'corrected-new-labels.pt'),unique_states=len(new),trained=False,not_independent_episodes=True),query_cost=stat([q['wall_s'] for q in base]),shadow_query_cost=stat([q['wall_s'] for q in shadow]),candidate_cost=stat([c['wall_s'] for q in allq for c in q['candidates']]),save_cost=stat([q['snapshot_s'] for q in allq]),restore_cost=stat([s for q in allq for s in q['restore_s']]),F_score_cost=stat([q['shadow']['F']['seconds'] for q in shadow]),FB=dict(n=len(fb),wins=sum(r['nominee_wins'] for r in fb),top4_preserved=sum(r['F_top4_retains_top7'] for r in fb),rank_correlation_median=float(np.median(ranks)),same4_fixed_reference=[0,1,4,5],same4_reference_scope='posthoc descriptive only, not preregistered comparison',comparisons=fb,current_proposer_paused=True,formal_Teacher_unchanged=True,moving_State_not_in_shadow20=True),moving_cost=read(S/'runs/moving-cost-01/moving-cost.json'),BFM_call_accounting=dict(candidate_control_steps=sum(c['BFM_calls'] for q in allq for c in q['candidates']),candidate_actor_calls=sum(c['BFM_calls'] for q in allq for c in q['candidates'])*2,note='original field BFM_calls counts50Hz controlsteps; nominal standing diagnostic adds secondactor call perstep; F scoring has separate actor calls; setup calls separate'),robot_reward_audit=read(S/'runs/reward-readonly-02/result.json'),external_reward_audit=read(S/'runs/reward-readonly-02/external-score-check.json'),actor_observation_audit=read(D/'observation-source-audit.json'),resources=dict(jobs=ledgers,job_seconds_with_CPU_reserve_before_final_artifact_jobs=spent,output_bytes_so_far=sum(p.stat().st_size for p in S.rglob('*') if p.is_file())),source_runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
 (a.output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,3,figsize=(12,6),sharex=True)
 for col,case in enumerate(['short08','normal72','stop66']):
  for role,color,style in [('old_data_only','C0','-'),('interactive','C3','--')]:
   pts=[]
   for r in bc:
    label=r['label']
    if label=='S4000_dev_'+case:pts.append((0,r['all']))
    elif label.startswith(role+'_') and label.endswith('_dev_'+case):pts.append((int(label.split('_dev_')[0].split('_')[-1]),r['all']))
   pts=sorted(pts,key=lambda x:x[0]);lab='old only / valid control' if role=='old_data_only' else 'mixed / protocol affected'
   for row,key in enumerate(['error_cm','force_rms']):axs[row,col].plot([x[0] for x in pts],[x[1][key] for x in pts],style,marker='o',color=color,label=lab)
  axs[0,col].set_title(case);axs[1,col].set_xlabel('Additional BC updates');axs[0,col].set_ylabel('Active3D RMSE (cm)');axs[1,col].set_ylabel('Actual force RMS (N)')
  for row in range(2):axs[row,col].grid(alpha=.25)
 axs[0,0].legend(fontsize=7);fig.suptitle('Issue53 development only; mixed-data arm validity affected; no sealed or Student PPO');fig.tight_layout();fig.savefig(a.output/'development.png',dpi=150);plt.close(fig)
 (a.output/'result.json').write_text(json.dumps(dict(status='completed',scope='saved data only',new_query=0,new_Physics=0,new_updates=0,wall_s=time.monotonic()-start),indent=2));print(json.dumps(dict(gate=gate,counts=counts,spent=spent),indent=2))
if __name__=='__main__':main()
