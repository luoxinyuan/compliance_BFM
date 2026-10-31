"""Closure audit of recorded evidence, source hashes and owned processes. No model runs."""
import ast,hashlib,json,subprocess
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');ROOT=Path(__file__).resolve().parents[1]
R=P/'bridge-v2-owned/runs/m3-continuous-body-20261031A';M=P/'bridge-v2-owned/runs/m1-state-response-20261031A'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def main():
 protocol=read(ROOT/'research_notes/m3_protocol.json');ledger=read(R/'compute-ledger.json');final=read(R/'analysis/final-summary.json');prior=read(R/'analysis/qualification-analysis.json');checks={};updates=0
 assert len(ledger['attempts'])==9
 for a in ledger['attempts']:
  assert a['status']=='finished' and a['exit_code']==0 and a['result_status']=='completed' and not a['remaining_processes'] and a['resource_boundary'] is None
  proc=Path('/proc')/str(a['pid'])/'stat';assert not proc.exists() or proc.read_text().split()[21]!=a['initial_processes'][0]['start_ticks']
  r=read(Path(a['path'])/'outputs/result.json');assert r['bfm_updates']==0 and r['original_bfm_unchanged'];updates+=r['student_optimizer_updates']
 assert updates==1500
 checks['stages']=[{k:a[k] for k in ['stage','commit','pid','wall_seconds','peak_rss_bytes','peak_gpu_mib']} for a in ledger['attempts']]
 checks['wall_seconds']=sum(a['wall_seconds'] for a in ledger['attempts']);assert checks['wall_seconds']<protocol['resources']['compute_cumulative_wall_seconds']
 all_eps=list(prior['episodes'].values())+list(final['episodes'].values());assert len(all_eps)==35
 checks['physical_episodes']=len(all_eps);checks['recorded_physics_seconds']=sum(e['seconds'] for e in all_eps);checks['physical_seconds_with_prefixes']=checks['recorded_physics_seconds']+3*len(all_eps)
 checks['boundaries']=[e['boundary'] for e in all_eps];assert all(v is None for v in checks['boundaries'])
 checks['action_PD_max_errors']={k:max(e['deployment_audit'][k] for e in all_eps) for k in ['q_PD_max_error','torque_max_error']}
 checks['latent_norm_max_error']=max(e['latent_norm_max_error'] for e in all_eps)
 checks['paired_precontact_exact']={k:max(v['precontact_parity'].values())==0 for k,v in final['contact_vs_free'].items()};assert all(checks['paired_precontact_exact'].values())
 checks['old_checkpoints']={k:sha(M/'aggregate-attempt-01/outputs'/f'{k}-final.pt') for k in protocol['checkpoints']};assert checks['old_checkpoints']==protocol['checkpoints']
 train=read(R/'train-attempt-01/outputs/training.json');checks['new_checkpoint_sha']=sha(R/'train-attempt-01/outputs/projected-z-final.pt');assert checks['new_checkpoint_sha']==train['sha256']=='81300c1a1d4abcf25eb58a36a0a92924f92526771ac2c2bbbf9c8eb8f578cf81'
 checks['new_updates']=updates;checks['train_samples']=train['old_samples']+train['new_samples'];assert train['no_checkpoint_selection'] and train['no_student_visit_aggregation_this_stage']
 checks['label_manifest']=read(R/'train-attempt-01/outputs/dataset-manifest.json')
 constants={}
 for n in ast.parse((ROOT/'research_notes/g1_action_contract_check.py').read_text()).body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PINS','WEIGHT_SHA']:constants[n.targets[0].id]=ast.literal_eval(n.value)
 checks['core_pins']={f:sha(ROOT/f) for f in constants['PINS']};assert checks['core_pins']==constants['PINS']
 checks['bfm_sha']=sha(P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model/model.safetensors');assert checks['bfm_sha']==constants['WEIGHT_SHA']
 checks['shared_head']=git(P/'repo','rev-parse','HEAD');assert checks['shared_head']=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3'
 checks['shared_status']=git(P/'repo','status','--porcelain');assert not checks['shared_status']
 checks['source_head']=git(ROOT,'rev-parse','HEAD');checks['private_status']=git(ROOT,'status','--porcelain')
 checks['diff_stat']=git(ROOT,'diff','--stat',protocol['base_commit']);names=git(ROOT,'diff','--name-only',protocol['base_commit']).splitlines();assert all(n.startswith(('research_notes/m3','research_notes/M3','research_notes/test_m3')) for n in names)
 checks['output_bytes']=sum(f.stat().st_size for f in R.rglob('*') if f.is_file());assert checks['output_bytes']<3*1024**3
 checks['scope']='35 finite physics branches, finite saved-state diagnostics and 1500 supervised updates. Final data never fitted. No push/shared source changes.'
 (R/'analysis/final-audit.json').write_text(json.dumps(checks,indent=2,allow_nan=False));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
