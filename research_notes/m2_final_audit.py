"""Read-only scientific/source/resource closure; writes derived audit, no model execution."""
import ast,hashlib,json,subprocess
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM'); ROOT=Path(__file__).resolve().parents[1]
R=P/'bridge-v2-owned/runs/m2-unseen-body-20261031A'; M=P/'bridge-v2-owned/runs/m1-state-response-20261031A'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def main():
 protocol=read(ROOT/'research_notes/m2_protocol.json');ledger=read(R/'compute-ledger.json');summary=read(R/'analysis/physical-summary.json');checks={}
 assert len(ledger['attempts'])==4
 for a in ledger['attempts']:
  assert a['status']=='finished' and a['exit_code']==0 and a['result_status']=='completed' and not a['remaining_processes'] and a['resource_boundary'] is None
  proc=Path('/proc')/str(a['pid'])/'stat'
  assert not proc.exists() or proc.read_text().split()[21]!=a['initial_processes'][0]['start_ticks']
  r=read(Path(a['path'])/'outputs/result.json');assert r['bfm_updates']==r['student_optimizer_updates']==0 and r['original_bfm_unchanged']
 checks['completed_stages']=[{k:a[k] for k in ['stage','commit','pid','wall_seconds','peak_rss_bytes','peak_gpu_mib']} for a in ledger['attempts']]
 checks['wall_seconds']=sum(a['wall_seconds'] for a in ledger['attempts']);assert checks['wall_seconds']<protocol['resources']['compute_cumulative_wall_seconds']
 assert len(summary['episodes'])==14
 checks['physical_episodes']=14;checks['physical_seconds_including_repeated_prefix']=14*23.4;checks['recorded_seconds_after_prefix']=14*20.4
 checks['all_boundaries']={k:e['boundary'] for k,e in summary['episodes'].items()};assert all(v is None for v in checks['all_boundaries'].values())
 checks['action_PD_max_errors']={k:max(e['deployment_parity'][k] for e in summary['episodes'].values()) for k in ['q_PD_max_error','torque_max_error']}
 checks['latent_norm_max_error']=max(e['latent_norm_max_error'] for e in summary['episodes'].values())
 checks['five_precontact_pairs_exact']=all(max(p['errors'].values())==0 for p in summary['precontact_parity']);assert checks['five_precontact_pairs_exact']
 checks['frozen_checkpoints']={k:sha(M/'aggregate-attempt-01/outputs'/f'{k}-final.pt') for k in protocol['checkpoints']};assert checks['frozen_checkpoints']==protocol['checkpoints']
 c={}
 for node in ast.parse((ROOT/'research_notes/g1_action_contract_check.py').read_text()).body:
  if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['PINS','WEIGHT_SHA']:c[node.targets[0].id]=ast.literal_eval(node.value)
 checks['core_pins']={f:sha(ROOT/f) for f in c['PINS']};assert checks['core_pins']==c['PINS']
 checks['bfm_sha']=sha(P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model/model.safetensors');assert checks['bfm_sha']==c['WEIGHT_SHA']
 checks['shared_head']=git(P/'repo','rev-parse','HEAD');assert checks['shared_head']=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3'
 checks['shared_status']=git(P/'repo','status','--porcelain');assert not checks['shared_status']
 checks['source_head']=git(ROOT,'rev-parse','HEAD');checks['private_status']=git(ROOT,'status','--porcelain')
 checks['diff_stat']=git(ROOT,'diff','--stat',protocol['base_commit']);names=git(ROOT,'diff','--name-only',protocol['base_commit']).splitlines();assert all(n.startswith(('research_notes/m2','research_notes/M2','research_notes/test_m2')) for n in names)
 checks['output_bytes']=sum(f.stat().st_size for f in R.rglob('*') if f.is_file());assert checks['output_bytes']<2*1024**3
 checks['scope']='14 physical episodes, one saved-state diagnostic, zero training; no push/shared modifications. Diagnostic latent updates are temporary and never deployed.'
 (R/'analysis/final-audit.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
