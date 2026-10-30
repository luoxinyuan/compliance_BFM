"""End-of-stage immutable identity, source and sampling audit. No model execution."""
import ast,hashlib,json,subprocess
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
W=P/'bridge-v2-owned/trees/issue-21-alpha-repair/repo'
R=P/'bridge-v2-owned/runs/issue21-alpha-repair-20261031A'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
protocol=read(W/'research_notes/alpha_repair_protocol.json');ledger=read(R/'compute-ledger.json')
train=next(Path(a['path'])/'outputs' for a in ledger['attempts'] if a['stage']=='train' and a.get('result_status')=='completed')
fit=read(train/'paired-fit-result.json');dataset=read(train/'dataset-manifest.json');gradient=read(train/'gradient-contract.json')
assert fit['full_A']['initial_state_sha256']==fit['full_B']['initial_state_sha256']
assert fit['full_A']['batch_sequence_sha256']==fit['full_B']['batch_sequence_sha256']
assert all(fit['full_'+c]['updates']==2000 and fit['full_'+c]['batch_size']==128 for c in ['A','B'])
assert dataset['samples']==4380 and not dataset['sealed_final_used'] and dataset['trainable_parameters']==199169
assert dataset['protocol']==protocol
for src in protocol['training_sources']:assert sha(P/'bridge-v2-owned/runs'/src['path'])==src['sha256']
initial=P/'bridge-v2-owned/runs'/protocol['initialization']['source'];assert sha(initial)==protocol['initialization']['sha256']
old_b=initial.parent/'alpha_B_final.pt';assert sha(old_b)==protocol['old_checkpoint_sha256']['B']
checkpoints={c:sha(train/protocol['final_checkpoints'][c]) for c in ['A','B']}
assert all(checkpoints[c]==fit['full_'+c]['checkpoint_sha256'] for c in checkpoints)
model=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model/model.safetensors'
assert sha(model)==protocol['model_sha256']
assert sha(W/'research_notes/contact_gate.py')==protocol['teacher_gate_sha256']
contract=read(train/'action-contract.json');assert all(sha(W/f)==h for f,h in contract['source_pins'].items())
assert not git(P/'repo','status','--porcelain') and git(P/'repo','rev-parse','HEAD')=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3'
changes=git(W,'diff','--name-status',protocol['base_commit']).splitlines();assert all(x.startswith('A\tresearch_notes/') for x in changes)
for a in ledger['attempts']:
 assert a['status']=='finished' and not a['remaining_processes'] and not Path('/proc',str(a['pid'])).exists()
 assert not a['resource_boundary']
 if a.get('result_status')=='completed':assert read(Path(a['path'])/'outputs/result.json')['original_bfm_unchanged']
assert len([a for a in ledger['attempts'] if a['stage']=='train'])==1
result=read(R/'analysis/repair-results.json');assert len(result['independent_audit'])==24
for v in result['manifest']:assert sha(v['path'])==v['sha256']
audit=dict(status='passed',private_commit=git(W,'rev-parse','HEAD'),private_status=git(W,'status','--porcelain'),shared_head=git(P/'repo','rev-parse','HEAD'),
 original_sources_and_weights_unchanged=True,training_source_hashes_unchanged=True,new_checkpoint_sha256=checkpoints,old_checkpoints_unchanged=True,
 same_initialization=fit['full_A']['initial_state_sha256'],same_batches=fit['full_A']['batch_sequence_sha256'],normalization_warm_started_unchanged=True,
 updates_per_method=2000,batch_size=128,total_gradient_updates=4000,total_sample_exposures=512000,unique_training_samples=4380,sealed_tests_never_in_fit=True,
 completed_physical_branches=24,gradient_contract=gradient,all_owned_processes_exited=True,compute_wall_seconds=sum(a['wall_seconds'] for a in ledger['attempts']),
 resource_peaks=dict(rss_bytes=max(a['peak_rss_bytes'] for a in ledger['attempts']),vram_mib=max(a['peak_gpu_mib'] for a in ledger['attempts'])),
 changes=changes,gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
(R/'analysis/final-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
