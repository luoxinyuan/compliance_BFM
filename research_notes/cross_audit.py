"""Read-only final identity/resource/protocol and physical source audit."""
import hashlib,json,subprocess
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
W=Path(__file__).resolve().parents[1]
R=P/'bridge-v2-owned/runs/issue21-alpha-cross-20261031A'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
p=read(W/'research_notes/alpha_cross_protocol.json');ledger=read(R/'compute-ledger.json')
checkpoints={};training={}
if (R/'train-attempt-01/outputs/result.json').exists():
 tp=read(W/'research_notes/alpha_cross_training_protocol.json');tr=R/'train-attempt-01/outputs'
 fit=read(tr/'paired-fit-result.json');ds=read(tr/'dataset-manifest.json')
 assert fit['full_A']['initial_state_sha256']==fit['full_B']['initial_state_sha256']
 assert fit['full_A']['batch_sequence_sha256']==fit['full_B']['batch_sequence_sha256']
 assert ds['samples']==10020 and ds['trainable_parameters']==199169 and not ds['sealed_final_used']
 assert ds['protocol']==tp
 for src in tp['training_sources']:
  assert sha(P/'bridge-v2-owned/runs'/src['path'])==src['sha256']
  assert not read(P/'bridge-v2-owned/runs'/src['path'])['condition']['name'].startswith('sealed_')
 for ctl in ['A','B']:
  assert fit['full_'+ctl]['updates']==2000 and fit['full_'+ctl]['batch_size']==256
  checkpoints[ctl]=sha(tr/tp['final_checkpoints'][ctl]);assert checkpoints[ctl]==fit['full_'+ctl]['checkpoint_sha256']
 training=dict(updates_each=2000,samples_each=512000,unique_samples=10020,same_initialization=fit['full_A']['initial_state_sha256'],same_batches=fit['full_A']['batch_sequence_sha256'],gradient=read(tr/'gradient-contract.json'))
for ctl in ['A','B']:
 assert sha(P/f'bridge-v2-owned/runs/issue21-alpha-repair-20261031A/train-attempt-01/outputs/repair_{ctl}_final.pt')==p['frozen_checkpoints'][ctl]
assert sha(P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model/model.safetensors')==p['model_sha256']
assert sha(W/'research_notes/contact_gate.py')==p['teacher_gate_sha256']
contract=read(R/'diagnostic-attempt-01/outputs/action-contract.json')
assert all(sha(W/f)==h for f,h in contract['source_pins'].items())
assert not git(P/'repo','status','--porcelain') and git(P/'repo','rev-parse','HEAD')=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3'
changes=git(W,'diff','--name-status',p['base_commit']).splitlines();assert all(v.startswith('A\tresearch_notes/') for v in changes)
assert sum(a['stage']=='train' for a in ledger['attempts'])<=1
for a in ledger['attempts']:
 assert a['status']=='finished' and not a['remaining_processes'] and not Path('/proc',str(a['pid'])).exists()
 assert not a['resource_boundary']
 assert read(Path(a['path'])/'outputs/result.json')['original_bfm_unchanged']
for stage in ['diagnostic','final']:
 f=R/'analysis'/(stage+'-analysis.json')
 if f.exists():
  for v in read(f)['results'].values():assert sha(v['source'])==v['sha256']
wall=sum(a['wall_seconds'] for a in ledger['attempts']);assert wall<=3600
size=sum(f.stat().st_size for f in R.rglob('*') if f.is_file());assert size<4*1024**3
out=dict(status='passed',private_commit=git(W,'rev-parse','HEAD'),private_status=git(W,'status','--porcelain'),shared_head=git(P/'repo','rev-parse','HEAD'),original_sources_weights_and_old_students_unchanged=True,changes=changes,checkpoints=checkpoints,training=training,output_bytes=size,wall_seconds=wall,owned_processes_exited=True,resource_peaks=dict(rss_bytes=max(a['peak_rss_bytes'] for a in ledger['attempts']),gpu_mib=max(a['peak_gpu_mib'] for a in ledger['attempts'])),gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
(R/'analysis/final-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
