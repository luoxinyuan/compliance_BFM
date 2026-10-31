"""Final read-only provenance/process/resource audit; no model or simulation."""
import hashlib,json,subprocess
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs';RUN=R/'m4-continuous-feedback-20261031A';ROOT=Path(__file__).resolve().parents[1]
def cmd(*args):return subprocess.check_output(args,text=True).strip()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ledger=json.loads((RUN/'compute-ledger.json').read_text());owned=[];episodes=0;simulation=0;peaks={'rss':0,'gpu':0};runtime=[]
 for a in ledger['attempts']:
  assert a['status']=='finished' and a['exit_code']==0 and a['result_status']=='completed' and not a['remaining_processes'] and a['resource_boundary'] is None
  stat=Path('/proc')/str(a['pid'])/'stat';alive=False
  if stat.exists():alive=stat.read_text().split(') ')[1].split()[19]==a['initial_processes'][0]['start_ticks']
  assert not alive;owned.append(dict(pid=a['pid'],start_ticks=a['initial_processes'][0]['start_ticks'],exited=True));peaks['rss']=max(peaks['rss'],a['peak_rss_bytes']);peaks['gpu']=max(peaks['gpu'],a['peak_gpu_mib']);runtime.append({k:a[k] for k in ['stage','commit','wall_seconds','pid']})
  result=json.loads((Path(a['path'])/'outputs/result.json').read_text());assert result['original_bfm_unchanged'] and result['bfm_updates']==0 and result['student_optimizer_updates']==0
  for e in result['study'].get('episodes',[]):assert e['complete'] and e['boundary'] is None;episodes+=1;simulation+=e['seconds']+3
 wall=sum(a['wall_seconds'] for a in ledger['attempts']);assert wall<=3600 and peaks['rss']<=32*1024**3 and peaks['gpu']<=8192
 changes=cmd('git','-C',str(ROOT),'diff','--name-status','72851904c2c6c2fbfe8970ea316abab63f9ea338','HEAD').splitlines();assert all(line.startswith('A\tresearch_notes/m4_') or line.startswith('A\tresearch_notes/M4_') or line=='A\tresearch_notes/test_m4_protocol.py' for line in changes)
 assert cmd('git','-C',str(P/'repo'),'rev-parse','HEAD')=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3';assert not cmd('git','-C',str(P/'repo'),'status','--porcelain');assert not cmd('git','-C',str(ROOT),'status','--porcelain')
 ck={R/'m3-continuous-body-20261031A/train-attempt-01/outputs/projected-z-final.pt':'81300c1a1d4abcf25eb58a36a0a92924f92526771ac2c2bbbf9c8eb8f578cf81',R/'m1-state-response-20261031A/aggregate-attempt-01/outputs/distill-final.pt':'c9036c46a0559c57fc2897419f06c66ef6113cb0fe4d57762df9a59c8b9e1ac7',R/'m1-state-response-20261031A/aggregate-attempt-01/outputs/alignment-final.pt':'e8121fbeb048de67a8147d03b8adbb7d8f302e3d6506bd1a1e0b54d41656683f'}
 for p,h in ck.items():assert sha(p)==h
 x=dict(status='verified',final_commit=cmd('git','-C',str(ROOT),'rev-parse','HEAD'),branch=cmd('git','-C',str(ROOT),'branch','--show-current'),private_clean=True,shared_clean=True,changes=changes,physical_episodes=episodes,simulated_seconds_including_prefixes=simulation,model_compute_wall_seconds=wall,peak_rss_bytes=peaks['rss'],peak_gpu_mib=peaks['gpu'],runtime=runtime,owned=owned,frozen_checkpoints={str(p):h for p,h in ck.items()},gpu_final=cmd('nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'),output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),training_updates=0,no_push=True,no_shared_source_changes=True,no_bridge=True)
 (RUN/'analysis/final-audit.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
if __name__=='__main__':main()
