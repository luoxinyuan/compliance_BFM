"""Read-only provenance and recorded-state audit after all owned model processes exit."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUNS=P/'bridge-v2-owned/runs';R=RUNS/'issue21-skill-compat-20261031A'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def main():
 ledger=read(R/'compute-ledger.json');protocol=read(ROOT/'research_notes/skill_compat_protocol.json')
 completed=[a for a in ledger['attempts'] if a.get('result_status')=='completed'];pairs=next(a for a in completed if a['stage']=='pairs');out=Path(pairs['path'])/'outputs'
 assert all(a['status']=='finished' and not a.get('remaining_processes') for a in ledger['attempts'])
 assert all(not Path('/proc/'+str(a['pid'])).exists() for a in ledger['attempts'])
 contract=read(out/'action-contract.json');pins=contract['source_pins'];assert all(sha(ROOT/f)==h for f,h in pins.items())
 assert all(sha(RUNS/protocol[n]['path'])==protocol[n]['sha256'] for n in ['S1','A2'])
 shared=P/'repo';assert git(shared,'rev-parse','HEAD')=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3' and git(shared,'status','--porcelain')==''
 changes=git(ROOT,'diff','--name-only',protocol['base_commit'],'HEAD').splitlines();assert all(p.startswith('research_notes/skill_compat') or p=='research_notes/test_skill_compat.py' for p in changes)
 oldlocal=P/'bridge-v2-owned/trees/issue-21-targeted-repair/repo'
 assert git(ROOT,'diff','d968c3f42a9a3d867c01226167ab020d01134ba5','--','humanoidverse/force_compliance','humanoidverse/agents','research_notes/contact_projector.py')==''
 # This is a reproduction check, not new independent evidence: A2 dev_late was already evaluated.
 old=read(RUNS/'issue21-alpha-cross-20261031A/final-attempt-01/outputs/dev_late__A.json')
 new=read(out/'large__native.json');compare={}
 for k in ['qpos','qvel','force','wrist','root','anchor']:
  compare[k]=float(abs(np.array([r[k] for r in old['physics']])-np.array([r[k] for r in new['physics']])).max())
 for k in ['alpha','action','q_PD']:
  compare[k]=float(abs(np.array([r[k] for r in old['control']])-np.array([r[k] for r in new['control']])).max())
 assert max(compare.values())<1e-6,compare
 result=dict(runtime_commits=[a['commit'] for a in ledger['attempts']],final_commit=git(ROOT,'rev-parse','HEAD'),shared_clean=True,source_pins_unchanged=True,
  original_controllers_unchanged=True,student_SHA_unchanged=True,owned_processes_exited=True,
  native_large_vs_historical_A2_dev_late_max_abs=compare,wall_seconds=sum(a['wall_seconds'] for a in ledger['attempts']),
  peak_rss_gib=max(a['peak_rss_bytes'] for a in ledger['attempts'])/1024**3,peak_gpu_mib=max(a['peak_gpu_mib'] for a in ledger['attempts']),
  physical_results=[read(Path(a['path'])/'outputs/physical-result.json') for a in completed],core_files=changes,shared_head=git(shared,'rev-parse','HEAD'))
 (R/'analysis/final-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
