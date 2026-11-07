"""One finite Issue51 sequence after its exact owned Teacher job; no daemon/service."""
from pathlib import Path
import json,os,subprocess,time,sys
S=Path(__file__).resolve().parents[2]
def write(p,x):p.write_text(json.dumps(x,indent=2))
def main():
 deadline=time.monotonic()+12500
 teacher=json.loads((S/'teacher-supervisor.json').read_text());wait_start=time.monotonic()
 while True:
  l=json.loads((S/'ledger.json').read_text());j=next((x for x in l['jobs'] if x['name']=='teacher-coverage-01'),None)
  if j and j['status']!='running':break
  if time.monotonic()>deadline:raise TimeoutError('finite queue wait budget')
  # Only observe exact recorded supervisor identity; never signal any other task.
  p=Path('/proc')/str(teacher['pid'])/'stat'
  if p.exists() and p.read_text().split()[21]!=teacher['start_ticks']:raise RuntimeError('Teacher supervisor PID reused')
  time.sleep(30)
 write(S/'continuation-wait.json',dict(wait_seconds=time.monotonic()-wait_start,not_GPU_job_time=True))
 assert j['status']=='completed',('teacher failed; no automatic restart',j)
 repo=S/'bc-repo';py='/mnt/data/temp_user/yi.gao/projects/compliance_BFM/.venv/bin/python'
 for name,mode,limit in [('data-prepare-01','prepare',1800),('bc-fit-01','fit',1200),('bc-eval-01','eval',2400)]:
  out=S/'runs'/name
  if out.exists():raise RuntimeError('refuse overwrite/restart '+str(out))
  cmd=[py,'-u',str(repo/'research_notes/rolling_job.py'),name,str(repo/'research_notes/native_transfer.py'),str(limit),'--mode',mode,'--output',str(out)]
  with (S/'logs'/(name+'-supervisor.log')).open('w') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError(name+' failed; evidence kept, no automatic retry')
 l=json.loads((S/'ledger.json').read_text());l['status']='physics_and_BC_complete_pending_scientific_review_and_videos';write(S/'ledger.json',l)
 write(S/'finite-sequence-result.json',dict(status='completed',finished_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),Student_PPO=0))
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(S/'finite-sequence-result.json',dict(status='stopped_on_error',error=repr(e),when=time.time()));raise
