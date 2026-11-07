"""Finite post-BC saved-trajectory analysis and rendering; never starts training."""
from pathlib import Path
import json,subprocess,time
S=Path(__file__).resolve().parents[2]
def main():
 deadline=time.monotonic()+18000
 while not (S/'finite-sequence-result.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('finite review wait limit')
  time.sleep(30)
 result=json.loads((S/'finite-sequence-result.json').read_text())
 assert result['status']=='completed',result
 py='/mnt/data/temp_user/yi.gao/projects/compliance_BFM/.venv/bin/python';r=S/'review-repo'
 for name,mode,limit in [('review-01','analyze',300),('video-01','render',600)]:
  out=S/'runs'/name;assert not out.exists()
  cmd=[py,'-u',str(r/'research_notes/rolling_job.py'),name,str(r/'research_notes/native_transfer_review.py'),str(limit),'--mode',mode,'--output',str(out)]
  with (S/'logs'/(name+'-supervisor.log')).open('w') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,(name,rc)
 (S/'review-sequence-result.json').write_text(json.dumps(dict(status='server_artifacts_ready_pending_Mac_QA_and_scientific_closeout',time=time.time())))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (S/'review-sequence-result.json').write_text(json.dumps(dict(status='stopped_on_error',error=repr(e),time=time.time())));raise
