"""One bounded serial continuation of the two already approved Q-only arms."""
import json,time,subprocess,os
from pathlib import Path
S=Path(__file__).resolve().parents[1]
P=S.parents[1];PY=P/'.venv/bin/python';R=S/'runtime-v4'
DEST=S/'jobs/E-I-warmup3500-queue.json'
def write(x):
 q=DEST.with_suffix('.pending');q.write_text(json.dumps(x,indent=2));os.replace(q,DEST)
record=dict(pid=os.getpid(),start=time.time(),status='waiting_E',source_runtime='f62f918c7c545e534ef3b96583231324c0ab1848',scope='E then I 3500 Q-only; actor/U never enabled',waiting_accounting='nested E supervisor wait, counted by E sidecar, not double charged')
write(record)
deadline=time.monotonic()+2200
while True:
 path=S/'jobs/E-warmup3500-10.json'
 row=json.loads(path.read_text()) if path.exists() else {}
 if 'end' in row:break
 if time.monotonic()>deadline:
  record.update(status='stopped_wait_deadline',end=time.time());write(record);raise SystemExit(2)
 time.sleep(5)
if row.get('exit_code')!=0 or row.get('qualification_status')!='q_only_entry_complete':
 record.update(status='stopped_E_not_complete',E=row,end=time.time());write(record);raise SystemExit(3)
manifest=S/'runs/E-warmup3500-10/checkpoints/round-14/manifest.json'
assert json.loads(manifest.read_text())['transitions']==3500
assert not (S/'active-owned-job.lock').exists()
record.update(status='launching_I',E_end=row['end']);write(record)
cmd=[str(PY),str(R/'research/interaction_fb/job_q_entry.py'),'--name','I-warmup3500-11','--arm','I','--blocks','14','--resume',str(S/'runs/I-four-blocks-06/checkpoints/round-4'),'--limit','1900']
record['I_command']=cmd;write(record)
with (S/'jobs/I-warmup3500-11-supervisor.log').open('x') as log:
 result=subprocess.run(cmd,cwd=S,stdout=log,stderr=subprocess.STDOUT)
record.update(status='finished' if result.returncode==0 else 'stopped_I_failed',I_supervisor_exit=result.returncode,end=time.time());write(record)
