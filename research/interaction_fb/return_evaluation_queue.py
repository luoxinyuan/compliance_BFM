"""Fixed exposed evaluation sequence, serial GPU0, stop on every failed gate."""
import json,os,subprocess,time
from pathlib import Path
S=Path(__file__).resolve().parents[3];R=S/'runtime';P=S.parents[1]
assert not (S/'active-owned-job.lock').exists()
records=[]
for index in (2,3,4,5,6,7,0):
    name=f'evaluate-case-{index:02d}-01'
    command=[str(P/'.venv/bin/python'),'-u',str(R/'research/interaction_fb/return_job.py'),
             '--name',name,'--case-index',str(index),'--evaluate','--limit','600']
    with (S/'jobs'/(name+'-supervisor.log')).open('x') as output:
        proc=subprocess.Popen(command,cwd=R,stdout=output,stderr=subprocess.STDOUT)
        (S/'jobs'/(name+'-supervisor.pid')).write_text(str(proc.pid))
        rc=proc.wait()
    record=dict(case_index=index,name=name,supervisor_pid=proc.pid,returncode=rc)
    if rc==0:
        evaluation=json.loads((S/'runs'/name/'evaluation.json').read_text())
        record['scientific_status']=evaluation['status']
    records.append(record)
    (S/'evaluation-queue.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(record),flush=True)
    if rc!=0 or record.get('scientific_status')!='matched_evaluation_completed':
        print('STOPPED: no parent skipping, reroll or automatic extension',flush=True)
        break
