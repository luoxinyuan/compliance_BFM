"""Issue59 exact-owned process supervisor. SharedGPU0 lock; no background polling service."""
import argparse,fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--category',choices=['P1','P2'],required=True);p.add_argument('--script',required=True);p.add_argument('--timeout',type=int);p.add_argument('extra',nargs=argparse.REMAINDER);a=p.parse_args()
R=Path(__file__).resolve().parents[1];S=R.parent;P=S.parents[1];side=S/'jobs'/(a.name+'.json');assert not side.exists()
lock=(P/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prior=[json.loads(v.read_text()) for v in (S/'jobs').glob('*.json')];assert not any(v.get('status')=='running' for v in prior)
spent=(1800 if a.category=='P1' else 1200)+sum(v.get('elapsed_s',v.get('timeout_s',0)) for v in prior if v.get('category')==a.category)
assert shutil.disk_usage(S).free>100*(1<<30)
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.used','--format=csv,noheader'],text=True);uuid=gpu.split(',')[1].strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'RTX PRO 5000' in gpu and not any(uuid in v for v in apps.splitlines());assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
extra=a.extra[1:] if a.extra[:1]==['--'] else a.extra
cmd=[str(P/'.venv/bin/python'),str(R/'research_notes'/a.script),*extra];env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(R)+':'+str(R/'research_notes'))
start=time.time();limit=None if a.category=='P1' else min(a.timeout or 86400,86400-spent-180);assert limit is None or limit>60
row=dict(name=a.name,category=a.category,started=start,expected_s=14400 if a.category=='P1' else None,timeout_s=limit,status='running',supervisor_pid=os.getpid(),command=cmd,runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),gpu=gpu,training=('--mode' in extra and extra[extra.index('--mode')+1]=='high_train'))
def write():
    temp=side.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2));temp.replace(side)
write()
try:
    with (S/'jobs'/(a.name+'.log')).open('w') as log:
        c=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=c.pid;write()
        while c.poll() is None:
            time.sleep(2);elapsed=time.time()-start
            if a.category=='P1' and elapsed+spent>14400 and not row.get('expected_budget_exceeded'):
                row.update(expected_budget_exceeded=True,report_required=True,overrun_rule='report and continue finite preset work as authorized');write()
            if limit is not None and elapsed>limit:
                row['budget_timeout']=True;row['report_required']=True;write()
                # Cooperative normal stop first, then kill ONLY this owned process group if unresponsive.
                for i,v in enumerate(cmd):
                    if v=='--output':
                        out=Path(cmd[i+1]);out.mkdir(parents=True,exist_ok=True);(out/'STOP_REQUEST.json').write_text(json.dumps(dict(reason='hard24h budget',requested=time.time())))
                try:c.wait(90)
                except subprocess.TimeoutExpired:
                    os.killpg(c.pid,signal.SIGTERM)
                    try:c.wait(30)
                    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
        row.update(status='completed' if c.returncode==0 else 'failed',returncode=c.returncode)
finally:row['elapsed_s']=time.time()-start;write()
print(json.dumps(row))
