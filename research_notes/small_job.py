"""Bounded engineering-only job supervisor with immutable PID/cost sidecar."""
import argparse,fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('mode');p.add_argument('--reference');a=p.parse_args()
S=Path(__file__).resolve().parents[2];R=S/'runtime';P=S.parents[1];py=P/'.venv/bin/python'
side=S/'jobs'/(a.name+'.json');assert not side.exists()
lock=(P/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=[json.loads(v.read_text()) for v in (S/'jobs').glob('*.json')]
assert not any(v.get('status')=='running' for v in previous)
assert sum(v.get('elapsed_s',v['timeout_s']) for v in previous)+1500<=7200
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.used','--format=csv,noheader'],text=True)
uuid=gpu.split(',')[1].strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'RTX PRO 5000' in gpu and not any(uuid in v for v in apps.splitlines())
assert shutil.disk_usage(S).free>32*(1<<30)
assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
out=S/'runs'/a.name
cmd=[str(py),str(R/'research_notes/small_stage.py'),'--mode',a.mode,'--output',str(out)]
if a.reference:cmd+=['--reference',a.reference]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(R)+':'+str(R/'research_notes'))
start=time.time();row=dict(name=a.name,mode=a.mode,started=start,timeout_s=1500,status='running',supervisor_pid=os.getpid(),command=cmd,runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),gpu=gpu)
def write():side.write_text(json.dumps(row,indent=2))
write()
try:
 with (S/'jobs'/(a.name+'.log')).open('w') as log:
  c=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=c.pid;write()
  try:rc=c.wait(timeout=1500)
  except subprocess.TimeoutExpired:
   os.killpg(c.pid,signal.SIGTERM)
   try:rc=c.wait(30)
   except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);rc=c.wait()
  row.update(status='completed' if rc==0 else 'failed',returncode=rc)
finally:row['elapsed_s']=time.time()-start;write()
print(json.dumps(row))
