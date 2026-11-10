"""Single GPU0 Issue58 F-RL-v2 supervisor; child evaluation included once in wall."""
import argparse,fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--arm',choices=['A','B'],default='A')
p.add_argument('--smoke',action='store_true');p.add_argument('--reward-gate');a=p.parse_args()
R=Path(__file__).resolve().parents[1];S=R.parent;P=S.parents[1]
side=S/'jobs'/(a.name+'.json');assert not side.exists()
lock=(P/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prior=[json.loads(v.read_text()) for v in (S/'jobs').glob('*.json')]
assert not any(v.get('status')=='running' for v in prior)
spent=900+sum(v.get('elapsed_s',v['timeout_s']) for v in prior if v['category']=='P0-B')
timeout=min(42000,43200-spent-600);assert timeout>60
assert shutil.disk_usage(S).free>100*(1<<30)
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.used','--format=csv,noheader'],text=True)
uuid=gpu.split(',')[1].strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'RTX PRO 5000' in gpu and not any(uuid in v for v in apps.splitlines())
assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
cmd=[str(P/'.venv/bin/python'),str(R/'research_notes/fb_safe_stage.py'),'--output',str(S/'runs'/a.name)]
if a.smoke:cmd.append('--smoke')
if a.reward_gate:cmd+=['--reward-gate',a.reward_gate]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(R)+':'+str(R/'research_notes'))
start=time.time();row=dict(name=a.name,category='P0-B',started=start,timeout_s=timeout,status='running',supervisor_pid=os.getpid(),command=cmd,runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),gpu=gpu,formal_training=False,arm=a.arm)
def write():
 temp=side.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2));temp.replace(side)
write()
try:
 with (S/'jobs'/(a.name+'.log')).open('w') as log:
  c=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=c.pid;write()
  try:rc=c.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   # In-loop request saves the exact current batch checkpoint at next boundary.
   out=S/'runs'/a.name
   (out/'STOP_REQUEST.json').write_text(json.dumps(dict(reason='hard job budget',requested=time.time())))
   try:rc=c.wait(90)
   except subprocess.TimeoutExpired:
    os.killpg(c.pid,signal.SIGTERM)
    try:rc=c.wait(30)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);rc=c.wait()
   row['budget_timeout']=True
  row.update(status='completed' if rc==0 else 'failed',returncode=rc)
finally:row['elapsed_s']=time.time()-start;write()
print(json.dumps(row))
