"""Single owned L0 timing job; no queue, training or automatic retry."""
import fcntl,json,os,signal,subprocess,time,shutil
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/f1-l0-20261111';R=S/'runtime'
t=time.monotonic();row={'stage':'L0','budget_s':1800,'preparation_reserve_s':900,'postprocess_reserve_s':100,'job_timeout_s':800,'pid':os.getpid(),'started':time.time(),'training':False,'status':'preflight'}
def save():
 p=S/'jobs/l0-01.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
save()
try:
 with (P/'.dual-track-gpu0.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  free=shutil.disk_usage(P).free;assert free>=102*2**30,'100GiB guard plus 2GiB write reserve'
  g=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();assert g.startswith('GPU-3ae548cd-2131-2b5d-42ad-5d0f40ff9da0, 0 MiB, 0 %'),g
  runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HOME=str(P/'.cache/huggingface'),PIP_CACHE_DIR=str(P/'.cache/pip'))
  cmd=[str(P/'.venv/bin/python'),str(R/'research_notes/f1_l0.py'),'--output',str(S/'runs/l0-01')]
  with (S/'jobs/l0-01.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   row.update(status='running',child_pid=proc.pid,runtime=runtime,command=cmd,gpu=g,free_bytes=free);save()
   try:code=proc.wait(timeout=800)
   except subprocess.TimeoutExpired:
    row['timeout']=True;os.killpg(proc.pid,signal.SIGTERM)
    try:code=proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   row.update(returncode=code,status='completed' if code==0 else 'failed_no_automatic_retry')
except BaseException as e:row.update(status='blocked_no_launch',error=repr(e))
finally:row['elapsed_s']=time.monotonic()-t;row['charged_seconds_including_reserves']=row['elapsed_s']+1000;save()
