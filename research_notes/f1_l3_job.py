"""Bounded owned L3 process; additive authorization 6801908284."""
import fcntl,json,os,signal,subprocess,time,shutil
from pathlib import Path
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/f1-l1-l2-20261111';R=S/'runtime-l3'
start=time.monotonic();name='l3-01';f=S/'jobs'/f'{name}.json';assert not f.exists()
row=dict(layer='L3',pid=os.getpid(),started=time.time(),status='preflight',timeout_s=3600,preparation_reporting_reserved_s=1200,stage_hard_s=5400,prior_F1_charged_s=11372.572671389,training=False)
def save():
 q=f.with_suffix('.tmp');q.write_text(json.dumps(row,indent=2));q.replace(f)
save()
try:
 with (P/'.dual-track-gpu0.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  free=shutil.disk_usage(P).free;assert free>105*2**30
  gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();assert gpu.startswith('GPU-3ae548cd-2131-2b5d-42ad-5d0f40ff9da0, 0 MiB, 0 %'),gpu
  runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HOME=str(P/'.cache/huggingface'),PIP_CACHE_DIR=str(P/'.cache/pip'))
  cmd=[str(P/'.venv/bin/python'),str(R/'research_notes/f1_l3_stage.py'),'--layer','L3','--output',str(S/'runs'/name)]
  with (S/'jobs'/f'{name}.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   row.update(status='running',child_pid=proc.pid,runtime=runtime,command=cmd,gpu=gpu,free_bytes=free);save()
   try:code=proc.wait(timeout=3600)
   except subprocess.TimeoutExpired:
    row['timeout']=True;os.killpg(proc.pid,signal.SIGTERM)
    try:code=proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   row.update(returncode=code,status='completed' if code==0 else 'failed_preserved')
except BaseException as e:row.update(status='blocked_no_launch',error=repr(e))
finally:row['elapsed_s']=time.monotonic()-start;save()
