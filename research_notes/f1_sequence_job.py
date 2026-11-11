"""One owned, bounded L1 OR L2 job. No queue and no L3 implementation."""
import argparse,fcntl,json,os,signal,subprocess,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--layer',choices=['L1','L2'],required=True);p.add_argument('--name',required=True);p.add_argument('--seconds',type=int,required=True);a=p.parse_args()
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/f1-l1-l2-20261111';R=S/'runtime';start=time.monotonic()
row=dict(layer=a.layer,pid=os.getpid(),started=time.time(),status='preflight',timeout_s=a.seconds,training=False)
f=S/'jobs'/(a.name+'.json');assert not f.exists()
def save():
 q=f.with_suffix('.tmp');q.write_text(json.dumps(row,indent=2));q.replace(f)
save()
try:
 previous=[json.loads(x.read_text()) for x in (S/'jobs').glob('*.json') if x!=f]
 spent=sum(x.get('elapsed_s',0) for x in previous);layer=sum(x.get('elapsed_s',0) for x in previous if x.get('layer')==a.layer)
 assert layer+a.seconds+20<={'L1':10800,'L2':14400}[a.layer]
 assert 3346.9409750220075+1200+spent+a.seconds+20<=32400
 if a.layer=='L2':assert any(x.get('layer')=='L1' and x.get('status')=='completed' for x in previous),'L1 incomplete'
 with (P/'.dual-track-gpu0.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  free=shutil.disk_usage(P).free;assert free>115*2**30
  gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();assert gpu.startswith('GPU-3ae548cd-2131-2b5d-42ad-5d0f40ff9da0, 0 MiB, 0 %'),gpu
  runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HOME=str(P/'.cache/huggingface'),PIP_CACHE_DIR=str(P/'.cache/pip'))
  cmd=[str(P/'.venv/bin/python'),str(R/'research_notes/f1_sequence_stage.py'),'--layer',a.layer,'--output',str(S/'runs'/a.name)]
  with (S/'jobs'/(a.name+'.log')).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   row.update(status='running',child_pid=proc.pid,runtime=runtime,command=cmd,gpu=gpu,free_bytes=free,prior_charged_s=3346.9409750220075+1200+spent);save()
   try:code=proc.wait(timeout=a.seconds)
   except subprocess.TimeoutExpired:
    row['timeout']=True;os.killpg(proc.pid,signal.SIGTERM)
    try:code=proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   row.update(returncode=code,status='completed' if code==0 else 'failed_preserved')
except BaseException as e:row.update(status='blocked_no_launch',error=repr(e))
finally:row['elapsed_s']=time.monotonic()-start;save()
