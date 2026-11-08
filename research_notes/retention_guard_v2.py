"""Owned bounded subprocess; per-job ledger supports concurrent independent workflows."""
import os,sys,time,json,subprocess,signal
from pathlib import Path
s=Path(__file__).resolve().parent;work,name,limit=sys.argv[1:4];limit=float(limit);cmd=sys.argv[4:];bud={'A0':3000,'A1':6000,'B':1800,'C':1800};log=s/'jobs'/f'{name}.json';assert not log.exists()
def write(x):p=log.with_suffix('.tmp');p.write_text(json.dumps(x,indent=2));p.replace(log)
def used(w):
 total=0
 for f in (s/'jobs').glob('*.json'):
  d=json.loads(f.read_text())
  if d['workflow']==w and d['name']!=name:total+=d.get('elapsed_s',time.time()-d['started'])
 return total
assert used(work)+limit<=bud[work]
f=(s/'logs'/f'{name}.log').open('w');start=time.time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0')
p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);j=dict(workflow=work,name=name,started=start,pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),command=cmd,status='running',limit_s=limit);write(j)
reason=None
while p.poll() is None:
 time.sleep(2)
 size=sum(x.stat().st_size for x in (s/'runs').rglob('*') if x.is_file())
 if time.time()-start>limit or used(work)+time.time()-start>bud[work] or size>6*2**30:
  reason='budget';os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  break
j.update(elapsed_s=time.time()-start,returncode=p.returncode,status='completed' if p.returncode==0 else 'failed',stop_reason=reason);write(j);f.close()
