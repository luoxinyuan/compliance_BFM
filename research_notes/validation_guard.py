"""Bounded owned evaluation job; concurrent ledgers, no training or daemon queue."""
from pathlib import Path
import os,sys,time,json,subprocess,signal
s=Path(__file__).resolve().parent;name=sys.argv[1];limit=float(sys.argv[2]);cmd=sys.argv[3:];p=s/'jobs'/f'{name}.json';assert not p.exists()
def write(j):q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2));q.replace(p)
def used():
 total=0
 for f in (s/'jobs').glob('*.json'):
  if f==p:continue
  j=json.loads(f.read_text());total+=j.get('elapsed_s',time.time()-j['started'])
 return total
start=time.time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0');f=(s/'logs'/f'{name}.log').open('w');c=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
j=dict(name=name,pid=c.pid,start_ticks=Path(f'/proc/{c.pid}/stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started=start,limit_s=limit,command=cmd,status='running');write(j);reason=None
while c.poll() is None:
 time.sleep(2);size=sum(x.stat().st_size for x in (s/'runs').rglob('*') if x.is_file())
 if time.time()-start>limit or used()+time.time()-start>1650 or size>2*2**30:
  reason='budget';os.killpg(c.pid,signal.SIGTERM)
  try:c.wait(10)
  except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
  break
j.update(elapsed_s=time.time()-start,status='completed' if c.returncode==0 else 'failed',returncode=c.returncode,stop_reason=reason);write(j);f.close()
