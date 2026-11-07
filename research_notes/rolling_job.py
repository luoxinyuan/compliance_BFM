"""Bounded stage-only owned process supervisor; survives SSH, no daemon/watch service."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
S=Path(__file__).resolve().parents[2];ledger=S/'ledger.json'
name,script,limit=sys.argv[1:4];script=Path(script);assert script.is_relative_to(S)
l=json.loads(ledger.read_text());spent=l.get('prior_known_spent_s',0)+l.get('prior_interrupted_qualification_max_s',0)+sum(j.get('elapsed_s',0) for j in l['jobs']);limit=min(float(limit),18000-spent);assert limit>0
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'no independently idle GPU'
mi={x.split(':')[0]:int(x.split(':')[1].split()[0])*1024 for x in Path('/proc/meminfo').read_text().splitlines()};assert mi['MemAvailable']>64*1024**3
size=sum(p.stat().st_size for p in (S/'runs').rglob('*') if p.is_file());assert size<9*1024**3
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');py=str(P/'.venv/bin/python');repo=S/'repo'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',CLEAN_KIND='direct',PYTHONPATH=str(repo)+':'+str(repo/'research_notes'))
with (S/'logs'/f'{name}.log').open('w') as log:
 p=subprocess.Popen([py,'-u',str(script)]+sys.argv[4:],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 j=dict(name=name,pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=time.time(),status='running',max_seconds=limit,runtime=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),output_bytes_before=size)
 l['jobs'].append(j);ledger.write_text(json.dumps(l,indent=2));print(json.dumps(j),flush=True)
 try:rc=p.wait(timeout=limit)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:rc=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
 j.update(status='completed' if rc==0 else 'failed',returncode=rc,elapsed_s=time.time()-j['started']);l=json.loads(ledger.read_text());l['jobs'][-1]=j;ledger.write_text(json.dumps(l,indent=2));print(json.dumps(j),flush=True)
 sys.exit(0 if rc==0 else 1)
