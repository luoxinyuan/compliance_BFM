"""One GPU job: independent frozen 1/2/4-worker physics sampling, no training."""
import fcntl,gzip,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent;P=S.parents[1]
NAME='frl-parallel-physics-01';side=S/'jobs'/(NAME+'.json');assert not side.exists()
previous=[json.loads(p.read_text()) for p in (S/'jobs').glob('*.json')]
assert not any(x.get('status')=='running' for x in previous)
spent=1800+sum(x.get('elapsed_s',x.get('timeout_s',0)) for x in previous if x['category']=='F-RL-v2')
limit=min(1800,172800-spent-1800);assert limit>300
lock=(P/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(S).free>100*(1<<30)
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.used','--format=csv,noheader'],text=True);uid=gpu.split(',')[1].strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'RTX PRO 5000' in gpu and not any(uid in x for x in apps.splitlines())
assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
def cpu_snapshot():
 allowed=os.sched_getaffinity(0);result={}
 for line in Path('/proc/stat').read_text().splitlines():
  fields=line.split()
  if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
   index=int(fields[0][3:])
   if index in allowed:result[index]=sum(int(v) for v in fields[4:6])
 return result
before=cpu_snapshot();resource_started=time.perf_counter();time.sleep(1);after=cpu_snapshot()
idle_cores=sum(after[k]-v for k,v in before.items())/os.sysconf('SC_CLK_TCK')/(time.perf_counter()-resource_started)
quota=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text());period=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text());cores=min(len(os.sched_getaffinity(0)),quota/period if quota>0 else 1e9)
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
ram_free=mem['MemAvailable']
cg=Path('/sys/fs/cgroup/memory')
if (cg/'memory.limit_in_bytes').exists():ram_free=min(ram_free,int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text()))
resource=dict(effective_CPU_cores=cores,estimated_idle_allowed_cores=idle_cores,available_RAM_bytes=ram_free,CPU_PSI=Path('/proc/pressure/cpu').read_text(),IO_PSI=Path('/proc/pressure/io').read_text())
assert cores>=6 and idle_cores>=6 and ram_free>=48*(1<<30),('Insufficient independent CPU/RAM capacity; no extra workers launched',resource)
out=S/'runs'/NAME;out.mkdir(exist_ok=False)
started=time.time();row=dict(name=NAME,category='F-RL-v2',kind='frozen 1/2/4-worker sampling benchmark',started=started,status='running',timeout_s=limit,supervisor_pid=os.getpid(),formal_training=False,resources=resource,runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),GPU=gpu)
child=None

def save():
 t=side.with_suffix('.tmp');t.write_text(json.dumps(row,indent=2));t.replace(side)
def archives(folder):
 files={}
 for path in sorted(folder.rglob('*.json.gz')):
  assert path.name not in files,'Duplicate episode file name across workers'
  files[path.name]=hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
 return files
save();results=[];reference=None
original=S/'runs/frl-efficiency-physics-01/legacy'
original_archives=None
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT)+':'+str(ROOT/'research_notes'))
try:
 original_archives=archives(original) if original.exists() else None
 assert original_archives and len(original_archives)==8,'Original complete four-episode/eight-archive physics reference required'
 original_result=json.loads((original/'efficiency-result.json').read_text())
 for workers in (1,2,4):
  assert limit-(time.time()-started)>60,'No launch after budget exhaustion'
  target=out/f'workers-{workers}'
  cmd=[str(P/'.venv/bin/python'),str(ROOT/'research_notes/frl_parallel_physics.py'),'--role','broker','--workers',str(workers),'--output',str(target)]
  with (out/f'workers-{workers}.log').open('x') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row.update(pid=child.pid,command=cmd);save()
   while child.poll() is None:
    remaining=limit-(time.time()-started)
    if remaining<=0:raise RuntimeError('Hard parallel benchmark budget exhausted')
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<2*(1<<30),'2GiB benchmark output guard'
    row.setdefault('gpu_samples',[]).append(dict(workers=workers,elapsed_s=time.time()-started,values=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True).strip()));save()
    try:child.wait(timeout=min(10,remaining))
    except subprocess.TimeoutExpired:pass
   assert child.returncode==0,('Parallel prototype failed',workers,child.returncode)
  result=json.loads((target/'parallel-result.json').read_text());actual=archives(target)
  assert actual==original_archives,('Complete physics/history/action archive differs from original reference',workers)
  if reference is None:
   assert all(result['counts'][k]==v for k,v in original_result['counts'].items()),'Actual original physical/decision counts differ'
   reference=result
  assert result['counts']==reference['counts'],'Actual sampling/physics counts differ'
  result['sampling_speedup_vs_one_worker']=reference['sampling_s']/result['sampling_s'];result['complete_archive_bytes_exact_vs_original']=True
  results.append(result);(out/'comparison.json').write_text(json.dumps(results,indent=2))
 row.update(status='completed',returncode=0)
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 row.update(status='failed',returncode=1,error=repr(exc));raise
finally:row['elapsed_s']=time.time()-started;save()
