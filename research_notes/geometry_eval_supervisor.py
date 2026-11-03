"""Finite independent Expert runs; reservation-based shared stage budget, no watch loop."""
import argparse,datetime,fcntl,json,os,resource,signal,subprocess,time
from pathlib import Path
from contextlib import contextmanager
from causal_supervisor import processes
from m7_io import write
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');ROOT=Path(__file__).resolve().parents[1]
RUN=P/'bridge-v2-owned/runs/contact-geometry-20261103'
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def snapshot(index):
 fields=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
 return dict(uuid=fields[0],used_mib=int(fields[1]),free_mib=int(fields[2]),utilization=int(fields[3]))
@contextmanager
def ledger():
 with (RUN/'compute.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  path=RUN/'compute-ledger.json';data=json.loads(path.read_text()) if path.exists() else dict(attempts=[])
  yield data
  write(path,data)
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['A','B','C'],required=True);p.add_argument('--kind',choices=['structured','direct'],required=True);p.add_argument('--stage',choices=['prepare','train','final'],required=True);p.add_argument('--gpu-index',type=int,required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--iterations',type=int,default=20);p.add_argument('--seconds',type=int,default=2400);p.add_argument('--resume');a=p.parse_args()
 assert not git('status','--porcelain');assert git('branch','--show-current').startswith('gaoe05/contact-geometry')
 plan=json.loads((ROOT/'research_notes/geometry_protocol.json').read_text());RUN.mkdir(exist_ok=True)
 before=snapshot(a.gpu_index);assert before['used_mib']<100 and before['utilization']<5, 'Use an actually idle GPU'
 offset={'A':8,'B':24,'C':40}[a.arm];affinity=sorted(os.sched_getaffinity(0))[offset:offset+8]
 reservation=a.seconds;decisions=a.iterations*768 if a.stage=='train' else 0
 with ledger() as d:
  used=sum(r.get('wall_seconds',r['reserved_seconds']) for r in d['attempts'])
  assert used+reservation<=plan['resources']['sum_supervised_process_wall_seconds']
  assert sum(r.get('result',{}).get('decisions',r.get('reserved_decisions',0)) for r in d['attempts'])+decisions<=plan['training']['max_total_rl_decisions']
  active=[r for r in d['attempts'] if r['status']=='running'];assert len(active)<3
  assert not any(r['gpu']==a.gpu_index or set(r['affinity'])&set(affinity) for r in active)
  index=len(d['attempts']);path=RUN/f'{a.arm}-{a.stage}-{index+1:02d}';path.mkdir()
  reserved_updates=0
  assert sum(r.get('actual_updates',r.get('reserved_updates',0)) for r in d['attempts'])+reserved_updates<=8000
  row=dict(reserved_updates=reserved_updates,kind=a.kind,arm=a.arm,stage=a.stage,status='running',supervisor=os.getpid(),gpu=a.gpu_index,gpu_before=before,affinity=affinity,source=git('rev-parse','HEAD'),path=str(path),reserved_seconds=reservation,reserved_decisions=decisions,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());d['attempts'].append(row)
 env=dict(os.environ);env.update(PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=before['uuid'],MUJOCO_GL='disable',BFM_STAGE_SUPERVISED='1',BFM_STAGE_DEADLINE_MONOTONIC=str(time.monotonic()+reservation),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+reservation),TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',WANDB_MODE='disabled')
 cache=path/'cache';cache.mkdir()
 for k in ['TMPDIR','TEMP','TMP','XDG_CACHE_HOME','MPLCONFIGDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','HF_HOME','PYTHONPYCACHEPREFIX']:
  dest=cache/k;dest.mkdir();env[k]=str(dest)
 command=[str(P/'.venv/bin/python'),'-u',str(ROOT/('research_notes/geometry_train.py' if a.stage=='train' else 'research_notes/geometry_native.py' if a.arm=='C' and a.stage=='prepare' else 'research_notes/geometry_stage.py')),'--kind',a.kind,'--stage',a.stage,'--gpu',before['uuid'],'--output',str(path/'job'),'--iterations',str(a.iterations),'--seed',str(a.seed),'--max-seconds',str(max(60,reservation-120))]
 env['GEOMETRY_ARM']=a.arm
 if a.stage=='train':command+=['--arm','B1']
 if a.resume:command+=['--resume',a.resume]
 row['command']=command
 def limits():
  os.sched_setaffinity(0,affinity);os.nice(10);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 child=None;boundary=None;start=time.monotonic();peakrss=peakgpu=0
 try:
  with (path/'stdout.log').open('wb') as out,(path/'stderr.log').open('wb') as err:
   child=subprocess.Popen(command,cwd=path,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=limits);row['pid']=child.pid
   with ledger() as d:d['attempts'][index]=row.copy()
   nextpoll=nextsize=0
   while child.poll() is None:
    ps=processes(child.pid);rss=sum(x['rss_bytes'] for x in ps);peakrss=max(peakrss,rss)
    if rss>32*1024**3:boundary='owned_RSS_over_32GiB'
    if time.monotonic()-start>reservation:boundary='reserved_wall_limit'
    if time.monotonic()>nextpoll:
     nextpoll=time.monotonic()+3;apps=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
     ids={x['pid'] for x in ps};usage=0
     for line in apps.splitlines():
      pid,mem=line.split(', ')
      if int(pid) in ids:usage+=int(mem)
     peakgpu=max(peakgpu,usage)
     if usage>8192:boundary='owned_VRAM_over_8GiB'
     row.update(peak_rss_bytes=peakrss,peak_gpu_mib=peakgpu,elapsed_seconds=time.monotonic()-start)
     write(path/'resources.json',row)
    if time.monotonic()>nextsize:
     nextsize=time.monotonic()+30
     if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>plan['resources']['stage_output_gib']*1024**3:boundary='stage_output_limit'
    if boundary:break
    time.sleep(.5)
 finally:
  if child:
   if processes(child.pid):
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   else:child.wait()
   row.update(exit_code=child.returncode,remaining_processes=processes(child.pid))
  row['actual_updates']=sum(json.loads(f.read_text())['updates'] for f in (path/'job').glob('*-training.json') if json.loads(f.read_text()).get('role') is None)
  row.update(status='finished',wall_seconds=time.monotonic()-start,resource_boundary=boundary,peak_rss_bytes=peakrss,peak_gpu_mib=peakgpu,gpu_after=snapshot(a.gpu_index))
  if (path/'job/result.json').exists():row['result']=json.loads((path/'job/result.json').read_text())
  with ledger() as d:d['attempts'][index]=row
  write(path/'resources.json',row);print(json.dumps(row,indent=2))
if __name__=='__main__':main()
