"""Single-GPU approved fixed queue. No retries, threshold tuning or sealed reads."""
import os,sys,time,json,subprocess,signal,shutil,hashlib
from pathlib import Path
S=Path(sys.argv[1]).resolve();R=S/'runtime';P=S.parents[1];PY=P/'.venv/bin/python'
L=R/'research_notes/interaction_pilot_lock_v2_approved.json';lock=json.loads(L.read_text())
assert lock['status']=='approved_by_Yi'
J=S/'jobs';J.mkdir(exist_ok=True);(S/'runs').mkdir(exist_ok=True);(S/'tmp').mkdir(exist_ok=True)
PREP=P/'experiments/interaction-fb-pilot-lock-20261112'
CHARGED=2560.4250841140747+600 # current launch/report/CPU preparation reserved, not excluded
CAP=28800;OUTCAP=8*2**30
runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(S/'tmp'))
def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def size():
 seen=set();total=0
 for stage in (S,PREP):
  for name in ('runs','jobs','prep','tmp'):
   for f in (stage/name).rglob('*'):
    if f.is_file() and not f.is_symlink():
     st=f.stat();k=(st.st_dev,st.st_ino)
     if k not in seen:seen.add(k);total+=st.st_size
 return total
def spent():return CHARGED+sum(json.loads(f.read_text()).get('wall_s',0) for f in J.glob('job-*.json'))
def gpu_free():
 uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(line.split(',')[0].strip()==uuid for line in apps.splitlines()),'GPU0 occupied; queue stops without touching owner'
def run(name,args,training=False):
 gpu_free();assert shutil.disk_usage(S).free>108*2**30
 reserve=7300 if training else 300
 assert spent()<CAP-reserve-180 and size()<OUTCAP-1200*2**20
 cmd=[str(PY),'-u',str(R/'research/interaction_fb/benchmark.py'),'--output',str(S/'runs'/name),*args]
 start=time.time();side=J/('job-'+name+'.json');assert not side.exists()
 with (J/(name+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  row=dict(name=name,pid=p.pid,supervisor=os.getpid(),runtime=runtime,start=start,command=cmd,approval=lock['approval_reference'])
  write(side,row);signaled=False
  while p.poll() is None:
   elapsed=time.time()-start;used=size();free=shutil.disk_usage(S).free
   stop=spent()+elapsed>=CAP-reserve-120 or used>=OUTCAP-1200*2**20 or free<100*2**30
   if stop and not signaled:
    os.kill(p.pid,signal.SIGTERM);signaled=True;row['stop_requested_at']=time.time();row['stop_reason']='cumulative_budget_or_output_or_disk_guard';write(side,row)
   if spent()+elapsed>=CAP-10 or used>=OUTCAP-64*2**20 or (signaled and time.time()-row['stop_requested_at']>90):
    os.kill(p.pid,signal.SIGKILL);row['hard_stop']=True;break
   time.sleep(5)
  rc=p.wait();row.update(exit_code=rc,end=time.time(),wall_s=time.time()-start,output_bytes=size());write(side,row)
 assert rc==0 and not signaled,(name,'failed/stopped; no next job')
 if training:
  m=json.loads((S/'runs'/name/'checkpoints/round-128/manifest.json').read_text());assert m['complete'] and m['round_id']==128
 write(S/'ledger.json',dict(status='running',charged_job_seconds=spent(),output_bytes=size(),runtime=runtime,approval=lock['approval_reference']))
M=R/'research_notes/interaction_pilot_manifest_v2'
try:
 for arm in 'ABC':run('pilot-'+arm+'-r1',['--mode','pilot','--arm',arm,'--approval-lock',str(L),'--manifest',str(M/'train.json')],True)
 for arm in 'ABC':run('dev-'+arm,['--mode','eval','--arm',arm,'--model',str(S/'runs'/('pilot-'+arm+'-r1')/'checkpoints/round-128'),'--manifest',str(M/'dev.json')])
 run('dev-initial',['--mode','eval','--arm','A','--model',str(S/'runs/pilot-A-r1/checkpoints/round-0'),'--manifest',str(M/'dev.json')])
 run('dev-BC500',['--mode','eval','--arm','A','--reference-bc500','--manifest',str(M/'dev.json')])
 # CPU-only grouped first read, no policy or representation updates.
 start=time.time()
 subprocess.run([str(PY),str(R/'research/interaction_fb/probe.py'),str(S/'runs/dev-C/probe-features.json.gz'),str(S/'probe-first-read.json')],env=env,cwd=R,check=True,timeout=min(300,max(1,int(CAP-spent()-30))))
 write(J/'job-probe.json',dict(name='probe',runtime=runtime,wall_s=time.time()-start,exit_code=0))
 write(S/'ledger.json',dict(status='completed_needs_scientific_report',charged_job_seconds=spent(),output_bytes=size(),runtime=runtime,sealed_executed=False))
except BaseException as exc:
 write(S/'ledger.json',dict(status='stopped',reason=repr(exc),charged_job_seconds=spent(),output_bytes=size(),runtime=runtime,sealed_executed=False));raise
