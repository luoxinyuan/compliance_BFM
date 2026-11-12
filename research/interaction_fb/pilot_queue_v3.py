"""Single-GPU approved fixed queue. No retries, threshold tuning or sealed reads."""
import os,sys,time,json,subprocess,signal,shutil,hashlib
from pathlib import Path
from supervision import tree_bytes, stop_owned_group
S=Path(sys.argv[1]).resolve();R=S/'runtime';P=S.parents[1];PY=P/'.venv/bin/python'
L=R/'research_notes/interaction_pilot_lock_v3_approved.json';lock=json.loads(L.read_text())
assert lock['status']=='approved_by_Yi'
J=S/'jobs';J.mkdir(exist_ok=True);(S/'runs').mkdir(exist_ok=True);(S/'tmp').mkdir(exist_ok=True)
PREP=P/'experiments/interaction-fb-pilot-lock-20261112'
assert lock['acceptance']=='explicit_Yi_acceptance_despite_single_window_peak_screen'
CHARGED=lock['inherited_pilot_job_upper_s']+1200 # bounded CPU preparation/publication reserve
CAP=lock['total_job_seconds'];OUTCAP=lock['total_output_bytes']
assert CAP==54000 and OUTCAP==8*2**30
runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain'])
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(S/'tmp'))
def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def size():
 return tree_bytes([stage/name for stage in (S,PREP,P/'experiments/interaction-fb-pilot-20261112',P/'experiments/interaction-pilot-repair-20261112',P/'experiments/interaction-pilot-noise-20261112') for name in ('runs','jobs','prep','tmp')])
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
 start=time.time();cleanup_deadline=time.monotonic()+max(0,CAP-spent()-10)
 side=J/('job-'+name+'.json');assert not side.exists()
 with (J/(name+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  row=dict(name=name,pid=p.pid,supervisor=os.getpid(),runtime=runtime,start=start,command=cmd,approval=lock['approval_reference'])
  signaled=False
  try:
   write(side,row)
   while p.poll() is None:
    elapsed=time.time()-start;used=size();free=shutil.disk_usage(S).free
    stop=spent()+elapsed>=CAP-reserve-120 or used>=OUTCAP-1200*2**20 or free<100*2**30
    if stop and not signaled:
     os.kill(p.pid,signal.SIGTERM);signaled=True;row['stop_requested_at']=time.time();row['stop_reason']='cumulative_budget_or_output_or_disk_guard';write(side,row)
    if spent()+elapsed>=CAP-10 or used>=OUTCAP-64*2**20 or (signaled and time.time()-row['stop_requested_at']>90):
     os.kill(p.pid,signal.SIGKILL);row['hard_stop']=True;break
    time.sleep(5)
  except BaseException as exc:
   row['supervisor_error']=repr(exc)
   raise
  finally:
   # IO/ledger/scanner exceptions must never strand the owned training child.
   rc=stop_owned_group(p,grace=min(90,max(0,cleanup_deadline-time.monotonic())))
   row.update(exit_code=rc,end=time.time(),wall_s=time.time()-start)
   try:row['output_bytes']=size()
   except OSError as exc:row['output_measurement_error']=repr(exc)
   write(side,row)
 assert rc==0 and not signaled,(name,'failed/stopped; no next job')
 if training:
  m=json.loads((S/'runs'/name/'checkpoints/round-128/manifest.json').read_text());assert m['complete'] and m['round_id']==128
  assert m['runtime']==runtime
 write(S/'ledger.json',dict(status='running',charged_job_seconds=spent(),output_bytes=size(),runtime=runtime,approval=lock['approval_reference']))
def supervisor_stop(signum,frame):raise KeyboardInterrupt('explicit supervisor stop')
signal.signal(signal.SIGTERM,supervisor_stop)
signal.signal(signal.SIGINT,supervisor_stop)
M=R/'research_notes/interaction_pilot_manifest_v2'
try:
 # A past orphan must be reconciled before another queue may charge the ledger.
 for prior in J.glob('job-*.json'):
  record=json.loads(prior.read_text())
  assert all(k in record for k in ('wall_s','exit_code')), ('Unreconciled prior job; refuse launch',str(prior))
 for arm in 'ABC':run('pilot-'+arm+'-v3',['--mode','pilot','--arm',arm,'--approval-lock',str(L),'--manifest',str(M/'train.json')],True)
 for arm in 'ABC':run('dev-'+arm,['--mode','eval','--arm',arm,'--model',str(S/'runs'/('pilot-'+arm+'-v3')/'checkpoints/round-128'),'--manifest',str(M/'dev.json')])
 run('dev-initial',['--mode','eval','--arm','A','--model',str(S/'runs/pilot-A-v3/checkpoints/round-0'),'--manifest',str(M/'dev.json')])
 run('dev-BC500',['--mode','eval','--arm','A','--reference-bc500','--manifest',str(M/'dev.json')])
 # CPU-only grouped first read, no policy or representation updates.
 start=time.time()
 probe_rc=None
 try:
  completed=subprocess.run([str(PY),str(R/'research/interaction_fb/probe.py'),str(S/'runs/dev-C/probe-features.json.gz'),str(S/'probe-first-read.json')],env=env,cwd=R,timeout=min(300,max(1,int(CAP-spent()-30))))
  probe_rc=completed.returncode
  assert probe_rc==0,'CPU first read failed'
 finally:
  write(J/'job-probe.json',dict(name='probe',runtime=runtime,wall_s=time.time()-start,exit_code=probe_rc))
 write(S/'ledger.json',dict(status='completed_needs_scientific_report',charged_job_seconds=spent(),output_bytes=size(),runtime=runtime,sealed_executed=False))
except BaseException as exc:
 record=dict(status='stopped',reason=repr(exc),charged_job_seconds=spent(),runtime=runtime,sealed_executed=False)
 try:record['output_bytes']=size()
 except OSError as measurement:record['output_measurement_error']=repr(measurement)
 write(S/'ledger.json',record);raise
