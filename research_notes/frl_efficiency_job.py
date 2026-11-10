"""One finite serial GPU parity/profile job after the formal A/B pair; no training."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent;P=S.parents[1]
NAME='frl-efficiency-physics-01';side=S/'jobs'/(NAME+'.json');assert not side.exists()
rows=[json.loads(v.read_text()) for v in (S/'jobs').glob('*.json')]
assert not any(v.get('status')=='running' for v in rows)
spent=1200+sum(v.get('elapsed_s',v.get('timeout_s',0)) for v in rows if v['category']=='F-RL-v2')
for n in ('archive-benchmark-01','archive-benchmark-02'):
 f=S/'prep'/n/'job.json'
 if f.exists():spent+=json.loads(f.read_text())['elapsed_s']
spent+=600 # conservative CPU preparation allowance, not a measured duration
limit=min(900,172800-spent-1800);assert limit>300
lock=(P/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(S).free>100*(1<<30)
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.used','--format=csv,noheader'],text=True);uid=gpu.split(',')[1].strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'RTX PRO 5000' in gpu and not any(uid in x for x in apps.splitlines())
assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
out=S/'runs'/NAME;out.mkdir(exist_ok=False)
started=time.time();row=dict(name=NAME,category='F-RL-v2',kind='exposed-only acceleration qualification',status='running',started=started,timeout_s=limit,supervisor_pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),formal_training=False)
def save():
 t=side.with_suffix('.tmp');t.write_text(json.dumps(row,indent=2));t.replace(side)
save();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT)+':'+str(ROOT/'research_notes'))
try:
 for mode,profile in [('legacy',False),('fast',False),('legacy-profile',True)]:
  cmd=[str(P/'.venv/bin/python'),str(ROOT/'research_notes/frl_efficiency_physics.py'),'--output',str(out/mode),'--archive-mode','legacy' if mode.startswith('legacy') else 'fast']
  if profile:cmd+=['--profile']
  with (out/(mode+'.log')).open('x') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row.update(pid=child.pid,command=cmd);save()
   try:rc=child.wait(timeout=limit-(time.time()-started))
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise RuntimeError('Hard acceleration budget exhausted')
   assert rc==0,('Physical profiling subprocess failed',mode,rc)
 a=json.loads((out/'legacy/efficiency-result.json').read_text());b=json.loads((out/'fast/efficiency-result.json').read_text())
 assert a['counts']==b['counts'],'Physical/control counts changed'
 assert {k:v['uncompressed_sha256'] for k,v in a['archive_assets'].items()}=={k:v['uncompressed_sha256'] for k,v in b['archive_assets'].items()},'Physical/control/observation/history archive bytes differ'
 assert json.loads((out/'legacy/summaries.json').read_text())==json.loads((out/'fast/summaries.json').read_text()),'Physical metrics changed'
 result=dict(status='exact_physical_parity_passed',legacy_wall_s=a['evaluation_wall_s'],fast_wall_s=b['evaluation_wall_s'],measured_end_to_end_speedup=a['evaluation_wall_s']/b['evaluation_wall_s'],counts=a['counts'],all_physics_actions_qPD_history_features_archives_exact=True,RL_decisions=0,agent_updates=0,adopted_into_formal_AB=False)
 (out/'comparison.json').write_text(json.dumps(result,indent=2));row.update(status='completed',returncode=0)
except BaseException as exc:
 row.update(status='failed',returncode=1,error=repr(exc));raise
finally:row['elapsed_s']=time.time()-started;save()
