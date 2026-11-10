"""Fixed Issue58 serial follow-ups only; no general runner or new authorization."""
import argparse,fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent;P=S.parents[1]
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--kind',choices=['sealed','sentinel','safe','safe-resume','safe-post','physical-reference','physical-resumed','compile-control','compile-resume'],required=True)
p.add_argument('--target-runtime',type=Path);p.add_argument('--arm',choices=['A','B']);p.add_argument('--checkpoint',type=Path);p.add_argument('--sha256');p.add_argument('--iteration',type=int,default=0);p.add_argument('--package',type=Path)
a=p.parse_args();R=a.target_runtime or ROOT
category='F-RL-v2' if a.kind in ('sealed','sentinel') else 'P0-B' if a.kind.startswith('safe') else 'P0-A'
limit={'P0-A':14400,'P0-B':43200,'F-RL-v2':172800}[category];reserve={'P0-A':900,'P0-B':3600,'F-RL-v2':1200}[category]
previous=[json.loads(v.read_text()) for v in (S/'jobs').glob('*.json')]
assert not any(v.get('status')=='running' for v in previous),'Heavy stage task remains active'
spent=reserve+sum(v.get('elapsed_s',v['timeout_s']) for v in previous if v['category']==category)
timeout=min(6000 if a.kind in ('sealed','sentinel') else 42000 if a.kind.startswith('safe') else 1800,limit-spent-180);assert timeout>60
side=S/'jobs'/(a.name+'.json');assert not side.exists()
lock=(P/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(S).free>100*(1<<30)
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.used','--format=csv,noheader'],text=True)
uuid=gpu.split(',')[1].strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'RTX PRO 5000' in gpu and not any(uuid in v for v in apps.splitlines())
for repo in (ROOT,R):assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
out=S/'runs'/a.name;py=str(P/'.venv/bin/python')
if a.kind=='sealed':
 assert a.arm and a.checkpoint and a.sha256
 cmd=[py,str(R/'research_notes/frl_v2_stage.py'),'--mode','high_eval','--arm',a.arm,'--output',str(out),'--checkpoint',str(a.checkpoint),'--sha256',a.sha256,'--split','sealed','--iteration',str(a.iteration),'--record']
elif a.kind=='sentinel':
 assert a.checkpoint and a.sha256
 cmd=[py,str(ROOT/'research_notes/frl_sentinel_stage.py'),'--output',str(out),'--checkpoint',str(a.checkpoint),'--sha256',a.sha256]
elif a.kind=='safe':cmd=[py,str(ROOT/'research_notes/fb_safe_stage.py'),'--output',str(out)]
elif a.kind in ('safe-resume','safe-post'):
 assert a.package
 cmd=[py,str(ROOT/'research_notes'/('fb_safe_resume_stage.py' if a.kind=='safe-resume' else 'fb_safe_post_stage.py')),'--output',str(out),'--package',str(a.package)]
elif a.kind.startswith('physical-'):
 cmd=[py,str(ROOT/'research_notes/repair_physics_stage.py'),'--output',str(out),'--package',str(S/'runs/p0a-eager16-control-01/rolling-1064'),
  '--reference-model',str(S/'runs/p0a-eager16-control-01/fixed16-model.pt'),'--resumed-model',str(S/'runs/p0a-eager16-resume-01/fixed16-model.pt'),'--branch',a.kind.split('-')[1]]
else:
 cmd=[py,str(ROOT/'research_notes/repair_extended.py'),'--mode','d2control' if a.kind=='compile-control' else 'd2resume','--output',str(out),'--compiled']
 if a.kind=='compile-resume':assert a.package;cmd+=['--package',str(a.package)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(R)+':'+str(R/'research_notes'))
start=time.time();row=dict(name=a.name,category=category,kind=a.kind,started=start,timeout_s=timeout,status='running',supervisor_pid=os.getpid(),command=cmd,
 runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),supervisor_runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),gpu=gpu,formal_training=False)
def write():
 tmp=side.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(side)
write()
try:
 with (S/'jobs'/(a.name+'.log')).open('w') as log:
  c=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=c.pid;write()
  try:rc=c.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   if out.exists():(out/'STOP_REQUEST.json').write_text(json.dumps(dict(reason='hard budget',requested=time.time())))
   os.killpg(c.pid,signal.SIGTERM)
   try:rc=c.wait(90)
   except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);rc=c.wait()
   row['budget_timeout']=True
  row.update(status='completed' if rc==0 else 'failed',returncode=rc)
finally:row['elapsed_s']=time.time()-start;write()
print(json.dumps(row))
