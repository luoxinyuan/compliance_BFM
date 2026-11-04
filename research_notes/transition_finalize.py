"""One finite continuation of this pre-registered experiment, not a watch service.
Waits only for the two named owned jobs; no training launch/retry/extension.
Locks their predeclared developer-selected models, evaluates once, renders saved data.
"""
import json,time,hashlib,subprocess,gzip,os,sys
from pathlib import Path
from transition_supervisor import RUN,P,ROOT,snapshot
from m7_io import write
PLAN=json.loads((ROOT/'research_notes/transition_protocol.json').read_text())
start=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def idle():
 # Refresh independent GPU, RAM and CPU load at each bounded stage launch.
 mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
 assert mem['MemAvailable']>48*1024**2,'Insufficient independent RAM for original frozen model'
 assert os.getloadavg()[0]<len(os.sched_getaffinity(0))*.8,'Shared CPU currently busy; do not compete'
 for g in [5,6,1,2]:
  x=snapshot(g)
  if x['used_mib']<100 and x['utilization']<5:return g
 raise RuntimeError('No actually idle permitted GPU; leave evaluation pending')
try:
 jobs=[RUN/'A-ppo-03',RUN/'B-ppo-04']
 write(RUN/'continuation-status.json',dict(status='waiting_named_training_jobs',pid=os.getpid(),jobs=list(map(str,jobs)),no_training_launch=True))
 while True:
  led=json.loads((RUN/'compute-ledger.json').read_text())['attempts'];rows=[next(x for x in led if x['path']==str(j)) for j in jobs]
  if all(x['status']=='finished' for x in rows):break
  if time.monotonic()-start>8000:raise TimeoutError('Bounded continuation wait expired')
  time.sleep(30)
 models=dict(PLAN['sources'])
 for name,j,r in zip(['standing_curriculum','source_curriculum'],jobs,rows):
  result=json.loads((j/'job/result.json').read_text())
  assert r['exit_code']==0 and result['status']=='completed' and result['decisions']==49152, (name,result)
  selection=json.loads((j/'job/selection.json').read_text());p=j/'job/best.pt';assert sha(p)==selection['sha256']
  models[name]={'path':str(p),'sha256':sha(p),'selection':selection}
 config={'mode':'final','models':models,'conditions':PLAN['sealed'],'selection_locked_before_final':True}
 dest=RUN/'final-config.json';assert not dest.exists();write(dest,config)
 g=idle();cmd=[str(P/'.venv/bin/python'),str(ROOT/'research_notes/transition_supervisor.py'),'--arm','C','--kind','direct','--stage','final','--gpu-index',str(g),'--seed','4111','--seconds','3000','--resume',str(dest)]
 write(RUN/'continuation-status.json',dict(status='frozen_evaluation',pid=os.getpid(),config_sha256=sha(dest),gpu=g))
 subprocess.run(cmd,check=True)
 rows=json.loads((RUN/'compute-ledger.json').read_text())['attempts'];last=rows[-1];assert last['exit_code']==0 and last.get('result',{}).get('study',last.get('result',{})).get('status')=='completed',last
 job=Path(last['path'])/'job'
 # Reuse established saved-physics analyzer; only phase partition changes.
 import m39b_analyze
 from transition_env import phase
 m39b_analyze.phase=phase
 paths=[job/(name+'_'+key+'.json.gz') for name in models for key in PLAN['sealed']]
 t=time.monotonic();physics=[m39b_analyze.summarize(p) for p in paths];write(RUN/'final-physics.json',physics);write(RUN/'analysis-cost.json',dict(cpu_saved_analysis_seconds=time.monotonic()-t))
 # Full actual source prefix plus actual policy suffix; no new simulation.
 video_input=RUN/'video-input';video_input.mkdir();pairs=[]
 for index,key in enumerate(['walk_add_left_new','walk_add_right_new','stand_walk_new'],1):
  sides=[]
  for name in ['initial','standing_curriculum','source_curriculum','T30']:
   path=job/(name+'_'+key+'.json.gz');ep=json.loads(gzip.decompress(path.read_bytes()));pre=json.loads(gzip.decompress((job/(name+'_'+key+'-source-prefix.json.gz')).read_bytes()))
   ep['physics']=pre['physics']+ep['physics'];ep['control']=pre['control']+ep['control'];ep['initial']=pre['initial'];dst=video_input/path.name
   with gzip.open(dst,'wt') as f:json.dump(ep,f)
   sides.append(dict(title=name+' | frozen development-selected',path=str(dst)))
  pairs.append(dict(name=f'{index:02d}-{key}',grid4=True,sides=sides))
 vp=RUN/'video-pairs.json';write(vp,pairs);g=idle();subprocess.run([str(P/'.venv/bin/python'),str(ROOT/'research_notes/transition_render_saved.py'),'--spec',str(vp),'--output',str(RUN/'videos-final'),'--gpu',str(g),'--asset-root',str(P/'bridge-v2-owned/trees/m39a-demand/repo')],check=True)
 write(RUN/'continuation-status.json',dict(status='evaluation_and_render_completed_needs_scientific_review',pid=os.getpid(),final_job=str(job),config_sha256=sha(dest),no_postfinal_training=True,elapsed_orchestration_wall_seconds=time.monotonic()-start,orchestration_wait='idle waiting only; compute jobs separately ledgered'))
except BaseException:
 import traceback
 write(RUN/'continuation-status.json',dict(status='stopped_preserve_evidence',pid=os.getpid(),error=traceback.format_exc(),no_automatic_retry=True));raise
