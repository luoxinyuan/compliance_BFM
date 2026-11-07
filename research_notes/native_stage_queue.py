"""Stage-local queue for already-authorized frozen evaluation and saved-qpos render.
No further RL, checkpoint selection or new method is launched here.
"""
from pathlib import Path
import json,time,subprocess,os,hashlib
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/trees/native-contact-learning/repo';O=P/'bridge-v2-owned/runs/native-contact-20261106';START=time.monotonic()
status=O/'postprocessing/status.json'
def save(x):status.write_text(json.dumps(x,indent=2))
def job(name,limit,command):
 # Live resources at each GPU job boundary, never reuse the initial snapshot.
 if name not in ['public-summary']:
  for attempt in range(30):
   gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();mem,util=[int(v.strip()) for v in gpu.split(',')]
   mi={l.split(':')[0]:int(l.split(':')[1].split()[0])*1024 for l in Path('/proc/meminfo').read_text().splitlines()};snap={'time':time.time(),'gpu0':gpu,'RAM_available_bytes':mi['MemAvailable'],'CPU_affinity':len(os.sched_getaffinity(0)),'load':os.getloadavg()}
   (O/'postprocessing'/(name+'-resource-snapshot.json')).write_text(json.dumps(snap,indent=2))
   if mem<1000 and util<5 and mi['MemAvailable']>24*1024**3:break
   time.sleep(10)
  else:raise RuntimeError('No independently safe resource at job boundary; no CPU fallback or other-user intervention')
 subprocess.run([str(P/'.venv/bin/python'),str(R/'research_notes/native_job.py'),'--ledger',str(O/'ledger.json'),'--name',name,'--limit',str(limit),'--',*command],check=True,cwd=O)
try:
 save({'status':'waiting_for_two_frozen_training_results','pid':os.getpid(),'started':time.time(),'runtime':subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 while True:
  results=[]
  for arm in ['ordinary','guided']:
   f=O/f'ppo-{arm}/result.json'
   if f.exists():
    result=json.loads(f.read_text());assert result['status']=='completed',result;assert result['decisions']==24576,result;results.append(result)
  if len(results)==2 and not any(j.get('status')=='running' for j in json.loads((O/'ledger.json').read_text())['jobs']):break
  assert time.monotonic()-START<14500,'bounded queue wait exceeded'
  time.sleep(20)
 check=json.loads((O/'native-coordinate-check.json').read_text());assert check['status']=='passed'
 save({'status':'coordinate_training_running','training_results':results,'no_sealed_evaluation_opened':True})
 coordinate=P/'bridge-v2-owned/trees/native-contact-docs/repo'
 job('ppo-coordinate',6500,[str(P/'.venv/bin/python'),'-u',str(coordinate/'research_notes/native_ppo_coordinate.py'),'--kind','direct','--exploration','guided','--stage','native-coordinate','--output',str(O/'ppo-coordinate'),'--gpu','0','--iterations','32','--seed','4801','--resume',str(O/'native-initial.pt'),'--max-seconds','6200'])
 cr=json.loads((O/'ppo-coordinate/result.json').read_text());assert cr['status']=='completed' and cr['decisions']==24576,cr
 save({'status':'frozen_final_running','training_results':results+[cr]})
 job('frozen-final',5000,[str(P/'.venv/bin/python'),'-u',str(O/'postprocessing/native_final_control.py'),'--gpu','0','--output',str(O/'final-01'),'--seed','4810','--max-seconds','4900'])
 job('saved-qpos-video',1100,[str(P/'.venv/bin/python'),'-u',str(O/'postprocessing/native_render.py'),'--root',str(O),'--repo',str(R)])
 job('public-summary',300,[str(P/'.venv/bin/python'),str(O/'postprocessing/native_summarize.py'),'--root',str(O)])
 save({'status':'complete_server_artifacts_ready','videos':str(O/'videos'),'public':str(O/'public'),'Mac_sync_verified':False,'research_decision':'requires human-readable review; no automatic new training','ended':time.time()})
except BaseException:
 import traceback
 save({'status':'failed_preserved_no_automatic_restart','error':traceback.format_exc(),'ended':time.time()});raise
