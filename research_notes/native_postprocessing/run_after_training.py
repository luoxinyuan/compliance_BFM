"""Stage-local queue for already-authorized frozen evaluation and saved-qpos render.
No further RL, checkpoint selection or new method is launched here.
"""
from pathlib import Path
import json,time,subprocess,os,hashlib
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/trees/native-contact-learning/repo';O=P/'bridge-v2-owned/runs/native-contact-20261106';START=time.monotonic()
status=O/'postprocessing/status.json'
def save(x):status.write_text(json.dumps(x,indent=2))
def job(name,limit,command):
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
 save({'status':'frozen_final_running','training_results':results})
 job('frozen-final',4000,[str(P/'.venv/bin/python'),'-u',str(O/'postprocessing/native_final_control.py'),'--gpu','0','--output',str(O/'final-01'),'--seed','4810','--max-seconds','3900'])
 job('saved-qpos-video',1100,[str(P/'.venv/bin/python'),'-u',str(O/'postprocessing/native_render.py'),'--root',str(O),'--repo',str(R)])
 job('public-summary',180,[str(P/'.venv/bin/python'),str(O/'postprocessing/native_summarize.py'),'--root',str(O)])
 save({'status':'complete_server_artifacts_ready','videos':str(O/'videos'),'public':str(O/'public'),'Mac_sync_verified':False,'research_decision':'requires human-readable review; no automatic new training','ended':time.time()})
except BaseException:
 import traceback
 save({'status':'failed_preserved_no_automatic_restart','error':traceback.format_exc(),'ended':time.time()});raise
