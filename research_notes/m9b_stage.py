import json,os
from pathlib import Path
from locomotion_rollout import UnsafePhysics
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
RUN=P/'bridge-v2-owned/runs/m9-feedback-continuation-20261101B'
OLD=P/'bridge-v2-owned/runs/m9-contact-locomotion-20261101A'
def completed(stage):
 a=[a for a in json.loads((RUN/'compute-ledger.json').read_text())['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(a)==1;return Path(a[0]['path'])/'outputs'
def run_study(env,model,contract,out):
 stage=os.environ['M9_STAGE']
 if stage=='diagnose':
  from m9b_diagnose import run
 elif stage.startswith('train'):
  from m9b_learning import run
 else:
  from m9b_contact import run
 if stage=='collect0':
  from m9b_attribution import run as attribute
  attribute(env,model,contract,out)
 if stage not in ['diagnose','train0','train1']:
  import torch
  import hashlib
  assert hashlib.sha256((OLD/'native-cycles.json').read_bytes()).hexdigest()=='52a4d23cc723209c29851faffa5356245687561c62ead6f76549a0ad1db1a070'
  refs=json.loads((OLD/'native-cycles.json').read_text())['cycles']
  for r in refs.values():r['tensor']=torch.tensor(r['z'],device=env.standing.device)
  return run(env,model,contract,out,cycles=refs)
 return run(env,model,contract,out)
