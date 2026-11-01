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
 return run(env,model,contract,out)
