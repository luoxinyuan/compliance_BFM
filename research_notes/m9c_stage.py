import json,os,time
from pathlib import Path
from locomotion_rollout import UnsafePhysics
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m9c-expert-20261101';OLD=P/'bridge-v2-owned/runs/m9-contact-locomotion-20261101A'
def completed(stage):
 a=[a for a in json.loads((RUN/'compute-ledger.json').read_text())['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(a)==1;return Path(a[0]['path'])/'outputs'
def run_study(env,model,contract,out):
 import torch
 stage=os.environ['M9_STAGE'];start=time.monotonic();refs=json.loads((OLD/'native-cycles.json').read_text())['cycles']
 for r in refs.values():r['tensor']=torch.tensor(r['z'],device=env.standing.device)
 if stage=='qualification':
  from m9c_contact import run
  result=run(env,model,contract,out,cycles=refs)
 else:
  from m9c_rl import run
  result=run(env,model,contract,out,refs,stage)
 result['measured_wall_s']=time.monotonic()-start;return result
