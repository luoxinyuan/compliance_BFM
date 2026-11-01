import json,os,time
from pathlib import Path
from locomotion_rollout import UnsafePhysics
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m10-task-appropriate-20261101';OLD=P/'bridge-v2-owned/runs/m9-contact-locomotion-20261101A'
def completed(stage):
 a=[a for a in json.loads((RUN/'compute-ledger.json').read_text())['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(a)==1;return Path(a[0]['path'])/'outputs'
def run_study(env,model,contract,out):
 import torch
 from m10_reference import prepare
 from m10_contact import run
 stage=os.environ['M9_STAGE'];start=time.monotonic()
 if stage in ['qualification','source_control']:prepare(env,model,out)
 if stage in ['train','train1']:
  from m10_learning import run as train
  return train(env,model,contract,out,stage)
 refs=json.loads((OLD/'native-cycles.json').read_text())['cycles']
 for r in refs.values():r['tensor']=torch.tensor(r['z'],device=env.standing.device)
 result=run(env,model,contract,out,cycles=refs)
 result['measured_wall_s']=time.monotonic()-start;return result
