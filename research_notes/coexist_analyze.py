"""Read-only curve and saved-physics aggregation. Never selects on heldout data."""
import json,gzip,time,sys
from pathlib import Path
import numpy as np
from prior_metrics import summarize
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105')
def curves():
 from collections import Counter
 result={}
 for seed in [4601,4602]:
  d=R/('ppo'+str(seed));p=d/'training.jsonl'
  if not p.exists():continue
  lines=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];family=Counter();phase=Counter();episodes=Counter()
  for x in lines:family.update(x['coverage']['family']);phase.update(x['coverage']['phase']);episodes.update(x['coverage']['completed'])
  dev=[]
  for p in sorted(d.glob('dev-*/summary.json')):
   rr=json.loads(p.read_text());dev.append(dict(iteration=int(p.parent.name.split('-')[1]),rows=[{k:v for k,v in r.items() if k not in ['handoff']} for r in rr],score=json.loads(p.with_name('score.json').read_text()) if p.with_name('score.json').exists() else None))
  result[str(seed)]=dict(training=lines,development=dev,decisions_by_family=dict(family),decisions_by_phase=dict(phase),complete_episodes_by_family=dict(episodes),selection=json.loads((d/'selection.json').read_text()) if (d/'selection.json').exists() else None)
 (R/'learning-curves.json').write_text(json.dumps(result,indent=2));return result
def physics():
 rows=[]
 for folder in ['ppo4601','ppo4602','final-learned','final-reference','final-FR','audit']:
  for p in sorted((R/folder).rglob('*.json.gz')):
   if 'source-prefix' in p.name or 'queries' in p.name or 'labels' in p.name:continue
   if 'worker-' in str(p) and 'train-' in p.name:continue
   m=summarize(p)
   if m:rows.append(m)
 (R/'physical-metrics.json').write_text(json.dumps(rows,indent=2));return rows
if __name__=='__main__':
 st=time.monotonic();c=curves();rr=physics() if '--physics' in sys.argv else []
 (R/'analysis-status.json').write_text(json.dumps(dict(seconds=time.monotonic()-st,episodes=len(rr),rl_updates=0)));print({k:len(v['training']) for k,v in c.items()},len(rr))
