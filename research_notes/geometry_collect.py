"""Build compact review artifacts from completed immutable outputs; excludes model/trace blobs."""
import json,hashlib,gzip
from pathlib import Path
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/contact-geometry-20261103')

def collect(out):
 out.mkdir(exist_ok=True);runs={};identities=[]
 for p in sorted(R.iterdir()):
  if not p.is_dir() or not (p/'job').exists():continue
  job=p/'job';d={}
  for name in ['result.json','summaries.json','selection.json','resume.json','shared-r0-training.json','shared-r0-alignment.json','goal-initialized-training.json','goal-initialized-alignment.json','takeover-prefix-check.json','first-divergence.json','observation-audit.json','geometry-metrics.json']:
   if (job/name).exists():d[name]=json.loads((job/name).read_text())
  if (job/'training.jsonl').exists():
   curves=[json.loads(x) for x in (job/'training.jsonl').read_text().splitlines()];d['learning_curve']=curves
   d['dev']={q.parent.name:json.loads(q.read_text()) for q in sorted(job.glob('dev-*/summary.json'))}
   d['dev_scores']={q.parent.name:json.loads(q.read_text()) for q in sorted(job.glob('dev-*/score.json'))}
  for f in job.glob('*.pt'):
   if f.name not in ['best.pt','initial.pt','last.pt','shared-r0.pt','goal-initialized.pt']:continue
   identities.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
  runs[p.name]=d
 (out/'runs').mkdir(exist_ok=True)
 index={}
 for name,d in runs.items():
  f=out/'runs'/(name+'.json');f.write_text(json.dumps(d,indent=2));index[name]=dict(file='runs/'+f.name,result=d.get('result.json'),selection=d.get('selection.json'),summaries=d.get('summaries.json'),has_physical_metrics='geometry-metrics.json' in d)
 (out/'results.json').write_text(json.dumps(index,indent=2));(out/'models.json').write_text(json.dumps(identities,indent=2))
 (out/'resources.json').write_text((R/'compute-ledger.json').read_text());(out/'old-geometry-audit.json').write_text((R/'old-geometry-audit.json').read_text())
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 for name,d in runs.items():
  if 'learning_curve' not in d:continue
  curve=d['learning_curve'];ax[0].plot([x['rl_decisions'] for x in curve],[x['reward'] for x in curve],label=name)
  scores=d.get('dev_scores',{});ax[1].plot([int(k.split('-')[1]) for k in scores],[v['score'] for v in scores.values()],marker='.',label=name)
 ax[0].set(xlabel='RL decisions',ylabel='Sampled mean decision reward');ax[1].set(xlabel='Batch',ylabel='Development selection score (lower better)')
 for a in ax:a.legend();a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=140);plt.close(fig)
 print('collected',len(runs),'runs',len(identities),'models')
if __name__=='__main__':
 import sys;collect(Path(sys.argv[1]))
