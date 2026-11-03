"""Publish only bounded scalar summaries; raw physics, inputs and models stay remote."""
from pathlib import Path
import json,hashlib,gzip
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m38-rotation-20261103';DEST=Path(__file__).resolve().parent/'m38_artifacts'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def main():
 DEST.mkdir(exist_ok=True);(DEST/'runs').mkdir(exist_ok=True);index=[];models={}
 ledger=read(R/'compute-ledger.json');assert all(r['status']=='finished' for r in ledger['attempts'])
 for attempt in ledger['attempts']:
  path=Path(attempt['path']);job=path/'job'
  for name in ['summaries.json','source-summaries.json','geometry-metrics.json','audit.json','retained-training.json','result.json','learning-curve.json']:
   p=job/name
   if p.exists():write(DEST/'runs'/(path.name+'-'+name),read(p))
  for fn in ['summaries.json','source-summaries.json']:
   if (job/fn).exists():
    for row in read(job/fn):index.append(dict(run=path.name,source=attempt['source'],summary=row,trajectory=str(job/(row['label']+'.json.gz'))))
  for p in job.glob('retained-r*.pt'):models[p.stem]=dict(path=str(p),sha256=sha(p),runtime=attempt['source'])
 for name in ['diagnostics.json','exact-label-consistency.json','recovery-exact.json','recovery-inexact.json']:
  if (R/name).exists():write(DEST/name,read(R/name))
 write(DEST/'results.json',index);write(DEST/'models.json',models);write(DEST/'resources.json',ledger);write(DEST/'videos.json',read(R/'videos/manifest.json'))
 size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());tot=sum(r['wall_seconds'] for r in ledger['attempts']);write(DEST/'totals.json',dict(job_wall_seconds=tot,output_bytes=size,expert_updates=sum(r.get('actual_updates',0) for r in ledger['attempts']),rl_decisions=0,student_updates=0,owned_remaining=[r.get('remaining_processes',[]) for r in ledger['attempts']],episodes=len(index),scope='All supervised physical/train/render attempts including failures; lightweight offline file analysis is not an additional GPU job.'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,3.4))
 for label,folder in [('r1','A-goal-02'),('r2','A-goal-03')]:
  rows=read(R/folder/'job/learning-curve.json')
  for a,key in zip(ax,['qpd_loss','z_loss']):a.semilogy([r['step'] for r in rows],[r[key] for r in rows],label=label);a.set_xlabel('Supervised optimizer updates');a.set_ylabel(key);a.grid(alpha=.2);a.legend()
 fig.suptitle('M38 source-state fit: lower supervised error is not closed-loop retention');fig.tight_layout();fig.savefig(DEST/'learning-curves.png',dpi=150);print(json.dumps(read(DEST/'totals.json'),indent=2));print(models)
if __name__=='__main__':main()
