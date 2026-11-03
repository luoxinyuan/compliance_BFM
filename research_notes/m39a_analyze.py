"""Read-only stage aggregation; active task and released recovery kept separate."""
import json,gzip,sys
from pathlib import Path
import numpy as np
from m38b_analyze import metrics,read,yaw,wrap
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m39a-demand-20261104')
def main():
 out=Path(sys.argv[1]);out.mkdir(exist_ok=True);rows=[];physics={};curves={};seen=set()
 paths=sorted(set(R.glob('*/job/summaries.json'))|set(R.glob('*/job/dev-*/summary.json')))
 for p in paths:
  for r in read(p):
   f=p.parent/(r['label']+'.json.gz')
   if f.resolve() in seen:continue
   seen.add(f.resolve());r=dict(r,trace=str(f),evaluation=str(p.parent.relative_to(R)));rows.append(r)
   # Detailed physical diagnostics on initial, selected, endpoint and sealed; all other raw retained.
   if '/dev-' in str(p):
    job=p.parent.parent;sel=read(job/'selection.json') if (job/'selection.json').exists() else {'iteration':0};status=read(job/'result.json') if (job/'result.json').exists() else {};it=int(p.parent.name.split('-')[-1])
    if it not in [0,sel['iteration'],status.get('iterations',0)]:continue
   physics[str(f)]=metrics(f);ep=read(f);rr=ep['physics'];initial=ep['initial'];cc=[]
   for t in np.arange(3,ep['condition']['end']+.001,.5):
    x=min(rr,key=lambda r:abs(r['force_time']-t));ty=float(np.degrees(wrap(yaw(x,'torso_matrix')-yaw(initial,'torso_matrix'))));py=float(np.degrees(wrap(yaw(x,'pelvis_matrix')-yaw(initial,'pelvis_matrix'))))
    cc.append(dict(time=x['force_time'],active=x['active'],torso=ty,pelvis=py,waist=ty-py,target_deg=float(np.degrees(wrap(x['target_yaw']-yaw(initial,'torso_matrix')))),root_delta_cm=(100*(np.array(x['root'])-initial['root'])).tolist(),force_xyz=[c['force'] for c in x['contacts']],sum_force=x['sum_force_norm'],yaw_moment=x['net_yaw_moment_com']))
   curves[str(f)]=cc
 ledger=read(R/'compute-ledger.json');trains={}
 for p in R.glob('*/job/training.jsonl'):
  ts=[json.loads(l) for l in p.read_text().splitlines()];trains[str(p)]=ts
 totals=dict(job_wall=sum(r.get('wall_seconds',r.get('elapsed_seconds',0)) for r in ledger['attempts']),supervised_updates=sum(r.get('actual_updates',0) for r in ledger['attempts']),rl_decisions=sum(r.get('actual_decisions',0) for r in ledger['attempts']),evaluated_episodes=len(rows),new_output_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink()))
 for name,obj in [('results',rows),('physics',physics),('time-curves',curves),('resources',ledger),('training',trains),('totals',totals)]: (out/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False))
 print(json.dumps(totals))
if __name__=='__main__':main()
