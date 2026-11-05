import json,time,sys
from pathlib import Path
from prior_metrics import summarize
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/feedback-coexistence-20261105')
st=time.monotonic();out=[]
for seed in [4601,4602]:
 for ver in [0,8,16,24,32]:
  for task in ['local9','long65','stop']:
   p=R/f'ppo{seed}'/f'dev-{ver:04d}'/(task+'__mean.json.gz')
   if p.exists():
    m=summarize(p);out.append(m)
(R/'development-phase-metrics.json').write_text(json.dumps(out,indent=2))
print('rows',len(out),'seconds',time.monotonic()-st)
for m in out:
 print(m['path'].split('/')[-3:-1],m['label'],[(k,round(v['error_cm'] or 0,2),round(v['force_rms'],2),round(v['root_delta_cm'][0],2)) for k,v in m['phase'].items() if v and k in ['move','decelerate','hold']])
