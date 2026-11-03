"""Read-only label coverage and deployment-space errors; no training or rollout."""
import gzip,json
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/t30-transfer-20261103';J=R/'A-prepare-01/job';O=R/'analysis';O.mkdir(exist_ok=True)
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
coverage=[]
for f in sorted(J.glob('*-queries.json.gz')):
 name=f.name.removesuffix('-queries.json.gz');ep=J/(name+'.json.gz')
 if not ep.exists():continue
 e=read(ep);q=read(f);counts={k:sum(a<=x['time']<b for x in q) for k,(a,b) in e['phases'].items()};coverage.append(dict(episode=name,labels=len(q),phases=counts,teacher='frozen Teacher30; Student history when on-policy query',boundary=e['boundary']))
errors=[]
for j in [J,R/'C-final-02/job']:
 for f in j.glob('*-same-state.json'):
  if 'take_' in f.name:continue
  name=f.name.removesuffix('-same-state.json');e=read(j/(name+'.json.gz'));rows=json.loads(f.read_text());ph={}
  for k,(a,b) in e['phases'].items():
   z=[r for r in rows if a<=r['time']<b]
   if z:ph[k]={field:dict(mean=float(np.mean([r[field] for r in z])),p95=float(np.quantile([r[field] for r in z],.95)),max=max(r[field] for r in z)) for field in ['z_angle','qpd_rms_deg','action_rms']}
  errors.append(dict(label=name,phases=ph,scope='Teacher queried on actual Student state; action comparison on identical observation and actual prior z at first 50Hz substep; no counterfactual physical success claim'))
(O/'coverage-and-action-errors.json').write_text(json.dumps(dict(coverage=coverage,errors=errors),indent=2));print(len(coverage),len(errors))
