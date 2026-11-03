"""Independent reward integration and partner trajectory derivative audit."""
import json,gzip,sys,math
from pathlib import Path
import numpy as np
from m39a_env import PLAN,install_frame
install_frame()
from geometry_contact import frame
from geometry_env import wrap
out=Path(sys.argv[1]);report={'frame':{},'reward':{}}
for key,s in {**PLAN['dev'],**PLAN['sealed']}.items():
 err=[]
 for t in np.linspace(3,s['end'],401):
  p,v,a,w=frame(s,t);l=frame(s,t-1e-5);r=frame(s,t+1e-5);err+=[max(np.max(np.abs((r[0]-l[0])/2e-5-v)),abs((r[2]-l[2])/2e-5-w))]
 report['frame'][key]=max(err)
assert max(report['frame'].values())<1e-4
# Only committed summary entries: a concurrent gzip writer is not a finished episode.
completed=json.loads((out/'summaries.json').read_text())
for path in [out/(r['label']+'.json.gz') for r in completed]:
 if 'labels' in path.name or path.name=='coverage-data.json.gz':continue
 with gzip.open(path,'rt') as f:e=json.load(f)
 if 'physics' not in e:continue
 rows=e['physics'];stats=e['rl_stats'];offset=0;integral=0.;errors=[];parts={}
 for st in stats:
  rr=rows[offset:offset+st['physics_steps']];offset+=st['physics_steps'];pair=e['condition'].get('geometry')=='torso_pair'
  force=1.5*np.mean([len(r['contacts'])*sum(np.dot(c['force'],c['force']) for c in r['contacts'])/25 for r in rr]);ec=np.mean([np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in r['contacts']]) if r['active'] else 0 for r in rr]);heading=[]
  for r in rr:
   if not r['active']:heading.append(0.);continue
   P=np.array(r['pelvis_matrix']);T=np.array(r['torso_matrix']);p=wrap(r['target_yaw']-math.atan2(P[1,0],P[0,0]));t=wrap(r['target_yaw']-math.atan2(T[1,0],T[0,0]));heading.append(min(((.8*t*t+.2*p*p) if pair else p*p)/math.radians(20)**2,9)*(.5 if pair else .35*min(e['condition']['distance']/.8,1)))
  new=dict(force=force,tracking=.15*min(ec/.04,16),heading=np.mean(heading),uncapped=.15*np.mean([np.mean([min(np.log1p(np.dot(c['raw_force'],c['raw_force'])/25),8) for c in r['contacts']]) for r in rr]),relative_velocity=.1*np.mean([np.mean([min(np.sum((np.array(c['anchor_velocity'])-c['velocity'])**2)/.25,16) for c in r['contacts']]) if r['active'] else 0 for r in rr]))
  errors += [abs(new[k]-st['terms'][k]) for k in new];allterms={**st['terms'],**new};val=-.02*st['control_steps']*sum(allterms.values());errors.append(abs(val-st['reward']))
  integral+=val
  for k,v in allterms.items():parts[k]=parts.get(k,0)+.02*st['control_steps']*v
 report['reward'][path.stem]=dict(max_recompute_error=max(errors),return_recomputed=integral,integrated_costs=parts,physics=len(rows),decisions=len(stats),boundary=e['boundary'])
 assert max(errors)<1e-6,(path,max(errors))
(out/'reward-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:{'return':v['return_recomputed'],'maxerr':v['max_recompute_error']} for k,v in report['reward'].items()},indent=2))
