"""Independent geometry objective from saved contacts; old optimizer input preserved."""
import json,gzip,sys,math
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);rows=[]
for p in sorted(R.glob('physical-*/*.json.gz')):
 if any(s in p.name for s in ['prefix','labels','timelines']):continue
 e=json.load(gzip.open(p,'rt'))
 if not isinstance(e,dict) or 'rl_stats' not in e:continue
 offset=0;errors={};s=e['condition'];integral={}
 for st in e['rl_stats']:
  rr=e['physics'][offset:offset+st['physics_steps']];offset+=st['physics_steps'];active=[r for r in rr if r['active']]
  ecost=np.mean([np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in r['contacts']]) if r['active'] else 0. for r in rr]);ang=[]
  for r in rr:
   if not r['active']:ang.append(0.);continue
   def yaw(k):return math.atan2(r[k][1][0],r[k][0][0])
   def wrap(a):return math.atan2(math.sin(a),math.cos(a))
   pe=wrap(r['target_yaw']-yaw('pelvis_matrix'));te=wrap(r['target_yaw']-yaw('torso_matrix'));pair=s['geometry']=='torso_pair';ang.append(min((.8*te*te+.2*pe*pe if pair else pe*pe)/math.radians(20)**2,9)*(.5 if pair else .35*min(s['distance']/.8,1)))
  terms=dict(force=1.5*np.mean([len(r['contacts'])*sum(np.dot(c['force'],c['force']) for c in r['contacts'])/25 for r in rr]),tracking=.15*min(ecost/.04,16),heading=np.mean(ang),uncapped=.15*np.mean([np.mean([min(np.log1p(np.dot(c['raw_force'],c['raw_force'])/25),8) for c in r['contacts']]) for r in rr]),relative_velocity=.1*np.mean([np.mean([min(np.sum((np.array(c['anchor_velocity'])-c['velocity'])**2)/.25,16) for c in r['contacts']]) if r['active'] else 0. for r in rr]))
  for k,v in terms.items():errors[k]=max(errors.get(k,0),abs(v-st['terms'][k]));integral[k]=integral.get(k,0)+float(v)*st['control_steps']*.02
 rows.append(dict(label=e['label'],max_errors=errors,independent_cost_integrals=integral))
assert max(v for r in rows for v in r['max_errors'].values())<1e-10
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2));print('INDEPENDENT_REWARD_PASS',len(rows))
