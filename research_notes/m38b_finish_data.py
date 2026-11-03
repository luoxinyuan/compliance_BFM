"""Recover completed physical evidence after JSON numpy-bool serialization failure.
No simulation, no model update; original failed attempt remains immutable.
"""
from pathlib import Path
import gzip,json,hashlib,numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m38b-rotation-20261103';OLD=P/'bridge-v2-owned/runs/m38-rotation-20261103';S=P/'bridge-v2-owned/runs/contact-geometry-20261103/C-goal-10/job';a=R/'A-prepare-02/job';out=R/'validated-data';out.mkdir()
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data=read(a/'base-data.json.gz');v=read(a/'visited-data.json.gz');assert len(data)==1554
manifest=[]
for p in sorted(S.glob('retain_*-labels.json.gz'))+sorted((OLD/'A-goal-03/job').glob('causal_*-labels.json.gz'))+sorted((OLD/'A-goal-03/job').glob('retain_stop_*-labels.json.gz')):
 e=Path(str(p).replace('-labels.json.gz','.json.gz'));manifest.append(dict(path=str(p),sha256=sha(p),physical_path=str(e),physical_sha256=sha(e),samples=len(read(p))))
recovery=[]
for n,t in [('pair_left',7.),('pair_right',7.),('straight',5.)]:
 now=read(a/('recovery_'+n+'.json.gz'));old=read(OLD/'A-goal-03/job'/('retained_'+n+'.json.gz'));pairs=[(x,y) for x,y in zip(old['physics'],now['physics']) if x['force_time']<t-1e-8]
 diff={k:max(float(np.max(np.abs(np.array(x[k])-y[k]))) for x,y in pairs) for k in ['qpos','qvel','force']};assert max(diff.values())==0
 def tail(d):
  rs=[x for x in d['physics'] if x['active'] and x['force_time']>=t-1e-8]
  return dict(error_cm=float(100*np.sqrt(np.mean([np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in x['contacts']]) for x in rs]))),force_rms=float(np.sqrt(np.mean([x['sum_force_norm']**2 for x in rs]))))
 b,c=tail(old),tail(now);accepted=not now['boundary'] and c['error_cm']<b['error_cm'] and c['force_rms']<b['force_rms'];assert accepted
 recovery.append(dict(condition=n,takeover=t,prefix_max_abs=diff,continued=b,recovery=c,accepted=bool(accepted),samples=len(read(a/('recovery_'+n+'-labels.json.gz'))),trace_sha=sha(a/('recovery_'+n+'.json.gz'))))
first=[next(r for r in data if r['episode']==e) for e in sorted(set(r['episode'] for r in data))];pairs=[]
for i,x in enumerate(first):
 for y in first[i+1:]:
  if np.array_equal(x['x'],y['x']):
   diff=float(np.max(np.abs(np.array(x['z'])-y['z'])));assert diff==0;pairs.append(dict(a=x['episode'],b=y['episode'],target_max=diff))
for name in ['base-data.json.gz','visited-data.json.gz']:(out/name).symlink_to(a/name)
d=dict(base_samples=len(data),base_episodes=10,visited_samples=len(v),source=manifest,base_sha256=sha(out/'base-data.json.gz'),visited_sha256=sha(out/'visited-data.json.gz'),identical_input_pairs=pairs,completion_note='Physics/data complete; numpy.bool serialization failed only metadata; this offline independent manifest recovers existing evidence without rerunning physics.')
for name,obj in [('dataset',d),('recovery',recovery)]:(out/(name+'.json')).write_text(json.dumps(obj,indent=2))
print(json.dumps(dict(base=len(data),visited=len(v),recovery=recovery),indent=2))
