import sys,json,collections,time,hashlib
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/source-coverage-m1-20261109';R=S/'repo';sys.path[:0]=[str(R),str(R/'research_notes')]
from p4000_env import Task
start=time.monotonic();old=P/'experiments/ppo-retention-audit-20261109/runs/A1-train-5403'
t=object.__new__(Task);t.rng=np.random.default_rng(5403)
episodes=[];joint=collections.defaultdict(lambda:dict(decisions=0,physics_ticks=0,active_ticks=0,cap_ticks=0,reward=0.,force_squared=0.,error_squared=0.));spec=t.sample_spec();ep=dict(index=0,spec=spec,decisions=0,active=0,fs=0.,es=0.,cap=0,boundary=None,phases=collections.Counter());files=[]
for b in range(1,8):
 sp=old/f'rollout-{b}-stats.json';npf=old/f'rollout-{b}.npz';stats=json.loads(sp.read_text());data=np.load(npf)
 files.append(dict(path=str(sp),sha256=hashlib.sha256(sp.read_bytes()).hexdigest()))
 for st,done in zip(stats,data['done']):
  assert st['demand_family']==spec['demand_family'],(ep['index'],st['demand_family'],spec)
  key='|'.join([spec['source'],spec['demand_family'],st['phase']]);x=joint[key];x['decisions']+=1;x['physics_ticks']+=st['physics_steps'];x['active_ticks']+=st['active_steps'];x['cap_ticks']+=st['cap_steps'];x['reward']+=st['reward'];x['force_squared']+=st['active_force_squared'];x['error_squared']+=st['error_squared']
  ep['decisions']+=1;ep['active']+=st['active_steps'];ep['fs']+=st['active_force_squared'];ep['es']+=st['error_squared'];ep['cap']+=st['cap_steps'];ep['boundary']=st['boundary'] or ep['boundary'];ep['phases'][st['phase']]+=1
  if done:
   ep['complete']=True;episodes.append(ep);spec=t.sample_spec();ep=dict(index=len(episodes),spec=spec,decisions=0,active=0,fs=0.,es=0.,cap=0,boundary=None,phases=collections.Counter())
if ep['decisions']:ep['complete']=False;episodes.append(ep)
for ep in episodes:
 n=ep['active'];ep['error_cm']=100*(ep['es']/n)**.5 if n else None;ep['force_rms']=(ep['fs']/n)**.5 if n else 0.;ep['cap_fraction']=ep['cap']/n if n else 0.;ep['outcome_scope']='actual force/error/cap/boundary only; historical logs do not preserve full slip/yaw for all train episodes, full M1 gate unknown'
counts=collections.Counter((x['spec']['source'],x['spec']['demand_family'],x['complete']) for x in episodes)
ans=dict(runtime='8ef38c44ce71b910d11af10e3abe44b2bff9283e',seed=5403,actual_decisions=sum(x['decisions'] for x in episodes),complete=sum(x['complete'] for x in episodes),episodes=episodes,joint=dict(joint),counts=[dict(source=k[0],request=k[1],complete=k[2],episodes=v) for k,v in counts.items()],files=files,source_specs_reconstructed='exact p4000 sample_spec RNG seeded5403; all5376 family labels matched; independent eval leaves task RNG unchanged',limitations=['standing handoff always4.0','arm handoff5.6, requests.16-.30','no motion_walk_task episodes','no full train episode trajectory saved unless boundary; no invented full quality success count'],seconds=time.monotonic()-start)
out=R/'research_notes/source_coverage_m1_artifacts';out.mkdir(exist_ok=True);(out/'coverage5403.json').write_text(json.dumps(ans,indent=2));print(json.dumps({k:ans[k] for k in ['actual_decisions','complete','counts','seconds']}))
