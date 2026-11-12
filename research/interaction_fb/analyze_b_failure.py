import json,math,argparse
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--original-run',type=Path,required=True);args=p.parse_args()
S=args.stage

def diff(a,b,path='root'):
 if torch.is_tensor(a) or torch.is_tensor(b):
  return None if torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes() else path
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return None if isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() else path
 if isinstance(a,dict):
  if a.keys()!=b.keys():return path+'.keys'
  for k in a:
   d=diff(a[k],b[k],path+'.'+str(k))
   if d:return d
  return None
 if isinstance(a,(list,tuple)):
  if len(a)!=len(b):return path+'.length'
  for i,(x,y) in enumerate(zip(a,b)):
   d=diff(x,y,path+f'[{i}]')
   if d:return d
  return None
 if isinstance(a,float):return None if isinstance(b,float) and a.hex()==b.hex() else path
 return None if a==b else path
old=torch.load(args.original_run/'uncommitted-stop.pt',map_location='cpu',weights_only=False)
out={}
for mode in ('sample','mean'):
 d=S/'runs'/f'{mode}-B19';j=json.loads((d/'probe.json').read_text());t=torch.load(d/'probe-trace.pt',map_location='cpu',weights_only=False);ob=json.loads((d/'same-state-actions.json').read_text())
 rows=t['physics'];contracts=ob['contract'];idx=next(i for i,c in enumerate(contracts) if c['name']=='left_ankle_pitch_joint');co=contracts[idx];qa=co['qposadr'];va=co['qveladr'];lo,hi=co['range']
 actions=ob['controls'];peak=max(rows,key=lambda x:x['joint_limit_excess_max_rad']);last=actions[-1]
 def view(x):
  return dict(time=x['time'],q=float(x['pre_qpos'][qa]),dq=float(x['pre_qvel'][va]),mean_action=x['mean'][0][idx],actual_action=x['actual'][0][idx],base_action=x['base_mean'][0][idx],adapter=x['adapter'][0][idx],mean_q_PD=x['mean_q_PD'][0][idx],actual_q_PD=x['actual_q_PD'][0][idx],base_q_PD=x['base_q_PD'][0][idx])
 r=dict(status=j['status'],frozen_agent_digest=j['frozen_agent_digest'],restored_rng_digest=j['restored_rng_digest'],trace_sha256=j['trace_sha256'],updates=j['updates'],original_cursor=j['original_cursor'],end_cursor=j['end_cursor'],joint_contract=co,t_start=rows[0]['time'],t_end=rows[-1]['time'],physics_ticks=len(rows),peak_excess=peak['joint_limit_excess_max_rad'],peak_time=peak['time'],peak_joint_q=peak['qpos'][qa],peak_joint_dq=peak['qvel'][va],peak_active=peak['active'],peak_force=peak['force'],mean_q_PD_min=min(x['mean_q_PD'][0][idx] for x in actions),first_control=view(actions[0]),last_control=view(last),mean_target_outside_count=sum(x['mean_q_PD'][0][idx]<lo or x['mean_q_PD'][0][idx]>hi for x in actions),controls=len(actions))
 if mode=='sample':
  matched=[v for v in old['physics'] if v['time']>=rows[0]['time']]
  r['original_first_difference']={key:diff(t[key],old[key] if key!='physics' else matched) for key in ('commands','actions','controls','physics')}
  assert all(v is None for v in r['original_first_difference'].values()),r['original_first_difference']
 assert all(abs(x['actual_q_PD'][0][idx]-(next(c for c in t['controls'] if abs(c['time']-x['time'])<1e-7)['q_PD'][idx]))<1e-6 for x in actions),'Recorded action mapping differs from execution'
 out[mode]=r
assert out['sample']['frozen_agent_digest']==out['mean']['frozen_agent_digest']
assert out['sample']['restored_rng_digest']==out['mean']['restored_rng_digest']
out['conclusion']=dict(sample_exactly_reproduces_original=True,mean_physical_branch_also_violates=out['mean']['status']=='hard_physics_boundary',no_new_updates=True,scope='same B19 actual visited state only; no claim of full mean episode from initial state')
(S/'b-failure-analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
