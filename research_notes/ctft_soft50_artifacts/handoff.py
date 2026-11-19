import copy,gzip,json
from pathlib import Path
import numpy as np
import torch,mujoco
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def setup(env,model,old):
 import clean_env
 from locomotion_rollout import snapshot
 archived=old/'compatibility-prefix'
 common=json.loads((archived/'backend-common-state.json').read_text())
 trace=json.loads((archived/'backend-standing-prefix.json').read_text())['physics']
 def set_tensor(dest,src): dest.copy_(torch.as_tensor(src,dtype=dest.dtype,device=dest.device))
 def restore(e,m,act):
  e.qualification_active=False;e.m1_anchor_hook=None;e.interruption_window=None
  e.cfg.enabled=False;e.cfg.body='right_wrist_yaw_link';e.cfg.local_point=[0,0,0]
  e.reset(seed=0,trajectory=(trace[0]['wrist'],[0,0,0]))
  d=e.sim.data;mask=mujoco.mjtState.mjSTATE_INTEGRATION
  for key,obj in [('base_tensors',e.base),('sim_tensors',e.sim),('wrapped_tensors',e.wrapped)]:
   for k,v in common[key].items():set_tensor(getattr(obj,k),v)
  for k,v in common['bfm_obs'].items():set_tensor(e.obs[k],v)
  for k,v in common['history'].items():set_tensor(e.base.history_handler.history[k],v)
  set_tensor(e.previous_z,common['previous_z']);e.previous_action=torch.as_tensor(common['previous_action'],device=e.standing.device,dtype=e.standing.dtype)
  e.history=np.asarray(common['force_history'],dtype=e.history.dtype);e.steps=common['steps'];e.base.common_step_counter=common['common_step_counter']
  for k,v in common['env_bookkeeping'].items():setattr(e,k,np.array(v) if k in ('last_point','last_qpos') else copy.deepcopy(v))
  for k,v in common['tether'].items():setattr(e.tether,k,np.asarray(v).copy() if isinstance(v,list) else v)
  integration=np.asarray(common['mj_integration']);mujoco.mj_setState(e.sim.model,d,integration,mask);mujoco.mj_forward(e.sim.model,d);mujoco.mj_setState(e.sim.model,d,integration,mask)
  now=snapshot(e);assert now==common,('3s-state', [k for k in common if now[k]!=common[k]])
  e.trace=copy.deepcopy(trace)
 clean_env.reset_common=restore
 env.qualification_active=False;env.interruption_window=None;env.m1_anchor_hook=None;env.reset(seed=0)
 return restore
def prefix(task,condition,old,out):
 e=task.e;m=task.m;p=old/'runs/rolling-02'
 pref=read(p/'rolling_continue-source-prefix.json.gz');ep=read(p/'rolling_continue.json.gz')
 ctrl=pref['control']+ep['control'][:50]
 assert len(ctrl)==100
 orig=m.act
 def from_archive(*args,**kw):
  idx=round((e.tether.time-3)*50)
  assert 0<=idx<len(ctrl),idx
  return torch.as_tensor(ctrl[idx]['action'],device=e.standing.device,dtype=e.standing.dtype).reshape(1,-1)
 m.act=from_archive
 try:
  task.reset(condition,record=True,label='source')
  fields=['integration_state','bfm_obs','previous_action','last_z','last_qpd','force_history','features']
  audit=task.prefix_data['audit'];diff=[k for k in fields if audit[k]!=pref['audit'][k]]
  (out/'prefix-comparison.json').write_text(json.dumps({k:(audit[k]==pref['audit'][k]) for k in fields}))
  if diff==['force_history']:
   saved=pref['audit']['force_history'];live=audit['force_history']
   assert len(saved)==len(live)
   changes={}
   for aa,bb in zip(live,saved,strict=True):
    for aa,bb in zip(aa,bb,strict=True):
     for key in aa:
      if aa[key]!=bb[key]:
       changes[key]=max(changes.get(key,0.),float(np.max(np.abs(np.asarray(aa[key])-np.asarray(bb[key])))))
   (out/'contact-log-differences.json').write_text(json.dumps(changes))
   assert set(changes)<=set(['velocity','reference_com','point','anchor']),changes
   # Restore the archived full rows, including inactive precontact kinematic metadata.
   # Every executed force/activation, solver, BFM input/action, and physical tick must match.
   task.contact_history.clear();task.contact_history.extend(copy.deepcopy(saved))
   audit['force_history']=list(task.contact_history)
  assert all(audit[k]==pref['audit'][k] for k in fields),('4s-audit',diff)
  for row in ep['decisions'][:10]:
   task.step(torch.as_tensor(row['raw'],device=e.standing.device,dtype=e.standing.dtype).reshape(1,-1))
  for oldrows,newrows in [(pref['physics'],task.prefix_data['physics']),(ep['physics'][:200],task.physics)]:
   assert len(oldrows)==len(newrows)
   for j,(a,b) in enumerate(zip(oldrows,newrows)):
    for k in ('qpos','qvel','force','foot_normal_force_n','time','applied_contact_wrench','torque'):
     assert a[k]==b[k],('prefix-tick',j,k)
  for j,(a,b) in enumerate(zip(ep['control'][:50],task.controls,strict=True)):
   for k in ('executed_z','action','q_PD','bfm_obs','pre_qpos','pre_qvel'):assert a[k]==b[k],('control',j,k)
 finally:m.act=orig
 return ep
