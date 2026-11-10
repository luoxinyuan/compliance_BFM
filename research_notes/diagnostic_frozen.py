"""Complete exposed-state replay, matched frozen-function probes and physics branches."""
import copy, csv, gzip, json, math, time
from pathlib import Path
import numpy as np
import torch
import mujoco
from dual_track_task import Task,parent_manifests,global_rng
from dual_track_high import build,sha,write,sha_tensor
from dual_track_runtime import TrainingBranch
from dual_track_quality import quality
from native_transfer import phase_report
from clean_policy import direct_command
from locomotion_rollout import spherical
from rolling_snapshot import state_of

OLD=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/dual-track-formal-20261109')
PREFIXES=['3ffd2fc10497','42f40dfe854d','1d3ee72c35d3']

def portable(t):
 """Data-only archive; callback graph rebuilt by actual action replay, not pickled closures."""
 d=t.e.sim.data;mask=mujoco.mjtState.mjSTATE_INTEGRATION
 a=np.empty(mujoco.mj_stateSize(t.e.sim.model,mask));mujoco.mj_getState(t.e.sim.model,d,a,mask)
 arrays={k:getattr(d,k).copy() for k in dir(d) if not k.startswith('_') and isinstance(getattr(d,k),np.ndarray)}
 return dict(integration=a,mjdata_arrays=arrays,task=state_of(t,exclude=('physics','controls','stats','decision_rows','execution_stats','bootstrap_rows','prefix_data','native_z','refs','free','initial','assets')),
  env=state_of(t.e,exclude=('trace',)),base=state_of(t.e.base),history=copy.deepcopy(t.e.base.history_handler.history),
  tether=state_of(t.e.tether),force_history=state_of(t.h),rng=global_rng(),task_rng=copy.deepcopy(t.rng.bit_generator.state))

def same_state(t,nets,c):
 x=torch.cat([t.student_features(),t.features()],-1);previous=t.last_command.clone()
 raw=[n.distribution(x).mean for n in nets];z=[direct_command(v,t.e.standing) for v in raw]
 acts=[];q=[]
 for target in z:
  aa=torch.stack([t.m.act(t.e.obs,spherical(previous,target,f),mean=True) for f in (.2,.4,.6,.8,1.)]);acts.append(aa)
  q.append(torch.stack([t.qof(a) for a in aa]))
 dq=torch.rad2deg(q[1]-q[0]);cos=(z[0]*z[1]).sum()/256
 return x,dict(time=float(t.e.tether.time),raw_l2=float((raw[1]-raw[0]).norm()),
  z_angle_deg=float(torch.rad2deg(torch.acos(cos.clamp(-1,1)))),action_rms=float((acts[1]-acts[0]).square().mean().sqrt()),
  qpd_rms_deg=float(dq.square().mean().sqrt()),qpd_max_deg=float(dq.abs().max()),
  qpd_substep_max_deg=dq.abs().flatten(1).max(1).values.tolist(),
  own_last_z=previous.cpu(),raw=[v.cpu() for v in raw],z=[v.cpu() for v in z],
  BFM_observation={k:v.cpu().clone() for k,v in t.e.obs.items()},action=[v.cpu() for v in acts],q_PD=[v.cpu() for v in q])

def summary(t):return dict(boundary=t.boundary,quality=quality(t),**phase_report(t))

def run(e,m,c,out,args):
 start=time.monotonic();nets=[];identities=[]
 for i in (0,48):
  p=OLD/f'runs/expert-kl002/candidate-{i}.pt';n=build();n.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'],strict=True);n.eval().requires_grad_(False)
  nets.append(n);identities.append(dict(iteration=i,sha256=sha(p),checkpoint=str(p)))
 groups=[next(g for g in parent_manifests()['dev'] if g['parent_id'].startswith(prefix)) for prefix in PREFIXES]
 task=Task(e,m,c,56017);allrows=[];interventions=[];frames={};reproduction=[]
 # Immutable hook records solver/RNG at every5ms tick, including an unsafe final tick.
 original=task.e.sim.simulate_at_each_physics_step;tick_states=[];capture=False
 def observed_tick():
  try:return original()
  finally:
   if capture:
    d=e.sim.data;mask=mujoco.mjtState.mjSTATE_INTEGRATION;a=np.empty(mujoco.mj_stateSize(e.sim.model,mask));mujoco.mj_getState(e.sim.model,d,a,mask)
    tick_states.append(dict(time=float(e.tether.time),integration=a,qacc=d.qacc.copy(),qacc_warmstart=d.qacc_warmstart.copy(),
     qfrc_constraint=d.qfrc_constraint.copy(),efc_force=d.efc_force.copy(),ctrl=d.ctrl.copy(),xfrc_applied=d.xfrc_applied.copy(),
     solver_niter=d.solver_niter.copy(),rng=global_rng()))
 task.e.sim.simulate_at_each_physics_step=observed_tick
 for group in groups:
  prefix=group['parent_id'][:12]
  for ni in (0,1):
   torch.manual_seed(56017);np.random.seed(56017);label=f'{prefix}-i{identities[ni]["iteration"]}'
   task.reset(group['spec'],record=True,label=label);tick_states=[];capture=True;high=[];saved=[];branchstates=[]
   # All requests are already exposed. Times aligned to actual decision boundaries.
   boundaries=[group['spec']['source_until'],group['spec']['onset']+.1,group['spec']['onset']+.5,
      group['spec']['onset']+1.,group['spec']['onset']+2.,round((group['spec']['onset']+group['spec']['ramp'])*10)/10+.1]
   wanted=set(round(v*10) for v in boundaries) if prefix==PREFIXES[0] else set()
   while not task.done:
    task.prepare();t=float(e.tether.time);x,probe=same_state(task,nets,c)
    state=portable(task);state['fingerprint']=TrainingBranch(task).fingerprint
    saved.append(state);high.append(dict(input=x.cpu().clone(),**probe))
    if round(t*10) in wanted:branchstates.append((t,TrainingBranch(task)))
    task.step(nets[ni].distribution(x).mean)
   capture=False;result=summary(task);archive=OLD/f'runs/expert-kl002/dev-{identities[ni]["iteration"]}/summaries.json'
   old=next(r for r in json.loads(archive.read_text()) if r['parent_id']==group['parent_id'] and r['child']=='main')
   comparisons={k:abs(result['all'][k]-old['all'][k]) for k in ['error_cm','force_rms','cap_fraction']}
   assert max(comparisons.values())<1e-8,('Archived evaluation mismatch',label,comparisons)
   assert task.prefix_data['audit']['sha256']==old['source_state_sha256']
   reproduction.append(dict(label=label,source_sha256=old['source_state_sha256'],differences=comparisons))
   task.save(out)
   torch.save(dict(high=high,states=saved,physics_solver_rng=tick_states,spec=task.spec,model=identities[ni],
     reconstruction='original native source and archived actual29D actions; full solver and callback history fingerprint at every high boundary'),out/(label+'-complete.pt'))
   frames[label]=torch.cat([r['input'] for r in high])
   allrows.append(dict(label=label,parent_id=group['parent_id'],model=identities[ni],**result))
   print('D1_REPRODUCED',label,comparisons,flush=True)
   # Counterfactuals restore complete real prefixes and use only current feedback.
   for t,ss in branchstates:
    for kind in ('sustained','pulse300ms'):
     ss.restore();assert ss.digest()==ss.fingerprint
     first=len(task.physics);continued=[];capture=True;tick_states=[]
     while not task.done:
      x=torch.cat([task.student_features(),task.features()],-1)
      other=1-ni if kind=='sustained' or float(e.tether.time)<t+.3-1e-7 else ni
      task.step(nets[other].distribution(x).mean);continued.append(other)
     capture=False;row=dict(source_model=identities[ni]['iteration'],to_model=identities[1-ni]['iteration'],
      at=t,kind=kind,prefix_fingerprint=ss.fingerprint,prefix_exact=True,**summary(task))
     interventions.append(row);label2=f'{prefix}-from{ni}-at{round(t*10)}-{kind}'
     task.label=label2;task.save(out);torch.save(dict(first_physics_index=first,policy_sequence=continued,solver_rng=tick_states),out/(label2+'-solver.pt'))
     write(out/'interventions.json',interventions)
     print('D1_BRANCH',ni,t,kind,row['all']['error_cm'],flush=True)
   write(out/'reproduction.json',reproduction);write(out/'summaries.json',allrows)
 torch.save(frames,out/'developer-inputs.pt')
 return dict(status='completed',reproductions=len(reproduction),interventions=len(interventions),wall_s=time.monotonic()-start,
  formal_RL_decisions=0,optimizer_updates=0,models=identities,conditions='all exposed development; no sealed evaluation')
