"""Frozen-policy causal startup intervention: regenerate identical complete live prefixes.
No reset/copy at handoff. Same-state five-substep queries hold observation fixed;
they are not a differentiable or counterfactual physical rollout.
"""
import json,time,hashlib,copy,gzip
from pathlib import Path
import transfer_student_validated as runner

def study(env,model,contract,out):
 import torch,numpy as np,mujoco
 from clean_policy import Expert,direct_command
 from geometry_policy import expand
 from transition_env import Task,PLAN,install_frame
 from transition_metrics import extra
 from m11_runtime_speed import install
 from locomotion_rollout import spherical
 install_frame();import geometry_env,reach_contact;geometry_env.install_contact=reach_contact.install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 config=json.loads(Path(runner.ARGS.resume).read_text());task=Task(env,model,contract,4111);nets={};identities={};rows=[];baseline={};queries=[]
 for name,ident in config['models'].items():
  assert runner.sha(ident['path'])==ident['sha256'],name
  ck=torch.load(ident['path'],map_location=env.standing.device,weights_only=True)
  net=Expert(1088,'direct').to(env.standing.device);expand(net,ck);nets[name]=net.eval().requires_grad_(False)
  identities[name]={**ident,'runtime':ck.get('runtime'),'iteration':ck.get('iteration'),'input_dim':ck['input_dim'],'std':float(net.log_std.exp().mean())}
 runner.write(out/'identities.json',identities)
 def norm(v):
  if torch.is_tensor(v):return v.detach().cpu().tolist()
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,dict):return {str(k):norm(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)) or type(v).__name__=='deque':return [norm(x) for x in v]
  return v
 def digest(x):return hashlib.sha256(json.dumps(norm(x),sort_keys=True,allow_nan=False).encode()).hexdigest()
 def snapshot():
  d=env.sim.data;mask=mujoco.mjtState.mjSTATE_INTEGRATION
  state=np.empty(mujoco.mj_stateSize(env.sim.model,mask));mujoco.mj_getState(env.sim.model,d,state,mask)
  # The SAME instance continues; hashes additionally certify regenerated prefix parity.
  sections=dict(integration=state,solver=[d.qacc_warmstart,d.qacc,d.qfrc_constraint,d.ctrl,d.xfrc_applied],bfm=env.obs,previous_action=env.previous_action,
   last_z=task.last_command,last_qpd=task.lastq,prevz=task.prevz,raw=task.raw,contact_history=list(task.contact_history),force_history=list(task.h.rows),
   memory=[task.alpha,task.phase,task.phase_rate,task.tick,task.prepared],input=task.features(),support=env.last_support,
   tether=dict(time=env.tether.time,force=env.tether.force,anchor=env.tether.anchor,active=env.tether.active,rows=env.tether.geometry_rows),
   physics_prefix=task.physics,control_prefix=task.controls,source_prefix=task.prefix_data)
  return {k:digest(v) for k,v in sections.items()}
 def same_state(label,t):
  x=task.features();prev=task.last_command.clone();ob={k:v.clone() for k,v in env.obs.items()};raws={k:nets[k].distribution(x).mean for k in config['query_models']};zs={k:direct_command(v,env.standing) for k,v in raws.items()}
  qq={k:torch.stack([task.qof(model.act(ob,spherical(prev,z,(j+1)/5),mean=True))[0] for j in range(5)]) for k,z in zs.items()}
  for k in raws:
   queries.append(dict(prefix=label,time=t,model=k,raw_rms=float((raws[k]-raws['I']).square().mean().sqrt()),raw_norm=float(raws[k].norm()),z_deg=float(torch.rad2deg(torch.acos(((zs[k]*zs['I']).sum()/256).clamp(-1,1)))),qpd_substep_rms_deg=torch.rad2deg(qq[k]-qq['I']).square().mean(-1).sqrt().cpu().tolist(),qpd_joint_delta_deg=torch.rad2deg(qq[k]-qq['I']).cpu().tolist(),value=float(nets[k].value(x)),scope='same real state, identical BFM observations and previous executed z; frozen observation five-substep query'))
 def run(label,prefix,suffix=None,switch=None,pulse=False,noise_seed=None,condition="walk"):
  if time.monotonic()-runner.START>runner.ARGS.max_seconds-40:raise RuntimeError('bounded job time reached')
  spec=copy.deepcopy(config['specs'][condition]);task.reset(spec,record=True,label=label);snaps={};inter=None;labels=[]
  noise=torch.Generator(device=env.standing.device).manual_seed(noise_seed or 0)
  while not task.done:
   t=float(env.tether.time);tick=round((t-4.)*10);x=task.features()
   if tick in config['switch_ticks']:
    snaps[str(tick)]=snapshot()
    if suffix and tick==round((switch-4)*10):
     target=baseline[prefix+'__'+condition]['snapshots'][str(tick)];inter=dict(time=t,section_equal={k:snaps[str(tick)][k]==target[k] for k in target},reference=prefix)
     assert all(inter['section_equal'].values()),inter
   if suffix is None and noise_seed is None and prefix in ['I','A','B'] and 4-1e-6<=t<=8+1e-6:same_state(prefix,t)
   selected=prefix
   if suffix and tick>=round((switch-4)*10) and (not pulse or tick==round((switch-4)*10)):selected=suffix
   dist=nets[selected].distribution(x);raw=dist.mean
   if noise_seed is not None:raw=raw+dist.stddev*torch.randn(raw.shape,device=raw.device,generator=noise)
   previous=task.last_command.clone();z=direct_command(raw,env.standing);c=len(task.controls)
   task.step(raw)
   if suffix=='I' and switch<=t<switch+1.0-1e-7:
    labels.append(dict(x=x[0].cpu().tolist(),raw=raw[0].cpu().tolist(),z=z[0].cpu().tolist(),previous_z=previous[0].cpu().tolist(),time=t,episode=label,kind='walk',sequence=task.controls[c:],valid_steps=len(task.controls[c:])))
  row=task.save(out);row.update(extra(task));p=task.physics;active=[r for r in p if r['active']];qt=np.array([r['q_PD'] for r in task.controls]);initialx=p[0]['root'][0]
  root5=next((r['time'] for r in p if r['root'][0]-initialx>.05),None)
  def sustained(key,seconds=.1):
   n=round(seconds/.005)
   for i in range(len(p)-n):
    if all(key(r) for r in p[i:i+n]):return p[i]['time']
   return None
  # Operational progress/unloading descriptors, not universal gait success thresholds.
  row.update(prefix=prefix,suffix=suffix,switch=switch,pulse=pulse,noise_seed=noise_seed,intervention=inter,root5cm_time=root5,
   foot_unload_100ms_time=sustained(lambda r:min(r['foot_normal_force_n'])<5),root_velocity_005_200ms=sustained(lambda r:r['qvel'][0]>.05,.2),
   root_x_cm_at14=100*(min(p,key=lambda r:abs(r['time']-14))['root'][0]-initialx),qpd_max_deg=float(np.degrees(abs(np.diff(qt,axis=0))).max()),
   q_actual_max_deg=float(np.degrees(abs(np.diff(np.array([r['qpos'][7:] for r in p]),axis=0))).max()),snapshot_hashes=snaps)
  if switch is not None:
   tail=[r for r in active if r['time']>=switch];row['active_tail_error_cm']=100*float(np.sqrt(np.mean([r['error']**2 for r in tail])));row['active_tail_force_rms']=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in tail])))
  rows.append(row);runner.write(out/'summaries.json',rows)
  if suffix is None and noise_seed is None:baseline[prefix+'__'+condition]={'snapshots':snaps,'row':row}
  if labels:
   with gzip.open(out/(label+'-labels.json.gz'),'wt') as f:json.dump(labels,f)
  print('CAUSAL',label,'error',round(row['error_cm'],3),'force',round(row['force_rms'],3),'root14',round(row['root_x_cm_at14'],2),'boundary',row['boundary'],flush=True)
  if row['boundary']:raise RuntimeError('physical safety boundary: '+label)
  return row
 for name in ['I','A','B']:run(name+'__walk',name)
 runner.write(out/'same-state.json',queries)
 for t in config['switch_times']:
  run(f'B_to_I_{t:.1f}', 'B','I',t)
  run(f'I_to_B_{t:.1f}', 'I','B',t)
 for name in ['I','A','B']:run(name+'__small',name,condition='small')
 for p,q in [('B','I'),('I','B')]:run(f'{p}_to_{q}_small_5.0',p,q,5.,condition='small')
 runner.write(out/'same-state.json',queries)
 return dict(status='completed',episodes=len(rows),decisions=0,rl_training_decisions=0,elapsed_s=time.monotonic()-runner.START,student_optimizer_updates=0,expert_supervised_updates=0)
runner.study=study
if __name__=='__main__':runner.main()
