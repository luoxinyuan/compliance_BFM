"""Bounded frozen diagnostics/final evaluation; no optimization or Student."""
import json,time,copy
from pathlib import Path
import transfer_student_validated as runner

def study(env,model,contract,out):
 import torch,numpy as np,mujoco
 from clean_policy import Expert,direct_command
 from geometry_policy import expand
 from transition_env import Task,PLAN,install_frame
 from transition_metrics import extra
 from wholebody_physics import goal_observation
 from m11_runtime_speed import install
 from locomotion_rollout import spherical
 from geometry_contact import smooth
 from m9_cyclic import sample
 install_frame();runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 config=json.loads(Path(runner.ARGS.resume).read_text());task=Task(env,model,contract,runner.ARGS.seed);results=[];embeddings=[];identities={}
 def load(ident):
  assert runner.sha(ident['path'])==ident['sha256'];net=Expert(1088,'direct').to(env.standing.device);expand(net,torch.load(ident['path'],map_location=env.standing.device,weights_only=True));return net.eval().requires_grad_(False)
 nets={name:load(ident) for name,ident in config['models'].items()}
 def run(name,spec,mode,net=None):
  task.reset(spec,record=True,label=name);initialq=env.sim.data.qpos.copy();initialq[env.sim.model.jnt_qposadr[mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint')]]+=np.deg2rad(spec['rotation_deg']);goal=model.goal_inference(goal_observation(env,initialq,np.zeros(env.sim.model.nv))) if mode.startswith('native') else None
  nextb=float(env.tether.time);bb=[]
  while not task.done:
   t=float(env.tether.time)
   if mode=='standing':raw=env.standing/16
   elif mode.startswith('native'):
    # Explicit clock/reference baseline, not a deployable contact policy.
    progress,dp=smooth(t,spec['onset'],spec['ramp']);alpha=min(1.,progress*3) if spec['distance'] else 0.
    if t>=spec['release']:fade,_=smooth(t,spec['release'],1.5);alpha*=1-fade
    zz=spherical(env.standing,sample(task.refs['straight'],max(0,t-spec['onset'])/task.refs['straight']['period']),alpha)
    if spec['rotation_deg']:
     _,_,angle,_=__import__('geometry_contact').frame(spec,t);frac=abs(angle/np.deg2rad(spec['rotation_deg']))
     if t>=spec['release']:fade,_=smooth(t,spec['release'],1.5);frac*=1-fade
     zz=spherical(zz,goal,frac)
    raw=zz/16
   else:raw=net.distribution(task.features()).mean
   if t>=nextb-1e-6:
    nextb+=.5;go=goal_observation(env,env.sim.data.qpos,env.sim.data.qvel);b=model.backward_map(go)
    parity={k:float((go[k]-env.obs[k]).abs().max()) for k in go if k in env.obs}
    bb.append(dict(time=t,B=b[0].cpu().tolist(),input_parity=parity,root=env.sim.data.qpos[:3].tolist(),velocity=env.sim.data.qvel[:3].tolist()))
   task.step(raw)
  row=task.save(out);row.update(extra(task));results.append(row);embeddings.append(dict(label=name,samples=bb,scope='native backward_map original normalization on wholebody reconstructed obs; not success probability; no shaping'))
  runner.write(out/'summaries.json',results);runner.write(out/'backward-diagnostic.json',embeddings)
  print('TRANSITION_EVAL',name,row['error_cm'],row['sum_force_norm_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
  return row
 if config['mode']=='prepare':
  # Exact independently regenerated live prefixes: same complete state and next physics.
  checks=[]
  for key in ['walk_keep','left_return','walk_add_right']:
   s=PLAN['dev'][key];task.reset(s,record=True,label='parity');a=copy.deepcopy(task.prefix_data['audit']);net=task.source_net('initial');task.step(net.distribution(task.features()).mean);pa=np.array([r['qpos']+r['qvel']+r['force'] for r in task.physics])
   task.reset(s,record=True,label='parity');b=copy.deepcopy(task.prefix_data['audit']);task.step(net.distribution(task.features()).mean);pb=np.array([r['qpos']+r['qvel']+r['force'] for r in task.physics]);err=float(abs(pa-pb).max());assert a['sha256']==b['sha256'] and err==0
   checks.append(dict(condition=key,exact_handoff_sha=a['sha256'],next100ms_max_abs=err,history_length=a['contact_history_length'],physical_source_seconds=a['time']-3.))
  runner.write(out/'prefix-parity.json',checks)
  for key in ['walk_add_left','walk_add_right','stand_walk']:
   for cap in [5.,10.,20.]:
    spec=dict(PLAN['dev'][key],source_until=3.,total_cap=cap)
    for name in ['initial','T30','standing']:
     row=run(f'cap{int(cap)}_{key}_{name}',spec,name,nets.get(name))
     if row['boundary']:
      runner.write(out/'safety-stop.json',dict(condition=row['label'],boundary=row['boundary'],remaining='not executed after safety boundary'));return dict(status='physical_boundary',decisions=0,episodes=len(results),elapsed_s=time.monotonic()-runner.START)
  for key in ['stand_walk','neutral_left','neutral_right','walk_add_left','walk_add_right']:
   run('native_'+key,PLAN['dev'][key],'native_smooth')
 else:
  for name,net in nets.items():
   for key,spec in config['conditions'].items():run(name+'_'+key,spec,'learned',net)
 return dict(status='completed',decisions=0,episodes=len(results),elapsed_s=time.monotonic()-runner.START,student_optimizer_updates=0,expert_supervised_updates=0)
runner.study=study
if __name__=='__main__':runner.main()
