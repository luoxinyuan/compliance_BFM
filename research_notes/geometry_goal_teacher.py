"""Bounded supervised Direct teacher acquisition from verified native waist goals.
Current target orientation is privileged; no future trajectory or online force rule.
Native reference rollouts supply initialization labels, not a claimed learned expert.
"""
from pathlib import Path
import json,time,gzip
import transfer_student_validated as runner

def study(env,model,contract,out):
 import numpy as np,torch,mujoco
 from geometry_env import Task,PLAN
 from geometry_policy import expand
 from clean_policy import Expert,direct_command
 from geometry_contact import smooth
 from wholebody_physics import goal_observation
 from locomotion_rollout import spherical
 from m11_runtime_speed import install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False);torch.manual_seed(1409)
 task=Task(env,model,contract,1409);task.reset(PLAN['single']);device=env.standing.device;ident=PLAN['teachers']['initial'];assert runner.sha(ident['path'])==ident['sha256'];ck=torch.load(ident['path'],map_location=device,weights_only=True)
 net=Expert(task.features().shape[-1],'direct').to(device);expand(net,ck);old=Expert(task.features().shape[-1],'direct').to(device);expand(old,ck);old.eval().requires_grad_(False);data=[];summaries=[]
 def run(name,spec,mode,collect=False):
  task.reset(spec,record=True,label=name);samples=[]
  if mode=='goal':
   q=env.sim.data.qpos.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');idx=env.sim.model.jnt_qposadr[j];q[idx]+=np.deg2rad(spec['rotation_deg']);assert env.sim.model.jnt_range[j,0]<q[idx]<env.sim.model.jnt_range[j,1];zgoal=model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))
  while not task.done:
   x=task.features();t=float(env.tether.time)
   if mode=='goal':
    a,_=smooth(t,spec['onset'],spec['ramp'])
    if t>=spec['release']:f,_=smooth(t,spec['release'],1.5);a*=1-f
    z=spherical(env.standing,zgoal,a);raw=z/16
   elif mode=='old':raw=old.distribution(x).mean;z=direct_command(raw,env.standing)
   else:raw=net.distribution(x).mean;z=direct_command(raw,env.standing)
   if collect:samples.append(dict(x=x[0].cpu().tolist(),z=z[0].cpu().tolist(),time=t,episode=name,obs={k:v.cpu().tolist() for k,v in env.obs.items() if k in env.keys}))
   task.step(raw)
  row=task.save(out);summaries.append(row);runner.write(out/'summaries.json',summaries)
  if samples:
   with gzip.open(out/(name+'-labels.json.gz'),'wt') as f:json.dump(samples,f)
  print('GOAL_TEACHER',name,row['error_cm'],row['sum_force_norm_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
  return samples
 for name in ['pair_left','pair_right']:data+=run('source_'+name,PLAN['dev'][name],'goal',True)
 for name in ['straight','local','pair_translate','pair_combined','no_contact']:data+=run('retain_'+name,PLAN['dev'][name],'old',True)
 no=dict(PLAN['dev']['pair_translate'],enabled=False,distance=0.,rotation_deg=0.);data+=run('retain_pair_no_contact',no,'old',True)
 x=torch.tensor([r['x'] for r in data],device=device);z=torch.tensor([r['z'] for r in data],device=device);groups=[torch.tensor([i for i,r in enumerate(data) if r['episode']==e],device=device) for e in sorted(set(r['episode'] for r in data))]
 net.requires_grad_(False);net.actor.requires_grad_(True);optimizer=torch.optim.Adam(net.actor.parameters(),lr=1e-4);curve=[]
 with torch.enable_grad():
  for step in range(4000):
   g=groups[step%len(groups)];ix=g[torch.randint(len(g),(128,),device=device)];zz=direct_command(net.distribution(x[ix]).mean,env.standing);loss=(zz-z[ix]).square().mean();optimizer.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));optimizer.step()
   if step%200==0:curve.append(dict(step=step,loss=float(loss.detach()),gradient=grad))
 net.eval().requires_grad_(False);p=out/'goal-initialized.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=x.shape[-1],kind='direct',updates=4000,seed=1409,source_teacher=ident),p)
 runner.write(out/'goal-initialized-training.json',dict(role='expert_initialization',updates=4000,samples=len(data),episodes=len(groups),curve=curve,sha256=runner.sha(p),permission='privileged current target orientation/error/velocity; no future trajectory; learned output only during execution'))
 checks=[]
 for r in data[::max(1,len(data)//128)]:
  xx=torch.tensor(r['x'],device=device)[None];zt=torch.tensor(r['z'],device=device)[None];zs=direct_command(net.distribution(xx).mean,env.standing);ob={k:torch.tensor(v,device=device) for k,v in r['obs'].items()};qt=task.qof(model.act(ob,zt,mean=True));qs=task.qof(model.act(ob,zs,mean=True));checks.append(dict(episode=r['episode'],angle_deg=float(torch.rad2deg(torch.acos(((zt*zs).sum()/256).clamp(-1,1)))),qpd_rms_deg=float(torch.rad2deg(qt-qs).square().mean().sqrt())))
 runner.write(out/'goal-initialized-alignment.json',checks)
 for name in ['pair_left','pair_right','straight','local','pair_translate','pair_combined','no_contact']:run('direct_'+name,PLAN['dev'][name],'learned')
 run('direct_pair_no_contact',no,'learned')
 # Fixed before this initialization: no development selection or optimization after these tests.
 for sign in [1,-1]:
  spec=dict(PLAN['dev']['pair_left'],rotation_deg=sign*12.,onset=4.65,ramp=6.4,release=13.05,end=18.05)
  for mode in ['goal','learned']:run('unseen_'+str(sign)+'_'+mode,spec,mode)
 return dict(status='completed',decisions=0,student_optimizer_updates=0,expert_supervised_updates=4000,elapsed_s=time.monotonic()-runner.START,episodes=len(summaries))
runner.study=study
if __name__=='__main__':runner.main()
