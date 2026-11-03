"""Offline same-actual-state counterqueries; memory intervention is NOT a rollout."""
import json,gzip,time,os
from pathlib import Path
import transfer_student_validated as runner
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m38b-rotation-20261103'
def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)
def study(env,model,contract,out):
 import numpy as np,torch,mujoco
 from geometry_env import Task,PLAN
 from geometry_policy import expand
 from clean_policy import Expert,direct_command
 from locomotion_rollout import spherical
 from wholebody_physics import goal_observation
 from m11_runtime_speed import install
 torch.set_num_threads(1);torch.set_grad_enabled(False);runner.write(out/'runtime-speed.json',install(env));task=Task(env,model,contract,3801);task.reset(PLAN['single']);dev=env.standing.device
 def load(p):
  n=Expert(1088,'direct').to(dev);expand(n,torch.load(p,map_location=dev,weights_only=True));return n.eval().requires_grad_(False)
 old=load(PLAN['teachers']['initial']['path']);rows=[];firstfork=[];full=[]
 for mp in sorted(R.glob('*-train-*/job/model.json')):
  identity=read(mp);assert runner.sha(identity['path'])==identity['sha256'];net=load(identity['path']);arm=identity['arm']
  for name in ['pair_left','pair_right','straight','pair_translate','no_contact']:
   tr=read(mp.parent/(arm+'_'+name+'.json.gz'));spec=tr['condition'];task.reset(spec);goal=None
   if spec.get('rotation_deg'):
    q=env.sim.data.qpos.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(spec['rotation_deg']);goal=model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))
   ss=[];previous_teacher=None;release_t=None;release_value=0.;last_a=0.;physics=tr['physics']
   for i,decision in enumerate(tr['decisions']):
    cc=tr['control'][i*5:i*5+5];t=cc[0]['time'];x=torch.tensor([decision['obs']],device=dev);prev=x[:,565:821]*16;z=direct_command(net.distribution(x).mean,env.standing);tz=direct_command(old.distribution(x).mean,env.standing)
    actual=physics[max(0,min(i*20-1,len(physics)-1))]
    if goal is not None:
     if actual['active']:a=float(np.clip((actual['target_yaw']-task.geometry_yaw)/np.deg2rad(spec['rotation_deg']),0,1));last_a=a
     elif last_a>0:
      if release_t is None:release_t=t;release_value=last_a
      a=release_value*max(0.,1-(t-release_t)/1.5)
     else:a=0.
     tz=spherical(tz,goal,a)
    changed=x.clone()
    if previous_teacher is not None:changed[:,565:821]=previous_teacher/16
    mz=direct_command(net.distribution(changed).mean,env.standing);previous_teacher=tz.clone()
    def degrees(a,b):return float(torch.rad2deg(torch.acos(((a*b).sum()/256).clamp(-1,1))))
    errors=[];memory=[];memory_error=[];qtgt=[]
    for k,c in enumerate(cc):
     ob={key:torch.tensor(c['bfm_obs'][key],device=dev) for key in env.keys}
     q1=task.qof(model.act(ob,spherical(prev,z,(k+1)/5),mean=True));q2=task.qof(model.act(ob,spherical(prev,tz,(k+1)/5),mean=True));qm=task.qof(model.act(ob,spherical(prev,mz,(k+1)/5),mean=True));errors.append(float(torch.rad2deg(q1-q2).square().mean().sqrt()));memory.append(float(torch.rad2deg(q1-qm).square().mean().sqrt()));memory_error.append(float(torch.rad2deg(qm-q2).square().mean().sqrt()));qtgt.append(q2)
    phase='pre' if t<spec['onset'] else 'release' if t>=spec['release'] else 'contact'
    ss.append(dict(time=t,phase=phase,z_angle_deg=degrees(z,tz),memory_only_z_change_deg=degrees(z,mz),five_tick_qpd_rms_deg=float(np.sqrt(np.mean(np.square(errors)))),memory_only_five_tick_qpd_rms_deg=float(np.sqrt(np.mean(np.square(memory)))),memory_counterfactual_error_deg=float(np.sqrt(np.mean(np.square(memory_error)))),own_raw_replay_max=float((net.distribution(x).mean-torch.tensor([decision['raw']],device=dev)).abs().max())))
   full.append(dict(arm=arm,condition=name,rows=ss))
   for phase in ['pre','contact','release']:
    v=[x for x in ss if x['phase']==phase]
    if v:rows.append(dict(arm=arm,condition=name,phase=phase,n=len(v),**{k:float(np.mean([x[k] for x in v])) for k in ['z_angle_deg','memory_only_z_change_deg','five_tick_qpd_rms_deg','memory_only_five_tick_qpd_rms_deg','memory_counterfactual_error_deg','own_raw_replay_max']}))
 runner.write(out/'same-state.json',dict(summary=rows,scope='Teacher queried on actual learned-state BFM observations and actual executed z history. Memory-only perturbation uses previous counterqueried teacher command, leaves all physical/history features fixed; diagnostic sensitivity, NOT proof of closed-loop recovery or unique cause.'))
 with gzip.open(out/'queries.json.gz','wt') as f:json.dump(full,f)
 return dict(status='completed',expert_supervised_updates=0,student_optimizer_updates=0,decisions=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
