"""Paired native standing controls, and privileged source recovery from learned states."""
from pathlib import Path
import json,time,math
import transfer_student_validated as runner
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m38-rotation-20261103'
def study(env,model,contract,out):
 import numpy as np,torch,mujoco
 from geometry_env import Task,PLAN
 from geometry_policy import expand
 from clean_policy import Expert,direct_command
 from wholebody_physics import goal_observation
 from locomotion_rollout import spherical
 from m11_runtime_speed import install
 torch.set_num_threads(1);torch.set_grad_enabled(False);runner.write(out/'runtime-speed.json',install(env));task=Task(env,model,contract,3801);task.reset(PLAN['single']);dev=env.standing.device;plan=json.loads(Path(__file__).with_name('m38_protocol.json').read_text())['m38']
 nets={}
 for who,path in [('T30',PLAN['teachers']['initial']['path']),('r2',str(R/'A-goal-03/job/retained-r2.pt'))]:
  ck=torch.load(path,map_location=dev,weights_only=True);n=Expert(1088,'direct').to(dev);expand(n,ck);nets[who]=n.eval().requires_grad_(False)
 summ=[]
 def run(name,spec,mode,takeover=None):
  task.reset(spec,record=True,label=name);last_a=0.;release_t=None;value=0.
  if mode=='source':
   q=env.sim.data.qpos.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(spec['rotation_deg']);zg=model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))
  while not task.done:
   x=task.features();t=float(env.tether.time);oldz=direct_command(nets['T30'].distribution(x).mean,env.standing)
   if takeover is not None and t<takeover:zz=direct_command(nets['r2'].distribution(x).mean,env.standing)
   elif mode=='standing':zz=env.standing
   elif mode=='source':
    if env.tether.active:a=float(np.clip((env.tether.geometry_target_yaw-task.geometry_yaw)/np.deg2rad(spec['rotation_deg']),0,1));last_a=a
    elif last_a>0:
     if release_t is None:release_t=t;value=last_a
     a=value*max(0.,1-(t-release_t)/1.5)
    else:a=0.
    zz=spherical(oldz,zg,a)
   else:zz=oldz
   task.step(zz/16)
  row=task.save(out);summ.append(row);runner.write(out/'summaries.json',summ);print('M38_CONTROL',name,row['error_cm'],row['sum_force_norm_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
 for name,changes in plan['final'].items():
  spec=dict(PLAN['dev']['pair_left'],**changes);spec.update(release=spec['onset']+spec['ramp']+2,end=spec['onset']+spec['ramp']+7);run('standing_'+name,spec,'standing')
  if name in ['left','right']:run('native_source_'+name,spec,'source')
 # Two representative source takeovers, full real learned prefix/history retained.
 for name in ['pair_left','pair_right']:run('source_takeover_'+name,PLAN['dev'][name],'source',7.)
 # Test old-feedback recovery on the actual new network walking state.
 run('T30_takeover_straight',PLAN['single'],'T30',5.)
 return dict(status='completed',decisions=0,expert_supervised_updates=0,student_optimizer_updates=0,episodes=len(summ),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
