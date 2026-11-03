"""Two explicit native goal waist-yaw expression probes, not a feedback teacher or controller."""
from pathlib import Path
import json,time
import transfer_student_validated as runner

def study(env,model,contract,out):
 import numpy as np,torch,mujoco
 from geometry_env import Task,PLAN
 from geometry_contact import smooth
 from wholebody_physics import goal_observation
 from locomotion_rollout import spherical
 from m11_runtime_speed import install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False);task=Task(env,model,contract,1203);rows=[]
 for angle in [15.,-15.]:
  for loaded in [False,True]:
   spec=dict(PLAN['dev']['pair_left' if angle>0 else 'pair_right'],enabled=loaded);label=('goal_left' if angle>0 else 'goal_right')+('_load' if loaded else '_free');task.reset(spec,record=True,label=label)
   q=env.sim.data.qpos.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');idx=env.sim.model.jnt_qposadr[j];q[idx]+=np.deg2rad(angle);assert env.sim.model.jnt_range[j,0]<q[idx]<env.sim.model.jnt_range[j,1]
   z=model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)));runner.write(out/(label+'-goal.json'),dict(waist_offset_deg=angle,latent=z.tolist(),source='native goal_inference; one specified waist-yaw target, no live joint override',scope='expression positive control only; not learned/privileged feedback expert'))
   while not task.done:
    t=float(env.tether.time);a,_=smooth(t,spec['onset'],spec['ramp'])
    if t>=spec['release']:f,_=smooth(t,spec['release'],1.5);a*=1-f
    zz=spherical(env.standing,z,a);task.step(zz/16)
   r=task.save(out);rows.append(r);runner.write(out/'summaries.json',rows);print('GOAL_GEOMETRY',label,r['error_cm'],r['sum_force_norm_rms'],r['actual_torso_turn_deg'],r['boundary'],flush=True)
 return dict(status='completed',episodes=4,decisions=0,elapsed_s=time.monotonic()-runner.START,student_optimizer_updates=0)
runner.study=study
if __name__=='__main__':runner.main()
