"""Finite same-weight source qualification and matched initialization data.
Native sources are privileged training references, not an online learned controller.
"""
import json,time,copy,gzip,subprocess
from pathlib import Path
import transfer_student_validated as runner
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
def study(env,model,contract,out):
 import numpy as np,torch,mujoco
 from clean_policy import Expert,direct_command
 from geometry_policy import expand
 from transition_env import Task,PLAN,install_frame
 from locomotion_rollout import spherical
 from wholebody_physics import goal_observation
 from m11_runtime_speed import install
 install_frame();import geometry_env,reach_contact;geometry_env.install_contact=reach_contact.install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 cfg=json.loads(Path(__file__).with_name('prior_protocol.json').read_text());task=Task(env,model,contract,4301);nets={}
 for k,v in cfg['models'].items():
  assert runner.sha(v['path'])==v['sha256'];n=Expert(1088,'direct').to(env.standing.device);expand(n,torch.load(v['path'],map_location=env.standing.device,weights_only=True));nets[k]=n.eval().requires_grad_(False)
 task.reset(copy.deepcopy(PLAN['dev']['stand_walk']),record=True,label='construction')
 q0=env.sim.data.qpos.copy();z0=model.goal_inference(goal_observation(env,q0,np.zeros(env.sim.model.nv)))
 oldik=json.loads((RUN/'arm-to-locomotion-20261105/physical-01/offline-ik.json').read_text());goals={}
 for r in oldik:
  if r['distance'] in [.05,.15]:goals[r['distance']]=torch.tensor(r['anchored_z'],device=env.standing.device)
 q=np.array(next(r for r in oldik if r['distance']==.15)['qpos']);j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_pitch_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(8)
 goals[.30]=model.project_z(env.standing+model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))-z0)
 yaw={}
 for sign in [-1,1]:
  q=q0.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(sign*15);yaw[sign]=model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))
 runner.write(out/'native-sources.json',dict(goals={str(k):v.tolist() for k,v in goals.items()},yaw={str(k):v.tolist() for k,v in yaw.items()},scope='fixed finite target set, goal_inference then original actor/PD; no live IK/joint override'))
 data={};summ=[]
 def source(kind,t,memory):
  if kind=='walk':return nets['I'].distribution(task.features()).mean
  if kind=='zero':return env.standing/16
  if env.tether.active:
   if kind in ['left','right']:
    val=float(env.tether.geometry_target_yaw-task.geometry_yaw);target=spherical(env.standing,yaw[1 if val>=0 else -1],min(abs(val)/np.deg2rad(15),1));memory['target']=target
   else:
    distance=max(0.,float(env.tether.anchor[0]-task.geometry_origin[0]));nodes=[0.,.05,.15,.30];zs=[env.standing,goals[.05],goals[.15],goals[.30]];i=min(2,next((i for i in range(3) if distance<=nodes[i+1]),2));target=spherical(zs[i],zs[i+1],float(np.clip((distance-nodes[i])/(nodes[i+1]-nodes[i]),0,1)));memory['target']=target
   memory['fade']=1.
  else:
   memory['fade']=max(0.,memory.get('fade',0.)-.1/1.5);target=spherical(env.standing,memory.get('target',env.standing),memory['fade'])
  return target/16
 cases=[('zero',0.,7.8),('arm5',.05,3.),('arm15',.15,3.),('body30',.30,7.8),('walk',.80,7.8),('left',0.,7.8),('right',0.,7.8)]
 for cap in [5.,10.]:
  for kind,dist,ramp in cases:
   for mode in ['I','source']:
    if time.monotonic()-runner.START>runner.ARGS.max_seconds-80:raise TimeoutError('bounded source stage')
    spec=copy.deepcopy(PLAN['dev']['stand_walk']);spec.update(distance=dist,ramp=ramp,release=4.25+ramp+2,end=4.25+ramp+6,anchor_mode='onset_actual',total_cap=cap)
    if kind in ['left','right']:spec.update(geometry='torso_pair',rotation_deg=15. if kind=='left' else -15.,anchor_mode='legacy')
    label=f'{int(cap)}_{mode}_{kind}';task.reset(spec,record=True,label=label);labels=[];memory={}
    while not task.done:
     t=float(env.tether.time);x=task.features();raw=nets['I'].distribution(x).mean if mode=='I' else source(kind,t,memory);prev=task.last_command.clone();z=direct_command(raw,env.standing);c=len(task.controls)
     task.step(raw);cc=task.controls[c:];labels.append(dict(x=x[0].cpu().tolist(),raw=raw[0].cpu().tolist(),z=z[0].cpu().tolist(),previous_z=prev[0].cpu().tolist(),time=t,episode=label,kind=kind,cap=cap,sequence=cc,valid_steps=len(cc)))
    row=task.save(out);summ.append(row);runner.write(out/'summaries.json',summ);data[label]=labels
    with gzip.open(out/(label+'-labels.json.gz'),'wt') as f:json.dump(labels,f)
    print('PRIOR_SOURCE',label,row['error_cm'],row['force_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
 runner.write(out/'data-index.json',{k:len(v) for k,v in data.items()})
 return dict(status='completed',episodes=len(summ),rl_decisions=0,supervised_updates=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
