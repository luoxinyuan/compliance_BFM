"""Bounded native assets qualification, original frozen BFM/PD, no old actor."""
import json,time,gzip,copy,hashlib
from pathlib import Path
import transfer_student_validated as runner

def study(env,model,contract,out):
 import torch,numpy as np,mujoco
 from m11_runtime_speed import install
 from wholebody_physics import goal_observation
 from clean_policy import direct_command
 from locomotion_rollout import spherical
 from native_env import Task,spec,install as install_contact,RUN
 install_contact();runner.write(out/'runtime-speed.json',install(env))
 P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs';proto=json.loads(Path(__file__).with_name('native_protocol.json').read_text())
 refs=R/'m10-task-appropriate-20261101/source_control-attempt-01/outputs/references.json';assert runner.sha(refs)=='211a192ddf6e3d534a1ef273c22e9cf977510ccb235e1c88dc2f0b24e8d6f2c0'
 ik=R/'arm-to-locomotion-20261105/physical-01/offline-ik.json';old=json.loads(ik.read_text());zs={'standing':env.standing.tolist()}
 for distance,name in [(.05,'goal_arm5'),(.15,'goal_arm15')]:zs[name]=next(v for v in old if v['distance']==distance)['anchored_z']
 # Temporary manifest enables only original Task reset; all source tensors derive from same BFM.
 runner.write(RUN/'native-assets.json',{'static_z':zs})
 task=Task(env,model,contract,4800);task.reset(spec(0.,4.,until=3.),record=True,label='construction')
 q0=env.sim.data.qpos.copy();z0=model.goal_inference(goal_observation(env,q0,np.zeros(env.sim.model.nv)));q=np.array(next(v for v in old if v['distance']==.15)['qpos']);j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_pitch_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(8)
 zs['goal_lean30']=model.project_z(env.standing+model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))-z0).tolist()
 reward=R/'issue21-reward-stage-20261030A/inference-attempt-01/outputs/result.json';rr=json.loads(reward.read_text());assert rr['model_sha256']==proto['bfm_sha256'];assert rr['buffer_sha256']==proto['buffer_sha256']
 # Existing full verified SHA record and current byte size, no 26.7GB load or download.
 buffer=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets/62b4206d68e026de5e5dc7efb1529bccfb95164c/train/buffer.hdf5');assert buffer.stat().st_size==26675212288
 assets=dict(static_z=zs,model_sha256=contract['model_sha256'],motion_reference={'path':str(refs),'sha256':runner.sha(refs),'variant':'task','source':'same-weight goal inference on traced native motion with documented M10 offline arm/waist pose correction; no learned actor'},goal_reference={'path':str(ik),'sha256':runner.sha(ik),'scope':'offline IK pose→same frozen goal_inference anchored difference; no online IK'},reward_reference={'path':str(reward),'sha256':runner.sha(reward),'model_sha256':rr['model_sha256'],'buffer_sha256':rr['buffer_sha256'],'buffer_bytes_now':buffer.stat().st_size,'used_for_training':False,'reason':'historical move-ego side drift/58.99deg qPD failure retained; not promoted from name'},normalizer='same model.safetensors obs normalizer, FP32, frozen',old_high_actor_loaded=False)
 runner.write(RUN/'native-assets.json',assets);task.native_z={k:torch.tensor(v,device=env.standing.device).reshape(1,256) for k,v in zs.items()};runner.write(out/'assets.json',assets)
 cases=[('standing',0.,False),('goal_arm5',0.,False),('goal_arm5',.05,True),('goal_arm15',.15,True),('goal_lean30',.30,True),('motion_walk_task',0.,False),('motion_walk_task',.72,True)]
 summaries=[];labels=[]
 for i,(name,distance,enabled) in enumerate(cases):
  s=spec(distance,6. if name=='motion_walk_task' else 3.,4.8,until=3.,enabled=enabled);s['release']=12.;s['end']=16.;label=f'{i:02d}-{name}-'+('contact' if enabled else 'free');task.reset(s,record=True,label=label)
  while not task.done:
   t=float(env.tether.time)
   if t<4.8:z=env.standing
   elif t<10.5:
    if name=='motion_walk_task':
     from m9_cyclic import sample
     u=min((t-4.8)/1.5,1.);z=spherical(env.standing,sample(task.refs['straight'],max(0.,t-6.3)/task.refs['straight']['period']),u*u*(3-2*u))
    else:
     u=min((t-4.8)/1.5,1.);z=spherical(env.standing,task.native_z[name],u*u*(3-2*u))
   elif t<12.:z=spherical(last,env.standing,min((t-10.5)/1.5,1.))
   else:z=env.standing
   if t<10.5:last=z.clone()
   x=task.features().clone();prev=task.last_command.clone();task.step(z/16)
   # Native continuation only: exclude clock-chosen entry/exit labels.
   # Physical continuation/hold states are identifiable by actual proprio/last-z.
   if (6.5<=t<10.4 or t>=13.5) and not task.boundary:
    labels.append(dict(x=x[0].cpu().tolist(),raw=(z/16)[0].cpu().tolist(),previous_z=prev[0].cpu().tolist(),episode=label,native=name,time=t,scope='native physical continuation, training prior not PPO rollout'))
  row=task.save(out);summaries.append(row);runner.write(out/'summaries.json',summaries);print('NATIVE_QUALIFICATION',label,row['error_cm'],row['force_rms'],row['root_max_displacement'],row['boundary'],flush=True)
 # Invalid native trajectories excluded from init, but full physics retained.
 valid={r['label'] for r in summaries if not r['boundary']};labels=[r for r in labels if r['episode'] in valid]
 with gzip.open(RUN/'native-continuation-labels.json.gz','wt') as f:json.dump(labels,f)
 walk=next(r for r in summaries if r['label']=='06-motion_walk_task-contact');arm=next(r for r in summaries if r['label']=='02-goal_arm5-contact')
 result=dict(status='completed',native_candidates=len(proto['candidates']),qualified_episodes=len(cases),initialization_labels=len(labels),valid_episodes=sorted(valid),walk_usable=not walk['boundary'] and walk['root_max_displacement']>.1,local_usable=not arm['boundary'],old_actor_loaded=False,rl_decisions=0,supervised_updates=0)
 runner.write(RUN/'qualification-gate.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
