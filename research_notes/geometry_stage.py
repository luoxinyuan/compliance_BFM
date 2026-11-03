"""Finite physical qualification/frozen final evaluation of contact geometry."""
from pathlib import Path
import os,json,time,math
import transfer_student_validated as runner

def study(env,model,contract,out):
 import numpy as np,torch
 from geometry_env import Task,PLAN
 from geometry_policy import expand
 from clean_policy import Expert,direct_command
 from m11_runtime_speed import install
 from m9_cyclic import sample
 from locomotion_rollout import spherical
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False);task=Task(env,model,contract,1203);obs=task.reset();identity=PLAN['teachers']['initial'];assert runner.sha(identity['path'])==identity['sha256'];ck=torch.load(identity['path'],map_location=env.standing.device,weights_only=True)
 net=Expert(obs.shape[-1],'direct').to(obs.device);expand(net,ck);net.eval().requires_grad_(False);summary=[]
 if runner.ARGS.resume:
  ck=torch.load(runner.ARGS.resume,map_location=obs.device,weights_only=True);net.load_state_dict(ck['state_dict'])
 def run(label,spec,mode):
  task.reset(spec,record=True,label=label)
  while not task.done:
   raw=net.distribution(task.features()).mean
   if mode.startswith('native_'):
    t=env.tether.time;alpha=max(0,min(1,(t-4)/1.5,(14-t)/1.5));phase=max(0,t-4)/task.refs['left']['period'];z=spherical(env.standing,sample(task.refs[mode[7:]],phase),alpha);raw=z/16
   task.step(raw,mode='standing' if mode=='standing' else 'learned')
  r=task.save(out);summary.append(r);runner.write(out/'summaries.json',summary);print('GEOMETRY',label,r['error_cm'],r['sum_force_norm_rms'],r['actual_pelvis_turn_deg'],r['actual_torso_turn_deg'],r['boundary'],flush=True)
 if runner.ARGS.stage=='prepare':
  for name in ['straight','left','right','pair_left','pair_right','pair_translate','pair_combined','no_contact']:
   for mode in ['standing','T30']:run(mode+'_'+name,PLAN['dev'][name],mode)
  for direction in ['left','right']:
   s=dict(PLAN['dev']['straight'],enabled=False);run('native_'+direction+'_free',s,'native_'+direction)
   run('native_'+direction+'_load',PLAN['dev']['pair_'+direction],'native_'+direction)
 else:
  for name,spec in {**PLAN['final'],**{k:PLAN['dev'][k] for k in ['straight','local','no_contact']}}.items():run('selected_'+name,spec,'learned')
 result=dict(status='completed',decisions=0,episodes=len(summary),elapsed_s=time.monotonic()-runner.START,student_optimizer_updates=0);runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
