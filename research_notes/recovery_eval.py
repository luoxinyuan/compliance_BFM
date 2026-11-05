"""Fixed-endpoint heldout evaluation, no test-based selection or further fitting."""
import json,time,copy,os
from pathlib import Path
import transfer_student_validated as runner
OLD=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105')
def study(env,model,contract,out):
 import numpy as np,torch
 from clean_policy import Expert
 from geometry_policy import expand
 from transition_env import Task,install_frame
 from locomotion_rollout import spherical
 from m11_runtime_speed import install
 install_frame();import geometry_env,reach_contact;geometry_env.install_contact=reach_contact.install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 plan=json.loads(Path(__file__).with_name('prior_training.json').read_text());cfg=json.loads(Path(__file__).with_name('recovery_protocol.json').read_text());manifest=json.loads(Path(runner.ARGS.resume).read_text());nets={}
 for name,d in manifest.items():
  assert runner.sha(d['path'])==d['sha256'];n=Expert(1088,'direct').to(env.standing.device);expand(n,torch.load(d['path'],map_location=env.standing.device,weights_only=True));nets[name]=n.eval().requires_grad_(False)
 runner.write(out/'identities.json',manifest);task=Task(env,model,contract,0);rows=[];specs={};s=cfg['sealed']
 for d,v in zip(s['distance_cm'],s['peak_speeds']):
  ramp=1.5*d/100/v;onset=s['onset'];specs['new'+str(d)]=dict(plan['dev']['walk'],distance=d/100,ramp=ramp,onset=onset,release=onset+ramp+2,end=onset+ramp+6)
 specs['new18_armready']=dict(specs['new18'],onset=6.65,release=specs['new18']['release']+2,end=specs['new18']['end']+2,arm_ready=True)
 for name,angle in zip(['newright','newleft'],s['yaw_deg']):specs[name]=dict(plan['dev']['left'],rotation_deg=angle,onset=4.65,ramp=8.05,release=14.7,end=18.7)
 specs['reg5N_small']=dict(plan['dev']['arm5'],total_cap=5.)
 specs['reg5N_walk']=dict(plan['dev']['walk'],total_cap=5.)
 native=json.loads((OLD/'qualify-02/native-sources.json').read_text());goal=torch.tensor(native['goals']['0.05'],device=env.standing.device)
 for name,spec in specs.items():
  for who,net in nets.items():
   if time.monotonic()-runner.START>runner.ARGS.max_seconds-40:raise TimeoutError('finite final evaluation')
   label=who+'__'+name;task.reset(spec,record=True,label=label)
   while not task.done:
    t=float(env.tether.time)
    if spec.get('arm_ready') and t<6.-1e-7:
     progress,_=reach_contact.smooth(t,4.,1.5);raw=spherical(env.standing,goal,.5*progress)/16
    else:raw=net.distribution(task.features()).mean
    task.step(raw)
   row=task.save(out);rows.append(row);runner.write(out/'summaries.json',rows);print('RECOVERY_FINAL',label,row['error_cm'],row['force_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
 return dict(status='completed',episodes=len(rows),rl_decisions=0,supervised_updates=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
