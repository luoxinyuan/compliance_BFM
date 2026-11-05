"""Frozen locked-selection physical evaluation. Never select models here."""
import json,time,copy
from pathlib import Path
import transfer_student_validated as runner
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105')
def study(env,model,contract,out):
 import numpy as np,torch
 from clean_policy import Expert
 from geometry_policy import expand
 from transition_env import Task,install_frame
 from locomotion_rollout import spherical
 from m11_runtime_speed import install
 install_frame();import geometry_env,reach_contact;geometry_env.install_contact=reach_contact.install
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 plan=json.loads(Path(__file__).with_name('prior_training.json').read_text());manifest=json.loads(Path(runner.ARGS.resume).read_text());nets={}
 for name,d in manifest.items():
  assert runner.sha(d['path'])==d['sha256'];n=Expert(1088,'direct').to(env.standing.device);expand(n,torch.load(d['path'],map_location=env.standing.device,weights_only=True));nets[name]=n.eval().requires_grad_(False)
 runner.write(out/'identities.json',manifest);task=Task(env,model,contract,0);rows=[];specs={}
 # Fixed before the matched PPO. 8/20/55cm, onset4.55, new9deg yaw.
 for name,dist,vel in [('new8',.08,.11),('new20',.20,.09),('new55',.55,.13)]:
  ramp=1.5*dist/vel;specs[name]=dict(plan['dev']['walk'],distance=dist,ramp=ramp,onset=4.55,release=4.55+ramp+2,end=4.55+ramp+6)
 for name in ['arm5','walk','early','late','no_contact']:specs['reg_'+name]=copy.deepcopy(plan['dev'][name])
 for name,angle in [('newleft',9.),('newright',-9.)]:specs[name]=dict(plan['dev']['left'],rotation_deg=angle,onset=4.55,ramp=8.05,release=14.6,end=18.6)
 specs['new8_armready']=dict(specs['new8'],onset=6.55,release=specs['new8']['release']+2,end=specs['new8']['end']+2,arm_ready=True)
 specs['new20_armready']=dict(specs['new20'],onset=6.55,release=specs['new20']['release']+2,end=specs['new20']['end']+2,arm_ready=True)
 # Common5N regression separates10N training-domain from original light-contact behavior.
 specs['reg5N_small']=dict(plan['dev']['arm5'],total_cap=5.)
 specs['reg5N_walk']=dict(plan['dev']['walk'],total_cap=5.)
 native=json.loads((RUN/'qualify-02/native-sources.json').read_text());goal=torch.tensor(native['goals']['0.05'],device=env.standing.device)
 for name,s in specs.items():
  for who,net in nets.items():
   if time.monotonic()-runner.START>runner.ARGS.max_seconds-60:raise TimeoutError('finite final evaluation')
   label=who+'__'+name;task.reset(s,record=True,label=label)
   while not task.done:
    t=float(env.tether.time)
    if s.get('arm_ready') and t<6.-1e-7:
     progress,_=reach_contact.smooth(t,4.,1.5);raw=spherical(env.standing,goal,.6*progress)/16
    else:raw=net.distribution(task.features()).mean
    task.step(raw)
   row=task.save(out);rows.append(row);runner.write(out/'summaries.json',rows);print('PRIOR_FINAL',label,row['error_cm'],row['force_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
 return dict(status='completed',episodes=len(rows),rl_decisions=0,supervised_updates=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
