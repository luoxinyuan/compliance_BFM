"""Small ideal-force-observation transfer from preselected frozen I; episode split."""
import json,time,gzip,subprocess
from pathlib import Path
import transfer_student_validated as runner
from adapter_evaluate import setup,episode,conditions

def study(env,model,contract,out):
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 task,I,goal=setup(env,model,contract);dev=env.standing.device;plan=json.loads(Path(__file__).with_name('prior_training.json').read_text())['dev'];specs={k:plan[k] for k in ['walk','early','late','arm15','zero','no_contact']};specs['devstop']=dict(plan['walk'],stop_time=6.8,deceleration=.8)
 cfg=env.base.config.obs.obs_auxiliary['history_actor'];assert set(cfg)=={'base_ang_vel','projected_gravity','dof_pos','dof_vel','actions'};assert set(env.keys)=={'state','last_action','history_actor'}
 assert env.obs['state'].shape[-1]==64 and env.obs['last_action'].shape[-1]==29 and env.obs['history_actor'].shape[-1]==372
 runner.write(out/'permissions.json',dict(proprio465='BFM normalized encoder positions29, velocities29, IMU gravity3/angularvelocity3; last_action29 and history372',last_z256='actual combined policy command history',contact918='100 measured force3 across3 slots in current body frame;18 known attachment/link/local mount encoding',excluded=['anchor/error','partner velocity','target yaw','clock','taskID','active','support truth','teacher base features/output'],hardware='ideal force simulation; G1 force sensor availability unknown',split='whole episodes; no random adjacent-frame train/test split',teacher='preselected I, not selected F/R',privileged_teacher='both F and R base read1088 privileged features; neither is a deployable Student'))
 groups=[];manifest=[];summ=[]
 for name,s in specs.items():
  r,rows=episode(task,I,goal,I,s,'teacher_train__'+name,out,collect=True);summ.append(r);runner.write(out/'summaries.json',summ)
  if r['boundary'] is None:groups.append(rows);manifest.append(dict(name=name,rows=len(rows),condition=s,teacher_error_cm=r['error_cm'],teacher_force_rms=r['force_rms'],scope='I own rollout; not arbitrary visited-state recovery labels'))
 samples=[r for g in groups for r in g];x=torch.tensor([r['x'] for r in samples],device=dev);z=torch.tensor([r['z'] for r in samples],device=dev);assert x.shape[1]==1639
 torch.manual_seed(4511);net=Expert(1639,'direct').to(dev);net.center.copy_(x.mean(0));net.scale.copy_(x.std(0,unbiased=False).clamp_min(.1));net.actor[-1].bias.copy_(env.standing[0]/16);net.requires_grad_(False);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=1e-4);curve=[];pools=[];start=0
 for g in groups:pools.append(torch.arange(start,start+len(g),device=dev));start+=len(g)
 with torch.enable_grad():
  for step in range(2000):
   if time.monotonic()-runner.START>runner.ARGS.max_seconds-300:raise TimeoutError('reserve actual closed-loop probe time')
   pool=pools[step%len(pools)];ix=pool[torch.randint(len(pool),(128,),device=dev)];zs=direct_command(net.distribution(x[ix]).mean,env.standing);loss=(zs-z[ix]).square().mean();opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
   if step%200==0:curve.append(dict(step=step+1,loss=float(loss),gradient=gn));runner.write(out/'curve.json',curve)
 net.eval().requires_grad_(False);ck=out/'ideal-force-student.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=1639,architecture='full',kind='ideal-force-student',updates=2000,seed=4511,teacher_sha=json.loads(Path(__file__).with_name('recovery_protocol.json').read_text())['models']['I']['sha256'],runtime=subprocess.check_output(['git','-C',str(runner.ROOT),'rev-parse','HEAD'],text=True).strip()),ck)
 errors=[];angles=[]
 for start in range(0,len(samples),64):
  ss=samples[start:start+64];zs=direct_command(net.distribution(x[start:start+len(ss)]).mean,env.standing);zt=z[start:start+len(ss)];ob={k:torch.tensor([r['bfm_obs'][k][0] for r in ss],device=dev) for k in env.keys};qt=task.qof(model.act(ob,zt,mean=True));qs=task.qof(model.act(ob,zs,mean=True));errors.append(qs-qt);angles.append(torch.rad2deg(torch.acos((zs*zt).sum(-1).div(256).clamp(-1,1))))
 runner.write(out/'training.json',dict(updates=2000,episodes=len(groups),samples=len(samples),manifest=manifest,checkpoint=str(ck),sha256=runner.sha(ck),same_state_endpoint_qpd_rms_deg=float(torch.rad2deg(torch.cat(errors)).square().mean().sqrt()),z_angle_mean_deg=float(torch.cat(angles).mean()),selection='fixed final2000, no test selection'))
 tests={k:v for k,v in conditions().items() if k in ['new9','new65','stop','continue','armready22']};tests['reg_no_contact']=plan['no_contact'];tests['reg_walk']=plan['walk']
 for name,s in tests.items():
  for who,n,student in [('I',I,False),('student',net,True)]:
   r,_=episode(task,I,goal,n,s,who+'__'+name,out,student=student);summ.append(r);runner.write(out/'summaries.json',summ)
 checks=[]
 for k in ['I','student']:
  pp=[r for r in summ if r['label'] in [k+'__stop',k+'__continue']];assert pp[0]['common_prefix_sha256']==pp[1]['common_prefix_sha256'];checks.append(dict(model=k,identical=True,sha256=pp[0]['common_prefix_sha256']))
 runner.write(out/'prefix-parity.json',checks)
 return dict(status='completed',student_updates=2000,expert_updates=0,rl=0,episodes=len(summ),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
