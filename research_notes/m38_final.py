"""Frozen M38 evaluation and causal-prefix diagnostics, no updates or model selection."""
from pathlib import Path
import json,time,gzip
import transfer_student_validated as runner
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m38-rotation-20261103'
def study(env,model,contract,out):
 import numpy as np,torch
 from geometry_env import Task,PLAN
 from geometry_policy import expand
 from clean_policy import Expert,direct_command
 from m11_runtime_speed import install
 from locomotion_rollout import spherical
 plan=json.loads(Path(__file__).with_name('m38_protocol.json').read_text())['m38'];torch.set_num_threads(1);torch.set_grad_enabled(False);runner.write(out/'runtime-speed.json',install(env));task=Task(env,model,contract,3801);task.reset(PLAN['single']);dev=env.standing.device
 identities={'T30':PLAN['teachers']['initial']['path'],'r1':str(R/'A-goal-02/job/retained-r1.pt'),'r2':str(R/'A-goal-03/job/retained-r2.pt')};nets={}
 for k,p in identities.items():
  ck=torch.load(p,map_location=dev,weights_only=True);net=Expert(1088,'direct').to(dev);expand(net,ck);nets[k]=net.eval().requires_grad_(False)
 runner.write(out/'models.json',{k:dict(path=p,sha256=runner.sha(p)) for k,p in identities.items()});summ=[];comp=[]
 sources=list((R/'A-goal-03/job').glob('*-labels.json.gz'))+list((P/'bridge-v2-owned/runs/contact-geometry-20261103/C-goal-10/job').glob('retain_*-labels.json.gz'));training=[]
 for path in sources:
  with gzip.open(path,'rt') as f:training+=json.load(f)
 tx=torch.tensor([r['x'] for r in training],device=dev);tn=nets['T30'].normalized(tx);off=[]
 for i in range(0,len(training),128):
  rows=training[i:i+128];x=tx[i:i+128];ob={k:torch.tensor([r['obs'][k][0] for r in rows],device=dev) for k in env.keys};zt=torch.tensor([r['z'] for r in rows],device=dev);qt=task.qof(model.act(ob,zt,mean=True))
  for k,n in nets.items():
   z=direct_command(n.distribution(x).mean,env.standing);q=task.qof(model.act(ob,z,mean=True));e=torch.rad2deg(q-qt).square().mean(-1).sqrt()
   for j,r in enumerate(rows):off.append(dict(episode=r['episode'],time=r['time'],model=k,qpd_rms_deg=float(e[j]),z_angle_deg=float(torch.rad2deg(torch.acos((z[j]*zt[j]).sum().div(256).clamp(-1,1))))))
 runner.write(out/'offline-same-state.json',off)
 def run(label,spec,who,prefix=False):
  task.reset(spec,record=True,label=label);diff=[];visited=[]
  while not task.done:
   x=task.features();visited.append(x[0].clone());raw=nets[who].distribution(x).mean
   if who=='T30' or (prefix and not env.tether.active and float(env.tether.time)<spec['onset']):raw=nets['T30'].distribution(x).mean
   # Compare all frozen policies at precisely the same observation/history.
   zz={k:direct_command(n.distribution(x).mean,env.standing) for k,n in nets.items()};qq={k:task.qof(model.act(env.obs,z,mean=True)) for k,z in zz.items()};qfirst={k:task.qof(model.act(env.obs,spherical(task.last_command,z,.2),mean=True)) for k,z in zz.items()}
   diff.append(dict(time=float(env.tether.time),active=bool(env.tether.active),relative_to_T30={k:dict(z_angle_deg=float(torch.rad2deg(torch.acos(((z*zz['T30']).sum()/256).clamp(-1,1)))),qpd_rms_deg=float(torch.rad2deg(qq[k]-qq['T30']).square().mean().sqrt()),qpd_max_deg=float(torch.rad2deg(qq[k]-qq['T30']).abs().max()),first_tick_qpd_rms_deg=float(torch.rad2deg(qfirst[k]-qfirst['T30']).square().mean().sqrt())) for k,z in zz.items() if k!='T30'}))
   task.step(raw)
  vx=torch.stack(visited);dist=torch.cdist(nets['T30'].normalized(vx),tn)/np.sqrt(vx.shape[-1]);near,ix=dist.min(-1);runner.write(out/(label+'-distribution.json'),[dict(time=d['time'],nearest_normalized_rms=float(v),nearest_episode=training[int(j)]['episode'],nearest_time=training[int(j)]['time']) for d,v,j in zip(diff,near,ix)]);r=task.save(out);summ.append(r);runner.write(out/'summaries.json',summ);runner.write(out/(label+'-same-state.json'),diff);print('M38_FINAL',label,r['error_cm'],r['sum_force_norm_rms'],r['actual_torso_turn_deg'],r['boundary'],flush=True)
 # Final combinations locked before r1. Fixed final r2 evaluated, no test-driven selection.
 for name,changes in plan['final'].items():
  spec=dict(PLAN['dev']['pair_left'],**changes);spec.update(release=spec['onset']+spec['ramp']+2,end=spec['onset']+spec['ramp']+7)
  for who in ['T30','r1','r2']:run('sealed_'+name+'_'+who,spec,who)
 # Current-contact removal: same moving virtual frame, but no physical connection.
 for sign in [1,-1]:
  spec=dict(PLAN['dev']['pair_left'],rotation_deg=15*sign,enabled=False)
  run('no_force_moving_frame_'+str(sign),spec,'r2')
 # One real complete common prefix intervention isolates early policy drift.
 for name in ['straight','pair_right']:
  run('T30_prefix_'+name,PLAN['dev'][name],'r2',True)
 # Same-state T30 query also on its own baseline trajectory.
 run('T30_straight',PLAN['single'],'T30')
 return dict(status='completed',decisions=0,expert_supervised_updates=0,student_optimizer_updates=0,episodes=len(summ),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
