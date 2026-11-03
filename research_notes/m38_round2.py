"""M38: audit original data then action-aligned retention. No online goal controller."""
from pathlib import Path
import json,time,gzip,os
import transfer_student_validated as runner
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
SOURCE=P/'bridge-v2-owned/runs/contact-geometry-20261103/C-goal-10/job'

def study(env,model,contract,out):
 import numpy as np,torch
 from geometry_env import Task,PLAN
 from geometry_policy import expand
 from clean_policy import Expert,direct_command
 from m11_runtime_speed import install
 from locomotion_rollout import spherical
 plan=json.loads(Path(__file__).with_name('m38_protocol.json').read_text())['m38']
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False);torch.manual_seed(plan['seed'])
 task=Task(env,model,contract,3801);task.reset(PLAN['single']);dev=env.standing.device
 ident=PLAN['teachers']['initial'];assert runner.sha(ident['path'])==ident['sha256']
 ck=torch.load(ident['path'],map_location=dev,weights_only=True);net=Expert(task.features().shape[-1],'direct').to(dev);expand(net,ck);old=Expert(task.features().shape[-1],'direct').to(dev);expand(old,ck);old.eval().requires_grad_(False)
 badpath=SOURCE/'goal-initialized.pt';assert runner.sha(badpath)=='2059302f4b72c204b020309f316d17d46dd8cb7718d1ceaa655185eca592941e';bad=Expert(1088,'direct').to(dev);bad.load_state_dict(torch.load(badpath,map_location=dev,weights_only=True)['state_dict']);bad.eval().requires_grad_(False)
 data=[];manifest=[];source_summary=[]
 # Causal reference correction: same T30 behavior before there is any current
 # geometric demand; goal source uses current observed guide-frame rotation,
 # not future direction/time. This is initialization data, not learned execution.
 from wholebody_physics import goal_observation
 import mujoco,math
 for path in sorted(SOURCE.glob('retain_*-labels.json.gz')):
  with gzip.open(path,'rt') as f:rows=json.load(f)
  manifest.append(dict(path=str(path),sha256=runner.sha(path),samples=len(rows)));data+=rows
 def source_run(name,spec,rotation=False):
  task.reset(spec,record=True,label=name);samples=[];release_value=None;release_start=None;last_a=0.
  if rotation:
   q=env.sim.data.qpos.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(spec['rotation_deg']);zg=model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))
  while not task.done:
   xx=task.features();t=float(env.tether.time);zold=direct_command(old.distribution(xx).mean,env.standing);zz=zold
   if rotation:
    angle=env.tether.geometry_target_yaw-task.geometry_yaw
    if env.tether.active:
     a=float(np.clip(angle/np.deg2rad(spec['rotation_deg']),0,1));last_a=a
    elif last_a>0:
     if release_start is None:release_start=t;release_value=last_a
     a=release_value*max(0.,1-(t-release_start)/1.5)
    else:a=0.
    zz=spherical(zold,zg,a)
   samples.append(dict(x=xx[0].cpu().tolist(),z=zz[0].cpu().tolist(),time=t,episode=name,obs={k:v.cpu().tolist() for k,v in env.obs.items() if k in env.keys}))
   task.step(zz/16)
  r=task.save(out);source_summary.append(r);runner.write(out/'source-summaries.json',source_summary)
  path=out/(name+'-labels.json.gz')
  with gzip.open(path,'wt') as f:json.dump(samples,f)
  manifest.append(dict(path=str(path),sha256=runner.sha(path),samples=len(samples)));print('M38_SOURCE',name,r['error_cm'],r['sum_force_norm_rms'],r['actual_torso_turn_deg'],r['boundary'],flush=True)
  return samples,r
 for name in ['pair_left','pair_right']:
  rows,r=source_run('causal_'+name,PLAN['dev'][name],True)
  if r['boundary'] or r['actual_torso_turn_deg']*PLAN['dev'][name]['rotation_deg']<=0:
   runner.write(out/'source-failure.json',dict(reason='causal reference failed physical qualification; no new labels trained',row=r));return dict(status='source_boundary',decisions=0,expert_supervised_updates=0,student_optimizer_updates=0)
  data+=rows
 for name,stop in [('early',7.2),('late',9.8)]:
  spec=dict(PLAN['single'],stop_time=stop,deceleration=.8,release=stop+2.8,end=stop+7.8)
  rows,r=source_run('retain_stop_'+name,spec);data+=rows
 x=torch.tensor([r['x'] for r in data],device=dev);z=torch.tensor([r['z'] for r in data],device=dev);ob={k:torch.tensor([r['obs'][k][0] for r in data],device=dev) for k in env.keys}
 norm={k:v.detach() for k,v in model._normalize(ob).items()}
 def qpred(zz,ix):return task.qof(model._actor({k:v[ix] for k,v in norm.items()},zz,model.cfg.actor_std).mean.float())
 qs=torch.cat([qpred(z[i:i+128],slice(i,i+128)) for i in range(0,len(data),128)])
 # Verify differentiable frozen actor against actual public inference, without changing weights.
 with torch.enable_grad():
  zz=z[:2].clone().requires_grad_(True);qq=qpred(zz,slice(0,2));gr=torch.autograd.grad(qq.square().sum(),zz)[0]
 assert gr.norm()>0 and torch.isfinite(gr).all()
 parity=float((qpred(z[:8],slice(0,8))-task.qof(model.act({k:v[:8] for k,v in ob.items()},z[:8],mean=True))).abs().max());assert parity<1e-5
 groups={e:torch.tensor([i for i,r in enumerate(data) if r['episode']==e],device=dev) for e in sorted(set(r['episode'] for r in data))}
 audit=[]
 for name,ix in groups.items():
  for phase,mask in [('pre',torch.tensor([data[int(i)]['time']<4.25 for i in ix],device=dev)),('contact',torch.tensor([4.25<=data[int(i)]['time']<14.05 for i in ix],device=dev)),('release',torch.tensor([data[int(i)]['time']>=14.05 for i in ix],device=dev))]:
   ii=ix[mask]
   if not len(ii):continue
   for tag,n in [('T30',old),('old_goal_fit',bad)]:
    zz=direct_command(n.distribution(x[ii]).mean,env.standing);q=qpred(zz,ii);e=torch.rad2deg(q-qs[ii]);audit.append(dict(episode=name,phase=phase,model=tag,n=len(ii),qpd_rms_deg=float(e.square().mean().sqrt()),qpd_max_deg=float(e.abs().max()),z_angle_deg=float(torch.rad2deg(torch.acos(((zz*z[ii]).sum(-1)/256).clamp(-1,1))).mean())))
 # Find near-identical actual teacher observations with conflicting labels; time not an input.
 xn=old.normalized(x);dist=torch.cdist(xn,xn)/np.sqrt(x.shape[1]);dist.fill_diagonal_(float('inf'));conflicts=[]
 for a,ia in groups.items():
  for b,ib in groups.items():
   if a>=b:continue
   dd=dist[ia][:,ib];flat=int(dd.argmin());i=int(ia[flat//len(ib)]);j=int(ib[flat%len(ib)]);conflicts.append(dict(a=a,b=b,normalized_input_rms=float(dist[i,j]),time_a=data[i]['time'],time_b=data[j]['time'],qpd_target_rms_deg=float(torch.rad2deg(qs[i]-qs[j]).square().mean().sqrt()),z_angle_deg=float(torch.rad2deg(torch.acos((z[i]*z[j]).sum().div(256).clamp(-1,1))))))
 runner.write(out/'audit.json',dict(source=manifest,model_action_parity_max=parity,gradient_norm=float(gr.norm()),same_state=audit,nearest_cross_episode=conflicts,interpretation='Nearest-input test is diagnostic, not proof of unidentifiability; compare full input and actual qPD labels, not u representations.'))
 # Keep original data untouched. Train all joints on actual corresponding BFM observations.
 net.requires_grad_(False);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=plan['lr']);curve=[];gns=list(groups)
 for step in range(plan['round_updates']):
  if time.monotonic()-runner.START>runner.ARGS.max_seconds-120:raise TimeoutError('reserved run deadline')
  # Mixed groups in every batch, rather than taking sequential single-episode Adam steps.
  ix=torch.cat([g[torch.randint(len(g),(16,),device=dev)] for g in groups.values()])
  with torch.enable_grad():
   zz=direct_command(net.distribution(x[ix]).mean,env.standing);qq=qpred(zz,ix);lq=((qq-qs[ix])/np.deg2rad(1.)).square().mean();lz=(zz-z[ix]).square().mean();loss=lq+.1*lz
   opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
  if step%100==0:
   curve.append(dict(step=step,qpd_loss=float(lq),z_loss=float(lz),grad=grad));runner.write(out/'learning-curve.json',curve);print('M38_FIT',curve[-1],flush=True)
 net.eval().requires_grad_(False);path=out/'retained-r2.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=1088,kind='direct',updates=plan['round_updates'],source=manifest),path)
 runner.write(out/'retained-training.json',dict(updates=plan['round_updates'],sha256=runner.sha(path),samples=len(data),episodes=len(groups),curve=curve,loss=plan['loss']))
 summaries=[]
 def run(name,spec,n):
  task.reset(spec,record=True,label=name);diff=[]
  while not task.done:
   xx=task.features();raw=n.distribution(xx).mean;zz=direct_command(raw,env.standing);oz=direct_command(old.distribution(xx).mean,env.standing)
   if len(diff)<250:diff.append(dict(time=float(env.tether.time),qpd_rms_deg=float(torch.rad2deg(task.qof(model.act(env.obs,zz,mean=True))-task.qof(model.act(env.obs,oz,mean=True))).square().mean().sqrt()),x=xx[0].cpu().tolist(),z=zz[0].cpu().tolist(),obs={k:v.cpu().tolist() for k,v in env.obs.items() if k in env.keys}))
   task.step(raw)
  row=task.save(out);summaries.append(row);runner.write(out/'summaries.json',summaries)
  with gzip.open(out/(name+'-queries.json.gz'),'wt') as f:json.dump(diff,f)
  print('M38_EVAL',name,row['error_cm'],row['sum_force_norm_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
 for name in ['pair_left','pair_right','straight','local','pair_translate','pair_combined','no_contact']:run('retained_'+name,PLAN['dev'][name],net)
 for name,stop in [('early',7.2),('late',9.8)]:run('retained_'+name,dict(PLAN['single'],stop_time=stop,deceleration=.8,release=stop+2.8,end=stop+7.8),net)
 run('retained_pair_no_contact',dict(PLAN['dev']['pair_translate'],enabled=False,distance=0.,rotation_deg=0.),net)
 return dict(status='completed',decisions=0,expert_supervised_updates=plan['round_updates'],student_optimizer_updates=0,episodes=len(summaries),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
