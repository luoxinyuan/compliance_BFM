"""Coverage baseline, frozen physical comparisons; no online force-to-motion rules."""
from pathlib import Path
import json,gzip,os,time,subprocess
import transfer_student_validated as runner
from m39a_env import PLAN,install_frame
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
R=P/'bridge-v2-owned/runs/m39a-demand-20261104'
def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)
def gz(p,x):
 with gzip.open(p,'wt') as f:json.dump(x,f)
def study(env,model,contract,out):
 import torch,numpy as np,mujoco
 from m39a_env import Task
 from clean_policy import Expert,direct_command
 from geometry_policy import expand
 from locomotion_rollout import spherical
 from wholebody_physics import goal_observation
 from m11_runtime_speed import install
 install_frame();runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False);torch.manual_seed(runner.ARGS.seed)
 task=Task(env,model,contract,runner.ARGS.seed);task.reset(PLAN['dev']['straight']);device=env.standing.device
 def load(ident):
  assert runner.sha(ident['path'])==ident['sha256'];ck=torch.load(ident['path'],map_location=device,weights_only=True);n=Expert(1088,'direct').to(device);expand(n,ck);return n.eval().requires_grad_(False)
 nets={k:load(v) for k,v in PLAN['teachers'].items()};summary=[];alllabels=[]
 def run(label,spec,net,labels=False,source=False,sampled=False):
  task.reset(spec,record=True,label=label);goal=None;last_a=0.;rt=None;rv=0.;ss=[]
  if source:
   q=env.sim.data.qpos.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(spec['rotation_deg']);goal=model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))
  g=torch.Generator(device=device).manual_seed(runner.ARGS.seed)
  while not task.done:
   t=float(env.tether.time);x=task.features();prev=task.last_command.clone();dist=net.distribution(x);raw=dist.mean
   if sampled:raw=raw+dist.stddev*torch.randn(raw.shape,device=device,generator=g)
   z=direct_command(raw,env.standing)
   if source:
    if env.tether.active:a=float(np.clip((env.tether.geometry_target_yaw-task.geometry_yaw)/np.deg2rad(spec['rotation_deg']),0,1));last_a=a
    elif last_a>0:
     if rt is None:rt=t;rv=last_a
     a=rv*max(0.,1-(t-rt)/1.5)
    else:a=0.
    z=spherical(z,goal,a);raw=z/16
   c=len(task.controls);obs={k:v.cpu().tolist() for k,v in env.obs.items() if k in env.keys};task.step(raw)
   if labels:
    cc=task.controls[c:];count=len(cc);cc=cc+[cc[-1]]*(5-count);ss.append(dict(x=x[0].cpu().tolist(),z=z[0].cpu().tolist(),raw=raw[0].cpu().tolist(),time=t,episode=label,obs=obs,previous_z=prev[0].cpu().tolist(),sequence=cc,valid_steps=count))
  row=task.save(out);summary.append(row);runner.write(out/'summaries.json',summary)
  if labels:gz(out/(label+'-labels.json.gz'),ss)
  print('M39_PHYSICS',label,row['error_cm'],row['sum_force_norm_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
  return row,ss
 if runner.ARGS.stage=='prepare':
  for key,spec in PLAN['dev'].items():
   for name in ['D2','T30']:
    wanted=(name=='D2' and key in ['pair_left','pair_right']) or (name=='T30' and key in ['zero','pair_noforce','straight','early','late','local','no_contact','pair_translate'])
    row,ss=run(name+'_'+key,spec,nets[name],labels=wanted)
    if wanted and row['boundary'] is None:alllabels+=ss
  for key in ['small_left','small_right']:
   row,ss=run('source_'+key,PLAN['dev'][key],nets['T30'],labels=True,source=True)
   b=next(r for r in summary if r['label']=='T30_'+key)
   ok=row['boundary'] is None and row['actual_torso_turn_deg']*PLAN['dev'][key]['rotation_deg']>0 and row['error_cm']<b['error_cm'] and row['sum_force_norm_rms']<b['sum_force_norm_rms']
   runner.write(out/(key+'-source-quality.json'),dict(accepted=ok,source=row,baseline=b))
   if ok:alllabels+=ss
  for key in ['zero','pair_left','pair_right','sequence']:run('D2_sampled_'+key,PLAN['dev'][key],nets['D2'],sampled=True)
  for key in ['zero','pair_left','pair_right','straight']:run('D0_'+key,PLAN['dev'][key],nets['D0'])
  gz(out/'coverage-data.json.gz',alllabels);runner.write(out/'dataset.json',dict(samples=len(alllabels),episodes=sorted(set(r['episode'] for r in alllabels)),sha256=runner.sha(out/'coverage-data.json.gz'),scope='actual full source trajectories only; no unsupported combined labels; not broad recovery evidence',std=float(nets['D2'].log_std.exp().mean())))
 elif runner.ARGS.stage=='train':
  data=read(Path(runner.ARGS.resume)/'coverage-data.json.gz');data=[r for r in data if not r['episode'].startswith('D2_') or (all(c['post_active'] for c in r['sequence']) and sum(v*v for v in r['x'][1041:1050])**.5>.02)];runner.write(out/'training-data.json',dict(samples=len(data),episodes=sorted(set(r['episode'] for r in data)),filter='D2 rotation only active loaded domain; no D2 no-demand/release labels'));net=nets['D2'];net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=PLAN['training']['supervised_lr']);x=torch.tensor([r['x'] for r in data],device=device);z=torch.tensor([r['z'] for r in data],device=device);groups=[torch.tensor([i for i,r in enumerate(data) if r['episode']==e],device=device) for e in sorted(set(r['episode'] for r in data))];curve=[]
  # Same legal-z targets; enough q_PD qualification is in the source and prior study.
  for it in range(PLAN['training']['supervised_updates']):
   ii=torch.cat([g[torch.randint(len(g),(16,),device=device)] for g in groups])
   with torch.enable_grad():
    pred=direct_command(net.distribution(x[ii]).mean,env.standing);loss=(pred-z[ii]).square().mean();opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
   runner.write(out/'progress.json',dict(updates=it+1))
   if it%100==0 or it==PLAN['training']['supervised_updates']-1:curve.append(dict(update=it+1,loss=float(loss),gradient=gn));runner.write(out/'learning-curve.json',curve)
  ck=out/'coverage.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=1088,kind='direct',updates=it+1,source_student_online=False,initial_sha=PLAN['teachers']['D2']['sha256'],runtime=subprocess.check_output(['git','-C',str(runner.ROOT),'rev-parse','HEAD'],text=True).strip()),ck);runner.write(out/'model.json',dict(path=str(ck),sha256=runner.sha(ck)));net.eval().requires_grad_(False)
  for key,spec in PLAN['dev'].items():run('coverage_'+key,spec,net)
 elif runner.ARGS.stage=='final':
  config=read(runner.ARGS.resume)
  for name,ident in config['models'].items():
   net=load(ident)
   for key in config['keys']:run(name+'_'+key,PLAN['sealed'][key],net)
 else:raise ValueError(runner.ARGS.stage)
 result=dict(status='completed',decisions=0,expert_supervised_updates=PLAN['training']['supervised_updates'] if runner.ARGS.stage=='train' else 0,student_optimizer_updates=0,episodes=len(summary),elapsed_s=time.monotonic()-runner.START)
 runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
