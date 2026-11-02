"""Ideal-force Student transfer, aligned decisions and tested visited-state queries.
No anchor, clock, activation flag, support truth or teacher hidden features in Student.
"""
import m11_ipc
import argparse,os,json,time,hashlib,subprocess,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 global ARGS,START
 p=argparse.ArgumentParser();p.add_argument('--kind');p.add_argument('--stage');p.add_argument('--gpu');p.add_argument('--output',type=Path);p.add_argument('--iterations',type=int);p.add_argument('--seed',type=int,default=303);p.add_argument('--max-seconds',type=float);p.add_argument('--resume');ARGS=p.parse_args();START=time.monotonic()
 assert not ARGS.output.exists();ARGS.output.mkdir();os.environ.update(CUDA_VISIBLE_DEVICES=ARGS.gpu,CLEAN_KIND='direct',BFM_RECOVERY_OUTPUT=str(ARGS.output),MUJOCO_GL='disable');os.chdir(ARGS.output)
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 write(ARGS.output/'launch.json',dict(commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(ARGS).items()}))
 import m10_stage,m10_physics
 m10_stage.run_study=study;m10_physics.main()
def study(env,model,contract,out):
 import numpy as np,torch
 from clean_env import Task
 from clean_policy import Expert,direct_command
 from locomotion_rollout import spherical
 from m11_runtime_speed import install
 write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 plan=json.loads(Path(__file__).with_name('transfer_protocol.json').read_text());torch.manual_seed(plan['student']['seed']);task=Task(env,model,contract,404);task.reset();dev=env.standing.device;summary=[]
 # Assert the actual BFM inputs, not just a policy feature name.
 cfg=env.base.config.obs.obs_auxiliary['history_actor'];assert set(cfg)=={'base_ang_vel','projected_gravity','dof_pos','dof_vel','actions'};assert set(env.keys)=={'state','last_action','history_actor'}
 assert env.obs['state'].shape[-1]==64 and env.obs['last_action'].shape[-1]==29 and env.obs['history_actor'].shape[-1]==372
 audit=dict(bfm_actor_keys=env.keys,state='29 encoder positions +29 velocities +IMU projected gravity3/angular velocity3',last_action='29 actually executed previous BFM actions',history_actor={k:int(v) for k,v in cfg.items()},student='normalized465 proprio/action + last100 measured force XYZ in current body frame + own last256 legal command/16 + fixed right-wrist link/local-point4',excluded=['privileged_state','anchor','partner velocity','active','time','root world translation','support truth','teacher command history'],force_source='ideal simulated Tether external force, no G1 force sensor capability claim',contact_location='known fixed wrist sensor mounting; constant [1,0,0,0], not contact activation')
 write(out/'observation-audit.json',audit)
 def sinput():
  task.prepare();R=env.sim.data.xmat[1].reshape(3,3)
  hist=task.h.features(R)[-100:,:3].reshape(-1)
  return torch.cat([task.pro.clamp(-10,10),torch.tensor(hist,dtype=torch.float32,device=dev)[None],task.last_command/16,torch.tensor([[1.,0.,0.,0.]],device=dev)],-1)
 def load_teacher(key):
  identity=plan['teachers'][key];assert sha(identity['path'])==identity['sha256'];c=torch.load(identity['path'],map_location=dev,weights_only=True);assert c['kind']=='direct';net=Expert(c['input_dim'],'direct').to(dev);net.load_state_dict(c['state_dict']);return net.eval().requires_grad_(False)
 def run(label,spec,teacher,student=None,takeover=None,mode='learned'):
  task.reset(spec,record=True,label=label,query_reference=mode=='m10_expert');samples=[];differences=[];takeover_seen=False
  while not task.done:
   t=float(env.tether.time);tx=task.features();sx=sinput();traw=teacher.distribution(tx).mean;raw=traw if student is None or (takeover is not None and t>=takeover) else student.distribution(sx).mean
   takeover_seen|=takeover is not None and t>=takeover
   # Permission isolation: this is the entire Student input. Teacher queries use
   # the robot's real executed action history and last_command, never oracle history.
   samples.append(dict(x=sx[0].cpu().tolist(),target=direct_command(traw,env.standing)[0].cpu().tolist(),time=t,obs={k:v.cpu().tolist() for k,v in env.obs.items() if k in env.keys},previous_z=task.last_command[0].cpu().tolist()))
   if student is not None:
    zt=direct_command(traw,env.standing);zs=direct_command(student.distribution(sx).mean,env.standing);qt=task.qof(model.act(env.obs,spherical(task.last_command,zt,.2),mean=True));qs=task.qof(model.act(env.obs,spherical(task.last_command,zs,.2),mean=True));at=model.act(env.obs,zt,mean=True);ass=model.act(env.obs,zs,mean=True)
    differences.append(dict(time=t,z_angle_deg=float(torch.rad2deg(torch.acos(((zt*zs).sum()/256).clamp(-1,1)))),first_tick_qpd_rms_deg=float(torch.rad2deg(qt-qs).square().mean().sqrt()),endpoint_action_rms=float((at-ass).square().mean().sqrt())))
   task.step(raw,mode=mode)
  result=task.save(out);result['takeover']=takeover;result['takeover_seen']=takeover_seen
  if takeover is not None:
   rr=[r for r in task.physics if r['time']>=takeover and r['active']];result['tail_error_cm']=100*float(np.sqrt(np.mean([r['error']**2 for r in rr])));result['tail_force_rms']=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in rr])))
  if differences:write(out/(label+'-same-state.json'),differences)
  summary.append(result);write(out/'summaries.json',summary)
  with gzip.open(out/(label+'-queries.json.gz'),'wt') as f:json.dump(samples,f)
  print('TRANSFER_EVAL',label,result['error_cm'],result['force_rms'],result['boundary'],flush=True)
  return samples,result
 def fit(samples,net,steps,label):
  x=torch.tensor([r['x'] for r in samples],device=dev);z=torch.tensor([r['target'] for r in samples],device=dev)
  if net is None:
   torch.manual_seed(plan['student']['seed']);net=Expert(x.shape[-1],'direct').to(dev);net.center.copy_(x.mean(0));net.scale.copy_(x.std(0,unbiased=False).clamp_min(.1));net.actor[-1].bias.copy_(env.standing[0]/16)
  net.requires_grad_(False);net.actor.requires_grad_(True);net.train();opt=torch.optim.Adam(net.actor.parameters(),lr=plan['student']['lr']);curve=[]
  with torch.enable_grad():
   for step in range(steps):
    ix=torch.randint(len(x),(plan['student']['batch'],),device=dev);zz=direct_command(net.distribution(x[ix]).mean,env.standing);loss=(zz-z[ix]).square().mean();opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
    if step%200==0:curve.append(dict(step=step,loss=float(loss.detach()),gradient=gn))
  net.eval().requires_grad_(False);errors=[];angles=[];identity=[]
  for start in range(0,len(samples),128):
   ss=samples[start:start+128];ob={k:torch.tensor([r['obs'][k][0] for r in ss],device=dev) for k in env.keys};zi=z[start:start+128];zs=direct_command(net.distribution(x[start:start+128]).mean,env.standing);prev=torch.tensor([r['previous_z'] for r in ss],device=dev)
   qt=task.qof(model.act(ob,torch.cat([spherical(p[None],z[None],.2) for p,z in zip(prev,zi)]),mean=True));qs=task.qof(model.act(ob,torch.cat([spherical(p[None],z[None],.2) for p,z in zip(prev,zs)]),mean=True));errors.append(qt-qs);angles.append(torch.rad2deg(torch.acos(((zi*zs).sum(-1)/256).clamp(-1,1))))
   identity.append(float((model.act(ob,zi,mean=True)-model.act(ob,zi.clone(),mean=True)).abs().max()))
  ck=out/(label+'.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=x.shape[-1],kind='ideal-force-student',updates=steps,source='aligned direct Teacher decisions; no privileged Student inputs',runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()),ck)
  write(out/(label+'-training.json'),dict(samples=len(samples),updates=steps,curve=curve,first_tick_qpd_rms_deg=float(torch.rad2deg(torch.cat(errors)).square().mean().sqrt()),z_angle_mean_deg=float(torch.cat(angles).mean()),legal_z_action_identity_max=max(identity),checkpoint_sha256=sha(ck)))
  return net
 try:
  for key in ['initial','ppo15']:
   teacher=load_teacher(key);dataset=[]
   for name,spec in {**plan['dev'],**plan['regression']}.items():
    rows,result=run(key+'__source_'+name,spec,teacher)
    if result['boundary'] is None:dataset+=rows
   student=fit(dataset,None,plan['student']['steps'],key+'-student-r0')
   evaluated={}
   for name,spec in plan['dev'].items():evaluated[name]=run(key+'__student_r0_'+name,spec,teacher,student)
   # Test recovery from two actual Student prefixes before accepting visited labels.
   rows,r0=evaluated['single'];accepted=[]
   for t in [7.,10.]:
    _,rr=run(key+f'__teacher_takeover_{int(t)}',plan['single'],teacher,student,takeover=t)
    with gzip.open(out/(key+'__student_r0_single.json.gz'),'rt') as f:ep=json.load(f)
    tail=[r for r in ep['physics'] if r['time']>=t and r['active']];ecm=100*float(np.sqrt(np.mean([r['error']**2 for r in tail])));fn=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in tail])))
    good=rr['boundary'] is None and rr['tail_error_cm']<=ecm and rr['tail_force_rms']<=fn*1.05
    accepted.append(dict(takeover=t,recovery_effective=good,student_tail_error_cm=ecm,student_tail_force_rms=fn,teacher_tail_error_cm=rr['tail_error_cm'],teacher_tail_force_rms=rr['tail_force_rms']))
   write(out/(key+'-recovery.json'),accepted)
   if any(r['recovery_effective'] for r in accepted):
    cutoff=min(r['takeover'] for r in accepted if r['recovery_effective']);visited=[r for r in rows if r['time']>=cutoff]
    # A bounded one-round aggregation, never tune against heldout conditions.
    student=fit(dataset+visited*3,student,plan['student']['aggregation_steps'],key+'-student-r1')
   for name,spec in {**plan['dev'],**plan['regression'],**plan['student_probe'],**plan['final']}.items():
    run(key+'__student_final_'+name,spec,teacher,student)
    if name in plan['final'] or name in plan['student_probe']:run(key+'__teacher_final_'+name,spec,teacher)
  # One shared physical M10 reference on novel and stopping conditions.
  for name,spec in {**plan['final'],**plan['student_probe']}.items():run('m10__'+name,spec,teacher,mode='m10_expert')
  result=dict(status='completed',decisions=0,rl_updates=0,student_updated=True,bfm_updated=False,elapsed_s=time.monotonic()-START)
 except BaseException:
  import traceback
  (out/'failure.txt').write_text(traceback.format_exc());write(out/'result.json',dict(status='failed',decisions=0,elapsed_s=time.monotonic()-START));raise
 write(out/'result.json',result);return result
if __name__=='__main__':main()
