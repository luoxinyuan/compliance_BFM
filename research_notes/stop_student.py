"""Matched extra updates vs independent stop episodes; genuine no-force retraining."""
from pathlib import Path
import json,os,time,gzip,hashlib,subprocess,copy
import transfer_student_validated as runner
from stop_env import PLAN,phase,install
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/direct-transfer-20261103/A-prepare-01/job')
RUN=OLD.parents[1]/'stop-sensing-20261103'
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def study(env,model,contract,out):
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 from m11_runtime_speed import install as speed
 from locomotion_rollout import spherical
 install();from clean_env import Task
 runner.write(out/'runtime-speed.json',speed(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 task=Task(env,model,contract,707);task.reset();device=env.standing.device;summary=[];mode=runner.ARGS.stage
 assert set(env.keys)=={'state','last_action','history_actor'}
 assert set(env.base.config.obs.obs_auxiliary['history_actor'])=={'base_ang_vel','projected_gravity','dof_pos','dof_vel','actions'}
 runner.write(out/'input-audit.json',dict(bfm_keys=env.keys,student='465 proprio/actions + 300 measured force + 256 own z + 4 fixed mounting',no_force='725 dimensions, force columns physically removed; no external torque/contact truth',hardware='wrist force sensors unknown; encoder/IMU assumed; all corruptions uncalibrated simulation stress tests',excluded=['anchor','error','partner_velocity','time','active','support_truth','teacher_memory','tau_est']))
 def load(identity,noforce=False):
  p=Path(identity['path']);assert runner.sha(p)==identity['sha256'];ck=torch.load(p,map_location=device,weights_only=True);sd=ck['state_dict'];n=ck['input_dim']
  if noforce:
   keep=torch.cat([torch.arange(465),torch.arange(765,1025)]).to(device);sd={k:v.clone() for k,v in sd.items()}
   for key in ['center','scale']:sd[key]=sd[key][keep]
   for key in ['actor.0.weight','critic.0.weight']:sd[key]=sd[key][:,keep]
   n=725
  net=Expert(n,'direct').to(device);net.load_state_dict(sd);return net.eval().requires_grad_(False)
 def sx(noforce=False):
  task.prepare();R=env.sim.data.xmat[1].reshape(3,3);force=task.sensor.features(R)[-100:,:3].reshape(-1)
  x=torch.cat([task.pro.clamp(-10,10),torch.tensor(force,device=device,dtype=torch.float32)[None],task.last_command/16,torch.tensor([[1.,0,0,0]],device=device)],-1)
  assert x.shape[-1]==1025
  return torch.cat([x[:,:465],x[:,765:]],-1) if noforce else x
 def run(label,spec,teacher,student=None,noforce=False,sensor='ideal',collect=False,takeover=None):
  task.sensor_mode=sensor;task.sensor_seed=707;task.reset(spec,record=True,label=label);samples=[];diff=[]
  while not task.done:
   t=float(env.tether.time);tx=task.features();x=sx(noforce);traw=teacher.distribution(tx).mean;zt=direct_command(traw,env.standing)
   raw=traw if student is None or (takeover is not None and t>=takeover) else student.distribution(x).mean
   if collect:samples.append(dict(x=x[0].cpu().tolist(),target=zt[0].cpu().tolist(),time=t,episode=label,phase=phase(spec,t),obs={k:env.obs[k].cpu().tolist() for k in env.keys},previous_z=task.last_command[0].cpu().tolist()))
   if student is not None:
    zs=direct_command(student.distribution(x).mean,env.standing);at=model.act(env.obs,zt,mean=True);ass=model.act(env.obs,zs,mean=True)
    diff.append(dict(time=t,z_angle=float(torch.rad2deg(torch.acos(((zt*zs).sum()/256).clamp(-1,1)))),action_rms=float((at-ass).square().mean().sqrt())))
   task.step(raw)
  row=task.save(out);row.update(sensor=sensor,takeover=takeover)
  if takeover is not None:
   rows=[r for r in task.physics if r['time']>=takeover and r['active']];row['tail_error_cm']=100*float(np.sqrt(np.mean([r['error']**2 for r in rows])));row['tail_force_rms']=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in rows])))
  summary.append(row);runner.write(out/'summaries.json',summary)
  if samples:
   with gzip.open(out/(label+'-queries.json.gz'),'wt') as f:json.dump(samples,f)
  if diff:runner.write(out/(label+'-action-difference.json'),diff)
  print('STOP_STUDY',label,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
  return samples,row
 def olddata():
  rows=[]
  for f in sorted(OLD.glob('ppo15__source_*-queries.json.gz')):
   episode=f.name.split('-queries')[0];spec=read(OLD/(episode+'.json.gz'))['condition']
   for r in read(f):r.update(episode=episode,phase=phase(spec,r['time']));rows.append(r)
  assert len(rows)==824
  return rows
 def fit(data_old,data_new,noforce,label,seed):
  net=load(PLAN['r0']['ppo15'],noforce);torch.manual_seed(seed);net.actor.requires_grad_(True)
  def tensor(rows):
   x=torch.tensor([r['x'] for r in rows],device=device);x=torch.cat([x[:,:465],x[:,765:]],-1) if noforce else x
   return x,torch.tensor([r['target'] for r in rows],device=device)
  xo,zo=tensor(data_old);xn,zn=tensor(data_new) if data_new else (xo,zo)
  opt=torch.optim.Adam(net.actor.parameters(),lr=PLAN['student']['lr']);curve=[];steps=PLAN['student']['steps']
  with torch.enable_grad():
   for i in range(steps):
    # Exactly 128 examples/update; A draws all original, B balanced original/new.
    io=torch.randint(len(xo),(64,),device=device);inn=torch.randint(len(xn),(64,),device=device)
    x=torch.cat([xo[io],xn[inn]]);z=torch.cat([zo[io],zn[inn]]);pred=direct_command(net.distribution(x).mean,env.standing)
    loss=(pred-z).square().mean();opt.zero_grad();loss.backward();g=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
    if i%200==0:curve.append(dict(step=i,loss=float(loss.detach()),gradient=g))
  net.eval().requires_grad_(False);ck=out/(label+'.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=len(net.center),kind='no-force-student' if noforce else 'ideal-force-student',updates=steps,seed=seed,runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()),ck)
  counts={}
  for r in data_old+data_new:counts[r['phase']]=counts.get(r['phase'],0)+1
  runner.write(out/(label+'-training.json'),dict(updates=steps,seed=seed,old_labels=len(data_old),new_labels=len(data_new),independent_episodes=sorted(set(r['episode'] for r in data_old+data_new)),phase_counts=counts,curve=curve,sha256=runner.sha(ck),initial=PLAN['r0']['ppo15'],force_columns_deleted=noforce))
  return net
 teacher=load(PLAN['teachers']['ppo15']);other=load(PLAN['teachers']['initial']);steps=0
 if mode=='prepare':
  data_new=[]
  for name,spec in PLAN['stop_sources'].items():
   rows,result=run('source_'+name,spec,teacher,collect=True)
   if result['boundary'] is None:data_new+=rows
  with gzip.open(out/'stop-dataset.json.gz','wt') as f:json.dump(data_new,f)
  runner.write(out/'dataset-identity.json',dict(teacher=PLAN['teachers']['ppo15'],sha256=runner.sha(out/'stop-dataset.json.gz'),samples=len(data_new),quality='own closed-loop teacher data; retain phase quality, no claim perfect stopping or universal recovery'))
  for name in ['early_stop','late_stop']:
   run('reference_initial_'+name,PLAN['dev'][name],other)
  r0=load(PLAN['r0']['ppo15'])
  for name,spec in {**PLAN['dev'],**PLAN['regression'],**PLAN['final']}.items():
   run('teacher_'+name,spec,teacher);run('r0_'+name,spec,teacher,r0)
  for name in ['early_stop','late_stop']:
   run('takeover_'+name,PLAN['dev'][name],teacher,r0,takeover=PLAN['dev'][name]['stop_time']+.3)
  data_old=olddata()
  for seed in PLAN['student']['seeds']:
   for arm,extra in [('A',[]),('B',data_new)]:
    label=f'{arm}{seed}';net=fit(data_old,extra,False,label,seed);steps+=PLAN['student']['steps']
    for name,spec in {**PLAN['dev'],**PLAN['regression'],**PLAN['final']}.items():run(label+'_'+name,spec,teacher,net)
 elif mode=='final':
  ledger=json.loads((RUN/'compute-ledger.json').read_text());source=next(Path(r['path'])/'job' for r in ledger['attempts'] if r['arm']=='A' and r.get('result',{}).get('status')=='completed')
  data_new=read(source/'stop-dataset.json.gz');data_old=olddata()
  for seed in PLAN['student']['seeds']:
   label=f'N{seed}';net=fit(data_old,data_new,True,label,seed);steps+=PLAN['student']['steps']
   for name,spec in {**PLAN['dev'],**PLAN['regression'],**PLAN['final']}.items():run(label+'_'+name,spec,teacher,net,noforce=True)
  # Fixed first update seed, never choose by final test.
  bpath=source/'B505.pt';covered=load(dict(path=str(bpath),sha256=runner.sha(bpath)));r0=load(PLAN['r0']['ppo15'])
  for label,net in [('r0',r0),('B505',covered)]:
   for noise in ['stress','strong']:
    for name in ['early_stop','stop_unseen_late','no_contact']:
     spec={**PLAN['dev'],**PLAN['regression'],**PLAN['final']}[name];run(label+'_'+noise+'_'+name,spec,teacher,net,sensor=noise)
 else:raise ValueError(mode)
 result=dict(status='completed',decisions=0,student_optimizer_updates=steps,bfm_updated=False,elapsed_s=time.monotonic()-runner.START)
 runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
