"""Frozen Teacher30 transfer and bounded local recovery aggregation; original physical runtime."""
from pathlib import Path
import copy,gzip,json,os,time,subprocess
import transfer_student_validated as runner
from stop_env import install
PLAN=json.loads(Path(__file__).with_name('t30_protocol.json').read_text())
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/t30-transfer-20261103';OLD=P/'bridge-v2-owned/runs/direct-transfer-20261103/A-prepare-01/job'
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def write_gz(p,x):
 with gzip.open(p,'wt') as f:json.dump(x,f)
def study(env,model,contract,out):
 import numpy as np,torch
 from clean_policy import Expert,direct_command
 from locomotion_rollout import spherical
 from m11_runtime_speed import install as speed
 install();from clean_env import Task
 runner.write(out/'runtime-speed.json',speed(env));torch.set_num_threads(1);torch.set_grad_enabled(False);task=Task(env,model,contract,805);task.reset();device=env.standing.device;summary=[];updates=0
 assert set(env.keys)=={'state','last_action','history_actor'}
 assert set(env.base.config.obs.obs_auxiliary['history_actor'])=={'base_ang_vel','projected_gravity','dof_pos','dof_vel','actions'}
 def load(identity,noforce=False):
  path=Path(identity['path']);assert runner.sha(path)==identity['sha256'];ck=torch.load(path,map_location=device,weights_only=True);sd={k:v.clone() for k,v in ck['state_dict'].items()};n=ck['input_dim']
  if noforce:
   assert n==1025;keep=torch.cat([torch.arange(465),torch.arange(765,1025)]).to(device)
   for k in ['center','scale']:sd[k]=sd[k][keep]
   for k in ['actor.0.weight','critic.0.weight']:sd[k]=sd[k][:,keep]
   n=725
  net=Expert(n,'direct').to(device);net.load_state_dict(sd);net.eval().requires_grad_(False);return net
 teacher=load(PLAN['teachers']['t30']);oldteacher=load(PLAN['teachers']['ppo15']);r0=load(PLAN['r0']['ppo15'])
 runner.write(out/'identities.json',dict(teacher=PLAN['teachers']['t30'],oldteacher=PLAN['teachers']['ppo15'],r0=PLAN['r0']['ppo15'],dims=dict(teacher=len(teacher.center),student=len(r0.center)),student_observation='465 deployed proprio/actually executed actions,300 actual past force,256 own z,4 fixed wrist mounting; no clock/anchor/active/root translation/teacher state',normalization='checkpoint center/scale unchanged; no-force physically removes force cols and corresponding normalizers',rate='200Hz physics;50Hz BFM/PD and force samples;10Hz high-level commands with original5-substep slerp'))
 def sx(noforce=False):
  task.prepare();R=env.sim.data.xmat[1].reshape(3,3);force=task.sensor.features(R)[-100:,:3].reshape(-1)
  x=torch.cat([task.pro.clamp(-10,10),torch.tensor(force,device=device,dtype=torch.float32)[None],task.last_command/16,torch.tensor([[1.,0,0,0]],device=device)],-1)
  assert x.shape[-1]==1025
  return torch.cat([x[:,:465],x[:,765:]],-1) if noforce else x
 def run(label,spec,student=None,noforce=False,takeover=None,collect=False,prior=None,filter_tau=None):
  task.sensor_mode='ideal';task.reset(spec,record=True,label=label);samples=[];diff=[];actual=None;filtered=None;executor=teacher if prior is None else prior
  while not task.done:
   t=float(env.tether.time);tx=task.features();x=sx(noforce);traw=executor.distribution(tx).mean;zt=direct_command(traw,env.standing)
   raw=traw if student is None or (takeover is not None and t>=takeover-1e-8) else student.distribution(x).mean
   if takeover is not None and t>=takeover-1e-8 and actual is None:actual=t
   if filter_tau is not None:
    target=direct_command(raw,env.standing);filtered=spherical(task.last_command,target,1-np.exp(-.1/filter_tau));raw=filtered/16
   if collect:
    samples.append(dict(x=x[0].cpu().tolist(),target=zt[0].cpu().tolist(),time=t,episode=label,obs={k:env.obs[k].cpu().tolist() for k in env.keys},previous_z=task.last_command[0].cpu().tolist()))
   if student is not None:
    zs=direct_command(raw,env.standing);prev=task.last_command;at=model.act(env.obs,spherical(prev,zt,.2),mean=True);ast=model.act(env.obs,spherical(prev,zs,.2),mean=True)
    diff.append(dict(time=t,z_angle=float(torch.rad2deg(torch.acos(((zt*zs).sum()/256).clamp(-1,1)))),action_rms=float((at-ast).square().mean().sqrt()),qpd_rms_deg=float(torch.rad2deg(task.qof(at)-task.qof(ast)).square().mean().sqrt())))
   task.step(raw)
  row=task.save(out);row.update(takeover_requested=takeover,takeover_actual=actual,filter_tau=filter_tau);summary.append(row);runner.write(out/'summaries.json',summary)
  if collect:write_gz(out/(label+'-queries.json.gz'),samples)
  if diff:runner.write(out/(label+'-same-state.json'),diff)
  print('T30_RUN',label,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
  return samples,row
 def fit(base,old,new,extra,noforce,steps,label):
  nonlocal updates
  net=copy.deepcopy(base);torch.manual_seed(805);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=.0003)
  def tens(rows):
   x=torch.tensor([r['x'] for r in rows],device=device);x=torch.cat([x[:,:465],x[:,765:]],-1) if noforce else x
   return x,torch.tensor([r['target'] for r in rows],device=device)
  xo,zo=tens(old);xn,zn=tens(new);xe,ze=tens(extra) if extra else (None,None);curve=[]
  with torch.enable_grad():
   for i in range(steps):
    no,nn,ne=(32,64,32) if extra else ((32,96,0) if steps==3000 else (64,64,0))
    io=torch.randint(len(xo),(no,),device=device);inn=torch.randint(len(xn),(nn,),device=device);xx=[xo[io],xn[inn]];zz=[zo[io],zn[inn]]
    if extra:
     ix=torch.randint(len(xe),(ne,),device=device);xx.append(xe[ix]);zz.append(ze[ix])
    x=torch.cat(xx);z=torch.cat(zz);pred=direct_command(net.distribution(x).mean,env.standing);loss=(pred-z).square().mean();opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
    if i%200==0:curve.append(dict(step=i,loss=float(loss.detach()),grad=grad))
  net.eval().requires_grad_(False);f=out/(label+'.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=len(net.center),kind='direct',updates=steps,seed=805),f);updates+=steps
  runner.write(out/(label+'-training.json'),dict(updates=steps,cumulative_job_updates=updates,old=len(old),new=len(new),aggregated=len(extra),seed=805,sha256=runner.sha(f),curve=curve,no_force=noforce))
  return net
 def tail(ep,t):
  rs=[r for r in ep['physics'] if r['time']>=t];active=[r for r in rs if r['active']];post=[r for r in rs if r['time']>=ep['condition']['release']];root=np.array([r['root'] for r in post]);w=np.array([r['wrist'] for r in post]);tt=np.array([r['time'] for r in post]);path=float(np.linalg.norm(np.diff(root,axis=0),axis=1).sum()) if len(post)>1 else 0.;ws=float(np.sqrt(np.mean(np.sum((np.diff(w,axis=0)/np.diff(tt)[:,None])**2,axis=1)))) if len(post)>1 else 0.
  force=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in active]))) if active else 0.;error=float(np.sqrt(np.mean([r['error']**2 for r in active]))) if active else 0.
  return dict(force=force,error_m=error,release_root_path=path,release_wrist_speed=ws,score=force+3*error if active else path+ws,boundary=ep['boundary'])
 def recover(label,spec,net,baseline,points,window=False):
  refs=read(out/(baseline+'.json.gz'));accepted=[];records=[]
  for phase,t in points.items():
   ends=[t,t+.4] if window else [t];ok=[]
   for endpoint in ends:
    name=f'{label}_{phase}_{endpoint:.2f}';run(name,spec,net,takeover=endpoint);ep=read(out/(name+'.json.gz'));actual=next(r['takeover_actual'] for r in summary if r['label']==name)
    if actual is None:
     records.append(dict(phase=phase,requested=endpoint,accepted=False,boundary=ep['boundary'],reason='terminated before takeover'));ok.append(False);continue
    before=[r for r in ep['physics'] if r['time']<actual-1e-8];old=[r for r in refs['physics'] if r['time']<actual-1e-8];assert len(before)==len(old)
    parity=max(float(np.abs(np.array([r[k] for r in before])-np.array([r[k] for r in old])).max()) for k in ['qpos','qvel','force']);assert parity==0,parity
    a,b=tail(ep,actual),tail(refs,actual);valid=a['boundary'] is None and a['score']<b['score'] and a['release_root_path']<=b['release_root_path']+.03
    records.append(dict(phase=phase,requested=endpoint,actual=actual,teacher=a,continuation=b,accepted=valid,prefix_max_diff=parity));ok.append(valid)
   if window and all(ok):accepted.append([ends[0],ends[-1]])
  runner.write(out/(label+'-recovery.json'),dict(records=records,accepted_windows=accepted,scope='local bracketed region only; no universal recovery claim'));return accepted
 old=[]
 for f in sorted(OLD.glob('ppo15__source_*-queries.json.gz')):old+=read(f)
 assert len(old)==824
 def points(s):return dict(entry=6.,pre_deceleration=s['stop_time']-.5,deceleration=s['stop_time']+.2,hold=s['stop_time']+s['deceleration']+1,release=s['release']+.2)
 if os.environ['T30_ARM']=='A':
  data=[]
  for name,spec in PLAN['sources'].items():
   samples,row=run('source_'+name,spec,collect=True)
   if row['boundary'] is None:data+=samples
  write_gz(out/'teacher30-data.json.gz',data);runner.write(out/'data-ready.json',dict(sha256=runner.sha(out/'teacher30-data.json.gz'),labels=len(data),teacher=PLAN['teachers']['t30']))
  for name,spec in {**PLAN['dev'],**PLAN['regression']}.items():run('S0_'+name,spec,r0)
  for name in ['early_stop','late_stop']:recover('S0take_'+name,PLAN['dev'][name],r0,'S0_'+name,points(PLAN['dev'][name]))
  s1=fit(r0,old,data,[],False,4000,'S1warm')
  gathered=[]
  for name,spec in {**PLAN['dev'],**PLAN['regression']}.items():
   qs,_=run('S1warm_'+name,spec,s1,collect=True)
   if name in ['early_stop','late_stop']:
    regions=recover('S1take_'+name,spec,s1,'S1warm_'+name,points(spec),True)
    gathered += [q for q in qs if any(a-1e-8<=q['time']<=b+1e-8 for a,b in regions)]
  write_gz(out/'accepted-visited.json.gz',gathered);runner.write(out/'aggregation-scope.json',dict(labels=len(gathered),unique_episodes=sorted(set(x['episode'] for x in gathered)),selection='two representative endpoint takeovers per .4s window on actual S1warm continuation; no labels from outside accepted intervals'))
  matched=fit(s1,old,data,[],False,3000,'S1matched');s2=fit(s1,old,data,gathered,False,3000,'S2') if gathered else None
  if s2 is None:runner.write(out/'S2-not-run.json',dict(reason='no representative window met recovery evidence',extra_updates=0))
  models={'Teacher30':None,'S0':r0,'S1matched':matched}
  if s2 is not None:models['S2']=s2
  for name,spec in {**PLAN['dev'],**PLAN['regression'],**PLAN['final']}.items():
   for label,net in models.items():
    if label=='S0' and name not in PLAN['final']:continue
    if label=='Teacher30' and name in PLAN['sources']:continue
    run(label+'_'+name,spec,net)
  # Main no-force comparison uses S1warm at exactly the same 4000-update budget.
  for name,spec in PLAN['final'].items():run('S1warm_'+name,spec,s1)
 else:
  jobs=[p.parent for p in RUN.glob('A-*/job/data-ready.json')];assert len(jobs)==1;source=jobs[0];identity=json.loads((source/'data-ready.json').read_text());assert runner.sha(source/'teacher30-data.json.gz')==identity['sha256'];data=read(source/'teacher30-data.json.gz');runner.write(out/'source-identity.json',identity)
  n=fit(load(PLAN['r0']['ppo15'],True),old,data,[],True,4000,'N30')
  for name,spec in {**PLAN['dev'],**PLAN['regression'],**PLAN['final']}.items():run('N30_'+name,spec,n,noforce=True)
 result=dict(status='completed',decisions=0,student_optimizer_updates=updates,elapsed_s=time.monotonic()-runner.START);runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
