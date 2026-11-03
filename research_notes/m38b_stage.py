"""M38-B matched endpoint / temporal / qualified-recovery supervision.
All learned deployments use the unchanged privileged Direct interface.
"""
from pathlib import Path
import json,time,gzip,os,subprocess,hashlib
import transfer_student_validated as runner
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
R=P/'bridge-v2-owned/runs/m38b-rotation-20261103'
OLD=P/'bridge-v2-owned/runs/m38-rotation-20261103'
SOURCE=P/'bridge-v2-owned/runs/contact-geometry-20261103/C-goal-10/job'
CONFIG=Path(__file__).with_name('m38b_protocol.json')
def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)
def gz(p,x):
 with gzip.open(p,'wt') as f:json.dump(x,f)
def dev_specs(plan):
 d={k:dict(plan['dev'][k]) for k in ['pair_left','pair_right','straight','local','pair_translate','pair_combined','no_contact']}
 for n,t in [('early',7.2),('late',9.8)]:d[n]=dict(plan['single'],stop_time=t,deceleration=.8,release=t+2.8,end=t+7.8)
 d['pair_no_contact']=dict(plan['dev']['pair_translate'],enabled=False,distance=0.,rotation_deg=0.)
 return d

def study(env,model,contract,out):
 import numpy as np,torch,mujoco
 from geometry_env import Task,PLAN
 from geometry_policy import expand
 from clean_policy import Expert,direct_command
 from locomotion_rollout import spherical
 from wholebody_physics import goal_observation
 from m11_runtime_speed import install
 cfg=read(CONFIG);runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 task=Task(env,model,contract,3801);task.reset(PLAN['single']);device=env.standing.device
 ident=PLAN['teachers']['initial'];assert runner.sha(ident['path'])==ident['sha256']
 ck=torch.load(ident['path'],map_location=device,weights_only=True)
 def load(path=None):
  n=Expert(1088,'direct').to(device);expand(n,ck if path is None else torch.load(path,map_location=device,weights_only=True));return n.eval().requires_grad_(False)
 old=load();summary=[]
 def source_setup(spec):
  if not spec.get('rotation_deg'):return None
  q=env.sim.data.qpos.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(spec['rotation_deg'])
  return model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))
 def rollout(name,spec,net=None,source=False,takeover=None,save_labels=False):
  task.reset(spec,record=True,label=name);goal=source_setup(spec) if source else None;last_a=0.;rt=None;rv=0.;samples=[]
  while not task.done:
   t=float(env.tether.time);x=task.features();prev=task.last_command.clone();obs={k:v.cpu().tolist() for k,v in env.obs.items() if k in env.keys}
   if net is not None and (takeover is None or t<takeover-1e-8):raw=net.distribution(x).mean;z=direct_command(raw,env.standing)
   else:
    z=direct_command(old.distribution(x).mean,env.standing)
    if goal is not None:
     if env.tether.active:a=float(np.clip((env.tether.geometry_target_yaw-task.geometry_yaw)/np.deg2rad(spec['rotation_deg']),0,1));last_a=a
     elif last_a>0:
      if rt is None:rt=t;rv=last_a
      a=rv*max(0.,1-(t-rt)/1.5)
     else:a=0.
     z=spherical(z,goal,a)
    raw=z/16
   c=len(task.controls);task.step(raw)
   if save_labels and (takeover is None or t>=takeover-1e-8):
    cc=task.controls[c:];count=len(cc);cc=cc+[cc[-1]]*(5-count);samples.append(dict(x=x[0].cpu().tolist(),z=z[0].cpu().tolist(),time=t,episode=name,obs=obs,previous_z=prev[0].cpu().tolist(),sequence=cc,valid_steps=count))
  row=task.save(out);summary.append(row);runner.write(out/'summaries.json',summary)
  if save_labels:gz(out/(name+'-labels.json.gz'),samples)
  print('M38B_PHYSICS',name,row['error_cm'],row['sum_force_norm_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
  return row,samples
 if runner.ARGS.stage=='prepare':
  data=[];manifest=[];alignment=[]
  paths=sorted(SOURCE.glob('retain_*-labels.json.gz'))+sorted((OLD/'A-goal-03/job').glob('causal_*-labels.json.gz'))+sorted((OLD/'A-goal-03/job').glob('retain_stop_*-labels.json.gz'))
  assert len(paths)==10
  for p in paths:
   labels=read(p);epath=Path(str(p).replace('-labels.json.gz','.json.gz'));ep=read(epath);controls=ep['control'];bytime={round(c['time'],6):i for i,c in enumerate(controls)}
   for r in labels:
    idx=bytime[round(r['time'],6)];cc=controls[idx:idx+5];count=len(cc);assert 1<=count<=5;cc=cc+[cc[-1]]*(5-count)
    prev=np.array(ep['decisions'][idx//5]['obs'][565:821])*16
    # Full input includes last executed z at fixed documented slots, not oracle memory.
    assert np.max(np.abs(prev-np.array(r['x'][565:821])*16))<1e-6
    assert max(np.max(np.abs(np.array(r['obs'][k])-cc[0]['bfm_obs'][k])) for k in env.keys)<1e-6
    r.update(previous_z=prev.tolist(),sequence=cc,valid_steps=count);data.append(r)
   manifest.append(dict(path=str(p),sha256=runner.sha(p),physical_path=str(epath),physical_sha256=runner.sha(epath),samples=len(labels)))
  assert len(data)==1554
  gz(out/'base-data.json.gz',data)
  # Reproduce only three already physically qualified exact takeovers. Labels
  # describe actual expert recovery trajectories, not all unexecuted r2 queries.
  r2path=OLD/'A-goal-03/job/retained-r2.pt';assert runner.sha(r2path)=='310f744427bbe48ab914699044d678fe97d729b6bac96b98173b02dd6cba47f1';r2=load(r2path);visited=[];recover=[]
  for name,take in [('pair_left',7.),('pair_right',7.),('straight',5.)]:
   row,ss=rollout('recovery_'+name,PLAN['dev'][name],r2,source=name!='straight',takeover=take,save_labels=True)
   saved=read(OLD/'A-goal-03/job'/('retained_'+name+'.json.gz'))['physics'];now=task.physics
   pairs=[(a,b) for a,b in zip(saved,now) if a['force_time']<take-1e-8]
   dif={k:max(float(np.max(np.abs(np.array(a[k])-b[k]))) for a,b in pairs) for k in ['qpos','qvel','force']};assert max(dif.values())==0,dif
   def tail(rr):
    v=[a for a in rr if a['active'] and a['force_time']>=take-1e-8]
    return dict(error_cm=100*np.sqrt(np.mean([np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in a['contacts']]) for a in v])),force_rms=np.sqrt(np.mean([a['sum_force_norm']**2 for a in v])))
   before,after=tail(saved),tail(now);valid=not row['boundary'] and after['error_cm']<before['error_cm'] and after['force_rms']<before['force_rms']
   recover.append(dict(condition=name,takeover=take,prefix_max_abs=dif,continued=before,recovery=after,accepted=valid,samples=len(ss),scope='Only actual successful recovery branch labels; no labels from other r2 states'))
   if valid:visited+=ss
  assert len(recover)==3 and all(v['accepted'] for v in recover)
  gz(out/'visited-data.json.gz',visited);runner.write(out/'recovery.json',recover)
  # Identical common input must have identical target; check all cross-episode first rows.
  first=[next(r for r in data if r['episode']==e) for e in sorted(set(r['episode'] for r in data))]
  conflicts=[]
  for i,a in enumerate(first):
   for b in first[i+1:]:
    if np.array_equal(a['x'],b['x']):conflicts.append(dict(a=a['episode'],b=b['episode'],target_max=float(np.max(np.abs(np.array(a['z'])-b['z'])))))
  assert all(r['target_max']==0 for r in conflicts)
  runner.write(out/'dataset.json',dict(base_samples=len(data),base_episodes=10,visited_samples=len(visited),source=manifest,base_sha256=runner.sha(out/'base-data.json.gz'),visited_sha256=runner.sha(out/'visited-data.json.gz'),identical_input_pairs=conflicts))
 elif runner.ARGS.stage=='train':
  arm=os.environ['M38B_ARM'];ds=Path(runner.ARGS.resume);manifest=read(ds/'dataset.json');assert runner.sha(ds/'base-data.json.gz')==manifest['base_sha256'];assert runner.sha(ds/'visited-data.json.gz')==manifest['visited_sha256']
  base=read(ds/'base-data.json.gz');visited=read(ds/'visited-data.json.gz') if arm=='C2' else [];data=base+visited
  torch.manual_seed(cfg['seed']);net=load();net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=cfg['lr'])
  x=torch.tensor([r['x'] for r in data],device=device);target=torch.tensor([r['z'] for r in data],device=device);prev=torch.tensor([r['previous_z'] for r in data],device=device)
  obs={k:torch.tensor([r['obs'][k][0] for r in data],device=device) for k in env.keys};norm={k:v.detach() for k,v in model._normalize(obs).items()}
  seq={k:torch.tensor([[c['bfm_obs'][k][0] for c in r['sequence']] for r in data],device=device) for k in env.keys};sh={k:v.shape[2:] for k,v in seq.items()};seqnorm=model._normalize({k:v.reshape(-1,*sh[k]) for k,v in seq.items()});seqnorm={k:v.detach().reshape(len(data),5,*v.shape[1:]) for k,v in seqnorm.items()}
  mask=torch.arange(5,device=device)[None,:]<torch.tensor([r['valid_steps'] for r in data],device=device)[:,None]
  qtarget=torch.tensor([[c['q_PD'] for c in r['sequence']] for r in data],device=device)
  def endpoint(z,ix):return task.qof(model._actor({k:v[ix] for k,v in norm.items()},z,model.cfg.actor_std).mean.float())
  qt=torch.cat([endpoint(target[i:i+128],slice(i,i+128)) for i in range(0,len(data),128)])
  def zslerp(p,z):
   # Batch equivalent of native spherical(), each row independently normalized.
   a=torch.arange(1,6,device=device,dtype=z.dtype)[None,:,None]/5
   cos=((p*z).sum(-1)/256).clamp(-1+1e-7,1-1e-7);theta=torch.acos(cos)[:,None,None]
   zz=(torch.sin((1-a)*theta)*p[:,None]+torch.sin(a*theta)*z[:,None])/torch.sin(theta)
   zz=zz*(16/zz.norm(dim=-1,keepdim=True));return torch.cat([zz[:,:4],z[:,None]],1)
  def sequence(z,ix,p=None):
   zz=zslerp(prev[ix] if p is None else p,z);ob={k:v[ix].reshape(-1,*v.shape[2:]) for k,v in seqnorm.items()}
   q=task.qof(model._actor(ob,zz.reshape(-1,256),model.cfg.actor_std).mean.float());return q.reshape(-1,5,29)
  idx=torch.arange(min(16,len(data)),device=device);interpolation_error=max(float((zslerp(prev[idx],target[idx])[j,k]-spherical(prev[idx[j]][None],target[idx[j]][None],(k+1)/5)[0]).abs().max()) for j in range(len(idx)) for k in range(5))
  replay_error=float((sequence(target[idx],idx)-qtarget[idx])[mask[idx]].abs().max());assert interpolation_error<1e-4 and replay_error<1e-5,(interpolation_error,replay_error)
  groups={e:torch.tensor([i for i,r in enumerate(base) if r['episode']==e],device=device) for e in sorted(set(r['episode'] for r in base))};vg={e:torch.tensor([i+len(base) for i,r in enumerate(visited) if r['episode']==e],device=device) for e in sorted(set(r['episode'] for r in visited))}
  # Separate sampling RNG: identical base draws in C0/C1; C2 keeps first128/160
  # base samples and replaces32 with qualified visit data, fixed20% proportion.
  gen=torch.Generator(device=device).manual_seed(cfg['seed']);vgen=torch.Generator(device=device).manual_seed(cfg['seed']+1)
  curve=[]
  for step in range(3000):
   if time.monotonic()-runner.START>runner.ARGS.max_seconds-120:raise TimeoutError('reserved deadline')
   ix=torch.cat([g[torch.randint(len(g),(16,),device=device,generator=gen)] for g in groups.values()])
   if arm=='C2':
    # balanced retention of every original episode, rotating exact128 slots.
    keep=torch.arange(160,device=device);keep=keep[(keep+step)%5!=0]
    vi=torch.cat([g[torch.randint(len(g),(11,),device=device,generator=vgen)] for g in vg.values()]);ix=torch.cat([ix[keep],vi[torch.randperm(len(vi),device=device,generator=vgen)[:32]]])
   with torch.enable_grad():
    z=direct_command(net.distribution(x[ix]).mean,env.standing)
    lq=((endpoint(z,ix)-qt[ix])/np.deg2rad(1.)).square().mean() if arm=='C0' else ((sequence(z,ix)-qtarget[ix])[mask[ix]]/np.deg2rad(1.)).square().mean()
    lz=(z-target[ix]).square().mean();loss=lq+.1*lz;opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
   # Count every completed update, including partial failures, in supervisor ledger.
   runner.write(out/'progress.json',dict(updates=step+1,arm=arm)) if step%25==0 or step==2999 else None
   if step%100==0 or step==2999:
    curve.append(dict(step=step+1,qpd_loss=float(lq),z_loss=float(lz),gradient=gn));runner.write(out/'learning-curve.json',curve);print('M38B_FIT',arm,curve[-1],flush=True)
  net.eval().requires_grad_(False);path=out/(arm+'.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=1088,kind='direct',updates=3000,arm=arm,data=manifest,runtime=subprocess.check_output(['git','-C',str(runner.ROOT),'rev-parse','HEAD'],text=True).strip()),path)
  runner.write(out/'model.json',dict(path=str(path),sha256=runner.sha(path),arm=arm,updates=3000,interpolation_parity=interpolation_error,teacher_sequence_replay_max_rad=replay_error,data=manifest))
  audit=[]
  for e,g in groups.items():
   zz=direct_command(net.distribution(x[g]).mean,env.standing);err=torch.rad2deg(sequence(zz,g)-qtarget[g]);audit.append(dict(episode=e,sequence_qpd_rms_deg=float(err.square().mean().sqrt()),per_substep_rms_deg=err.square().mean((0,2)).sqrt().tolist(),z_angle_mean_deg=float(torch.rad2deg(torch.acos(((zz*target[g]).sum(-1)/256).clamp(-1,1))).mean())))
  runner.write(out/'alignment.json',audit)
  for name,spec in dev_specs(PLAN).items():rollout(arm+'_'+name,spec,net)
 elif runner.ARGS.stage=='final':
  selection=read(Path(runner.ARGS.resume));nets={k:load(v['path']) for k,v in selection['models'].items()}
  for key,spec in cfg['sealed'].items():
   for name,n in nets.items():rollout(name+'_'+key,spec,n)
   if spec.get('rotation_deg'):rollout('source_'+key,spec,source=True)
  # Same-state last-command intervention on real source trajectories; changes
  # only memory for counterqueries, not physical history or training data.
 elif runner.ARGS.stage=='probe':
  for n,spec in cfg['probes'].items():rollout('source_'+n,spec,source=True)
 else:raise ValueError(runner.ARGS.stage)
 result=dict(status='completed',expert_supervised_updates=3000 if runner.ARGS.stage=='train' else 0,student_optimizer_updates=0,decisions=0,episodes=len(summary),elapsed_s=time.monotonic()-runner.START)
 runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
