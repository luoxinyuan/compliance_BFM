"""M38-C real-history causal and matched supervision study.
All learned deployments use the unchanged privileged Direct interface.
"""
from pathlib import Path
import json,time,gzip,os,subprocess,hashlib
import transfer_student_validated as runner
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
R=P/'bridge-v2-owned/runs/m38c-history-20261103'
OLD=P/'bridge-v2-owned/runs/m38-rotation-20261103'
SOURCE=P/'bridge-v2-owned/runs/contact-geometry-20261103/C-goal-10/job'
CONFIG=Path(__file__).with_name('m38c_protocol.json')
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
 cfg=read(CONFIG)
 # Optional preregistered partner-frame probe only; actual spring law unchanged.
 import geometry_contact
 native_frame=geometry_contact.frame
 def probe_frame(spec,t):
  p,v,a,w=native_frame(spec,t)
  if 'rotation_onset' in spec:
   u,du=geometry_contact.smooth(t,spec['rotation_onset'],spec['rotation_ramp']);r,dr=geometry_contact.smooth(t,spec['rotation_return'],spec['rotation_return_ramp']);angle=np.deg2rad(spec['rotation_deg']);a=angle*u*(1-r);w=angle*(du*(1-r)-u*dr)
  return p,v,a,w
 geometry_contact.frame=probe_frame
 runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 task=Task(env,model,contract,3801);task.reset(PLAN['single']);device=env.standing.device
 ident=PLAN['teachers']['initial'];assert runner.sha(ident['path'])==ident['sha256']
 ck=torch.load(ident['path'],map_location=device,weights_only=True)
 def load(path=None):
  n=Expert(1088,'direct').to(device);expand(n,ck if path is None else torch.load(path,map_location=device,weights_only=True));return n.eval().requires_grad_(False)
 old=load();summary=[]
 c2path=P/'bridge-v2-owned/runs/m38b-rotation-20261103/C-train-04/job/C2.pt'
 assert runner.sha(c2path)=='ff471b60c0bfc53c1d582a82fd3562416216c67c480e0e8039dee3d09277782c'
 c2=load(c2path)
 def source_setup(spec):
  if not spec.get('rotation_deg'):return None
  q=env.sim.data.qpos.copy();j=mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waist_yaw_joint');q[env.sim.model.jnt_qposadr[j]]+=np.deg2rad(spec['rotation_deg'])
  return model.goal_inference(goal_observation(env,q,np.zeros(env.sim.model.nv)))
 def rollout(name,spec,net=None,source=False,takeover=None,save_labels=False,replay=None):
  task.reset(spec,record=True,label=name);goal=source_setup(spec) if source else None;last_a=0.;rt=None;rv=0.;samples=[];decision_index=0
  while not task.done:
   t=float(env.tether.time);x=task.features();prev=task.last_command.clone();obs={k:v.cpu().tolist() for k,v in env.obs.items() if k in env.keys}
   if replay is not None:
    raw=torch.tensor([replay[decision_index]['raw']],device=device);z=direct_command(raw,env.standing)
   elif net is not None and (takeover is None or t<takeover-1e-8):raw=net.distribution(x).mean;z=direct_command(raw,env.standing)
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
   c=len(task.controls);task.step(raw);decision_index+=1
   if save_labels and (takeover is None or t>=takeover-1e-8):
    cc=task.controls[c:];count=len(cc);cc=cc+[cc[-1]]*(5-count);samples.append(dict(x=x[0].cpu().tolist(),z=z[0].cpu().tolist(),time=t,episode=name,obs=obs,previous_z=prev[0].cpu().tolist(),sequence=cc,valid_steps=count))
  row=task.save(out);summary.append(row);runner.write(out/'summaries.json',summary)
  if save_labels:gz(out/(name+'-labels.json.gz'),samples)
  print('M38C_PHYSICS',name,row['error_cm'],row['sum_force_norm_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
  return row,samples
 if runner.ARGS.stage=='prepare':
  # Match initial robot state and virtual partner frame. enabled=False affects
  # the physical spring, not a fake force observation ablation.
  cases={k:dict(PLAN['dev'][k]) for k in ['pair_left','pair_right']}
  cases.update(left_noforce=dict(cases['pair_left'],enabled=False),right_noforce=dict(cases['pair_right'],enabled=False),zero=dict(cases['pair_left'],rotation_deg=0.))
  for name,spec in cases.items():
   for label,net,src in [('C2',c2,False),('T30',old,False),('source',None,True)]:rollout(label+'_'+name,spec,net,source=src)
  # Saved full trajectories allow exact fork localization before recovery.
 elif runner.ARGS.stage=='probe':
  prepare=Path(runner.ARGS.resume);visited=[];recover=[]
  for side in ['left','right']:
   noforce=read(prepare/('C2_'+side+'_noforce.json.gz'))
   rollout('C2_noforce_command_replay_'+side,PLAN['dev']['pair_'+side],replay=noforce['decisions'])
  for name,take in cfg['recovery_points']:
   spec=PLAN['dev'][name];saved=read(prepare/('C2_'+name+'.json.gz')) if name!='straight' else read(P/'bridge-v2-owned/runs/m38b-rotation-20261103/C-train-04/job/C2_straight.json.gz')
   label='recovery_'+name+'_'+str(take).replace('.','p')
   row,ss=rollout(label,spec,c2,source=name!='straight',takeover=take,save_labels=True)
   pairs=[(a,b) for a,b in zip(saved['physics'],task.physics) if a['force_time']<take-1e-8]
   dif={k:max(float(np.max(np.abs(np.array(a[k])-b[k]))) for a,b in pairs) for k in ['qpos','qvel','force']};assert max(dif.values())==0,dif
   def tail(rr):
    v=[a for a in rr if a['active'] and a['force_time']>=take-1e-8]
    return dict(error_cm=float(100*np.sqrt(np.mean([np.mean([np.sum((np.array(c['anchor'])-c['point'])**2) for c in a['contacts']]) for a in v]))),force_rms=float(np.sqrt(np.mean([a['sum_force_norm']**2 for a in v]))))
   before,after=tail(saved['physics']),tail(task.physics)
   valid=bool(not row['boundary'] and after['error_cm']<before['error_cm'] and after['force_rms']<before['force_rms'])
   recover.append(dict(condition=name,takeover=take,prefix_max_abs=dif,continued=before,recovery=after,accepted=valid,samples=len(ss),trace=str(out/(label+'.json.gz')),scope='Only actually executed successful recovery successors, not all C2 states'))
   if valid:visited+=ss
   runner.write(out/'recovery.json',recover)
  gz(out/'visited-data.json.gz',visited)
  ds=P/'bridge-v2-owned/runs/m38b-rotation-20261103/validated-data'
  manifest=read(ds/'dataset.json')
  for f in ['base-data.json.gz','visited-data.json.gz']:assert runner.sha(ds/f)==manifest['base_sha256' if f.startswith('base') else 'visited_sha256']
  runner.write(out/'dataset.json',dict(base_path=str(ds/'base-data.json.gz'),old_visited_path=str(ds/'visited-data.json.gz'),base_sha256=runner.sha(ds/'base-data.json.gz'),old_visited_sha256=runner.sha(ds/'visited-data.json.gz'),visited_sha256=runner.sha(out/'visited-data.json.gz'),visited_samples=len(visited),recovery=recover))
 elif runner.ARGS.stage=='train':
  from m38c_train import train
  net=train(env,model,task,load,c2path,Path(runner.ARGS.resume),out,cfg)
  specs=dev_specs(PLAN);specs['zero']=dict(PLAN['dev']['pair_left'],rotation_deg=0.)
  specs['left_noforce']=dict(PLAN['dev']['pair_left'],enabled=False);specs['right_noforce']=dict(PLAN['dev']['pair_right'],enabled=False)
  for name,spec in specs.items():rollout(os.environ['M38C_ARM']+'_'+name,spec,net)
 elif runner.ARGS.stage=='final':
  selection=read(Path(runner.ARGS.resume));nets={k:load(v['path']) for k,v in selection['models'].items()}
  for v in selection['models'].values():assert runner.sha(v['path'])==v['sha256']
  for key,spec in cfg['sealed'].items():
   for name,n in nets.items():rollout(name+'_'+key,spec,n)
   rollout('source_'+key,spec,source=True)
 else:raise ValueError(runner.ARGS.stage)
 result=dict(status='completed',expert_supervised_updates=3000 if runner.ARGS.stage=='train' else 0,student_optimizer_updates=0,decisions=0,episodes=len(summary),elapsed_s=time.monotonic()-runner.START)
 runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
