"""M0 single-skill LPP: physical expert query, Projector visits, one aggregation update.
The teacher has privileged anchors and bounded existing joint correction. Student does not.
"""
import copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from locomotion_rollout import UnsafePhysics,snapshot,spherical
from contact_projector import Projector,ContactGate,deploy_input,constrained,check_deadline
from ccl02a_teacher import arm_ik,ReleaseFade
from frozen_bfm_placement import tensor_digest
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUNS=P/'bridge-v2-owned/runs';RUN=RUNS/'m0-lpp-20261031A'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(out,name,x):(out/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def completed(stage):
 ledger=read(RUN/'compute-ledger.json');xs=[a for a in ledger['attempts'] if a['stage']==stage and a.get('result_status')=='completed'];assert len(xs)==1,(stage,xs)
 return Path(xs[0]['path'])/'outputs'
def load_net(path,dev):
 state=torch.load(path,map_location=dev,weights_only=True)['state_dict'];net=Projector(torch.zeros(2,len(state['center']),device=dev)).to(dev);net.load_state_dict(state);return net.eval().requires_grad_(False)

def query_expert(env,ep,contract):
 """Exact existing expert on recorded pre-states, including its 50Hz release fade.
 Separate MjData: no stepping, no privileged Student inputs, no newly invented controller.
 """
 m=env.sim.model;d=mujoco.MjData(m);arm=contract['arm_indices']
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids]
 scale=np.asarray(contract['q_PD_scale'])[0];offset=np.asarray(contract['q_PD_offset'])[0]
 fade=ReleaseFade();targets=[];qtargets=[];corrections=[];point_errors=[]
 for i,decision in enumerate(ep['decisions']):
  cs=ep['control'][i*5:i*5+5];assert len(cs)==5;first=cs[0]
  d.qpos[:]=first['pre_qpos'];d.qvel[:]=first['pre_qvel'];mujoco.mj_forward(m,d)
  point=d.xpos[env.tether.body]+d.xmat[env.tether.body].reshape(3,3)@np.asarray(env.cfg.local_point)
  jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,env.tether.body)
  if 'pre_anchor' in first:
   error=np.asarray(first['pre_anchor'])-point;point_errors.append(float(abs(point-first['pre_wrist']).max()))
  else:
   # Original actual expert records already contain the error measured at this decision.
   error=np.asarray(decision['error_xyz_m'])
  desired=np.zeros(7)
  if decision['active']:
   desired=arm_ik(jac[:,vadr[arm]],error)
   q0=np.asarray(first['nominal_q_PD']);desired=np.clip(q0[arm]+desired,limits[arm,0],limits[arm,1])-q0[arm]
  now=decision['tether_time'];corr=fade.update(now,decision['active'],desired);corrections.append(corr.tolist())
  for c in cs:
   value=fade.value(c['tether_time']) if fade.release_start is not None else corr
   assert max(abs(value))<=.040001 and np.linalg.norm(value)<=.080001
   da=np.zeros(29,dtype=np.float32);da[arm]=value/scale[arm]
   action=np.asarray(c['nominal'],dtype=np.float32)+da
   q=offset+np.clip(action*5,-contract['clip'],contract['clip'])*scale/5
   targets.append(action.tolist());qtargets.append(q.tolist())
 assert max(point_errors,default=0)<1e-10
 return dict(target_action=targets,target_q=qtargets,correction=corrections,point_error=max(point_errors,default=0),release_fade_included=True)

def dataset(env,model,contract,episodes,out):
 keys=env.keys;dev=env.standing.device;records=[];query_audits=[];manifests=[]
 for group,(path,expert_actual) in enumerate(episodes):
  ep=read(path);oracle=query_expert(env,ep,contract);gate=ContactGate()
  if expert_actual:
   err=float(abs(np.asarray(oracle['target_action'])-np.asarray([c['action'] for c in ep['control']])).max())
   qerr=float(abs(np.asarray(oracle['target_q'])-np.asarray([c['q_PD'] for c in ep['control']])).max())
   assert err<1e-6 and qerr<1e-6,(path,err,qerr)
   query_audits.append(dict(file=str(path),action_parity=err,q_PD_parity=qerr,release_included=True))
  manifests.append(dict(path=str(path),sha256=sha(path),group=group,actual_expert=expert_actual,source='oldexpert' if group==0 else 'newexpert' if group==1 else 'Projector visited'))
  for i,d in enumerate(ep['decisions']):
   cs=ep['control'][5*i:5*i+5];assert len(cs)==5
   force=d.get('force_xyz_n',d.get('force_xyz'));gate.update(d['tether_time'],force)
   obs={k:torch.tensor(cs[0]['bfm_obs'][k],device=dev,dtype=torch.float32) for k in keys}
   h=deploy_input(model,obs,d['force_history'],keys)[0].detach()
   gs=[gate.value(c['tether_time']) for c in cs]
   if not expert_actual:
    assert all(abs(c['gate']-g)<1e-8 for c,g in zip(cs,gs))
   records.append(dict(h=h,obs={k:[c['bfm_obs'][k][0] for c in cs] for k in keys},gate=gs,
     target_q=oracle['target_q'][5*i:5*i+5],target_action=oracle['target_action'][5*i:5*i+5],nominal_q=[c['nominal_q_PD'] for c in cs],
     weights=[5. if g==1 else 2. if g>0 else 1. for g in gs],group=group,time=d['tether_time']))
  save(out,'query-group-'+str(group)+'.json',oracle)
 data=dict(h=torch.stack([r['h'] for r in records]),obs={k:torch.tensor([r['obs'][k] for r in records],device=dev).flatten(0,1) for k in keys})
 for k in ['gate','target_q','target_action','nominal_q','weights']:data[k]=torch.tensor([r[k] for r in records],device=dev,dtype=torch.float32)
 data['group']=torch.tensor([r['group'] for r in records],device=dev);data['metadata']=[{'group':r['group'],'time':r['time']} for r in records]
 torch.save({k:({kk:vv.cpu() for kk,vv in v.items()} if k=='obs' else v.cpu() if torch.is_tensor(v) else v) for k,v in data.items()},out/'aggregation-dataset.pt')
 save(out,'dataset-manifest.json',dict(sources=manifests,query_audits=query_audits,decisions=len(records),control_samples=5*len(records),student_inputs='Frozen normalized465D proprio/action-history plus3 measured forces /20N; no time,anchor,direction,Jacobian,teacher state or latent label'))
 return data

class Alignment:
 def __init__(self,model,standing,data,contract):
  self.model=model;self.standing=standing;self.d=data;self.dev=standing.device
  self.norm={k:v.detach() for k,v in model._normalize(data['obs']).items()};self.scale=torch.tensor(contract['q_PD_scale'],device=self.dev);self.offset=torch.tensor(contract['q_PD_offset'],device=self.dev)
 def zs(self,raw,ids):return constrained(self.model,self.standing,raw.repeat_interleave(5,0),self.d['gate'][ids].flatten())[0]
 def evaluate(self,raw,ids):
  ix=(ids[:,None]*5+torch.arange(5,device=self.dev)).flatten();z=self.zs(raw,ids)
  act=self.model._actor({k:v[ix] for k,v in self.norm.items()},z,self.model.cfg.actor_std).mean.float();q=self.offset+act.clamp(-1,1)*self.scale
  e=q-self.d['target_q'][ids].flatten(0,1);w=self.d['weights'][ids].flatten()
  loss=((e[:,22:].square().mean(-1)+.1*e[:,:22].square().mean(-1))*w).sum()/w.sum()+1e-6*raw.tanh().square().mean()
  return loss,e,z,act

def fit(env,model,contract,out,protocol):
 dev=env.standing.device;source=completed('collect');pinned=read(source/'dataset-manifest.json')
 assert all(sha(m['path'])==m['sha256'] for m in pinned['sources'])
 d=torch.load(source/'aggregation-dataset.pt',map_location=dev,weights_only=True);f=Alignment(model,env.standing,d,contract);n=len(d['h'])
 init=RUNS/protocol['track_B']['initial_checkpoint'];assert sha(init)=='14b5a32798586d0810ff5e5461ce745890a90c717d1e6de2e8d6dc3f2a296fd8'
 nets={k:load_net(init,dev) for k in ['alignment','distill']};assert tensor_digest(nets['alignment'])==tensor_digest(nets['distill'])
 initial_digest=tensor_digest(nets['alignment']);save(out,'initial-identity.json',dict(sha256=sha(init),tensor_digest=initial_digest))
 # Exact fixed-observation gradient adapter and public inference equality.
 with torch.no_grad():
  public=model.act({k:v[:5] for k,v in d['obs'].items()},env.standing.expand(5,-1),mean=True)
  direct=model._actor({k:v[:5] for k,v in f.norm.items()},env.standing.expand(5,-1),model.cfg.actor_std).mean
  parity=float(abs(public-direct).max());assert parity<1e-6
 targets=[];errors=[];norms=[];grad=[]
 for start in range(0,n,16):
  check_deadline();ids=torch.arange(start,min(n,start+16),device=dev)
  with torch.enable_grad():
   raw=torch.zeros(len(ids),256,device=dev,requires_grad=True);opt=torch.optim.Adam([raw],lr=.05)
   before=float(f.evaluate(raw,ids)[0].detach())
   for step in range(160):
    check_deadline();opt.zero_grad();loss,e,z,act=f.evaluate(raw,ids);loss.backward();opt.step()
   loss,e,z,act=f.evaluate(raw,ids);gn=float(raw.grad.norm());grad.append(gn)
  targets.append(raw.detach());errors.append(e.detach().reshape(len(ids),5,29));norms.append(float(abs(z.norm(dim=-1)-16).max().detach()))
  print('REPRESENTABILITY',start,float(loss.detach()),flush=True)
  save(out,'projection-progress.json',dict(completed=min(n,start+16),total=n,last_before_loss=before,last_after_loss=float(loss.detach())))
 target_raw=torch.cat(targets);e=torch.cat(errors);active=d['gate']>0
 teacher_delta=d['target_q']-d['nominal_q'];ref=float(teacher_delta[active][:,22:].square().mean().sqrt());arm=float(e[active][:,22:].square().mean().sqrt());nonarm=float(e[active][:,:22].square().mean().sqrt())
 reach=dict(active_arm_rms_rad=arm,active_nonarm_rms_rad=nonarm,teacher_delta_arm_rms_rad=ref,ratio=arm/max(ref,1e-12),norm_error=max(norms),public_actor_parity=parity,temporary_steps_per_batch=160,batches=len(targets),grad_norm_range=[min(grad),max(grad)],passed=arm/max(ref,1e-12)<=.5 and nonarm<=.005)
 save(out,'representability.json',reach);torch.save(target_raw.cpu(),out/'temporary-target-raw.pt')
 if not reach['passed']:return dict(status='capability-stopped',representability=reach,student_optimizer_updates=0)
 # Simple rule = mean bounded residual from actual expert groups only, active decisions.
 selected=(d['group']<=1)&(d['gate'].max(-1).values>0)
 rule_r=target_raw[selected].tanh().mean(0).clamp(-.999,.999);rule_raw=torch.atanh(rule_r).reshape(1,256);torch.save(rule_raw.cpu(),out/'rule-raw.pt')
 def stats(net):
  es=[];zs=[]
  with torch.no_grad():
   for start in range(0,n,16):
    ids=torch.arange(start,min(n,start+16),device=dev);_,err,z,_=f.evaluate(net(d['h'][ids]),ids);es.append(err.reshape(len(ids),5,29));zs.append(z)
   err=torch.cat(es);z=torch.cat(zs);groups={}
   for g in range(4):
    mask=(d['group']==g)[:,None].expand(-1,5)&active
    if mask.any():groups[str(g)]={'arm_rms_rad':float(err[mask][:,22:].square().mean().sqrt()),'nonarm_rms_rad':float(err[mask][:,:22].square().mean().sqrt())}
   return dict(groups=groups,active_arm_rms_rad=float(err[active][:,22:].square().mean().sqrt()),active_nonarm_rms_rad=float(err[active][:,:22].square().mean().sqrt()),norm_error=float(abs(z.norm(dim=-1)-16).max()),angle_max_deg=float(torch.rad2deg(torch.acos(((z*env.standing).sum(-1)/256).clamp(-1,1))).max()))
 before={name:stats(net) for name,net in nets.items()};save(out,'fit-before.json',before)
 buffers={name:{k:v.clone() for k,v in net.named_buffers()} for name,net in nets.items()}
 opts={name:torch.optim.Adam(net.train().requires_grad_(True).parameters(),lr=.0003) for name,net in nets.items()}
 groups=[torch.nonzero(d['group'].cpu()==g,as_tuple=True)[0] for g in range(4)];gen=torch.Generator().manual_seed(0);schedule=[];trace=[]
 with torch.enable_grad():
  for step in range(1000):
   check_deadline();ids=torch.cat([g[torch.randperm(len(g),generator=gen)[:4]] for g in groups]).to(dev);schedule.append(ids.tolist());row={'step':step+1}
   for name,net in nets.items():
    opts[name].zero_grad();raw=net(d['h'][ids])
    if name=='alignment':loss,*_=f.evaluate(raw,ids)
    else:
     z=f.zs(raw,ids);target=f.zs(target_raw[ids],ids).detach();w=d['weights'][ids].flatten()
     loss=((z-target).square().mean(-1)*w).sum()/w.sum()+1e-6*raw.tanh().square().mean()
    loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(gn) and all(p.grad is None for p in model.parameters())
    opts[name].step();row[name]={'loss':float(loss.detach()),'grad_norm':float(gn)}
   if step%100==0 or step==999:
    trace.append(row);save(out,'fit-trace.json',trace);print('LPP_FIT',row,flush=True)
 after={name:stats(net) for name,net in nets.items()};save(out,'fit-after.json',after);checkpoint={}
 for name,net in nets.items():
  assert all(torch.equal(v,buffers[name][k]) for k,v in net.named_buffers());net.eval().requires_grad_(False)
  for p in net.parameters():p.grad=None
  path=out/(name+'-final.pt');torch.save(dict(state_dict=net.state_dict(),updates=1000,initial_sha256=sha(init)),path);checkpoint[name]=sha(path)
 save(out,'checkpoints-before-evaluation.json',dict(checkpoints=checkpoint,steps_each=1000,heldout_seen=False,initial_tensor_digest=initial_digest,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),same_batches=True,dataset_sha256=sha(source/'aggregation-dataset.pt')))
 return dict(status='completed',student_optimizer_updates=2000,optimizer_steps_each=1000,representability=reach,before=before,after=after,checkpoints=checkpoint)

def run_study(env,model,contract,out):
 protocol=read(Path(__file__).with_name('m0_protocol.json'));save(out,'protocol.json',protocol);stage=os.environ['M0_STAGE']
 if stage=='fit':return fit(env,model,contract,out,protocol)
 dev=env.standing.device;m=env.sim.model;original_act=model.act
 initpath=RUNS/protocol['track_B']['initial_checkpoint'];assert sha(initpath)=='14b5a32798586d0810ff5e5461ce745890a90c717d1e6de2e8d6dc3f2a296fd8'
 nets={'P0':load_net(initpath,dev)};rule=None
 if stage=='final':
  trained=completed('fit');pin=read(trained/'checkpoints-before-evaluation.json')
  for name,h in pin['checkpoints'].items():
   path=trained/(name+'-final.pt');assert sha(path)==h;nets[name]=load_net(path,dev)
  rule=torch.load(trained/'rule-raw.pt',map_location=dev,weights_only=True)
 digests={k:tensor_digest(v) for k,v in nets.items()}
 from alpha_student import load_student,student_input
 from alpha_memory import ForceMemory
 a2path=RUNS/'issue21-alpha-cross-20261031A/train-attempt-01/outputs/cross_A_final.pt'
 assert sha(a2path)=='3cc125a0504a360db2a9472ce115b3ab6fafcbb624f9460a07fd9991fe171163'
 a2,_=load_student(a2path,dev);goal=torch.tensor(read(RUNS/'issue21-goalcalibration-20261030A/attempt-01/outputs/same-state-forward.json')['states'][-1]['per_mode']['native']['z'],device=dev,dtype=torch.float32)
 zero=read(RUNS/'issue20-ccl01-20261028T082201Z/attempt-04/outputs/anchor-reference.json')['physics'];initial0=read(RUNS/'issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json');common=read(RUNS/'issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json')
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev);qtarget=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 arm=contract['arm_indices'];ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']];vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids]
 control=[];decisions=[];record=False;controller='standing';tick=0;raw=torch.zeros_like(env.standing);gate=ContactGate();fade=ReleaseFade();corr=np.zeros(7);last_q=None;memory=None
 def act(obs,z,mean=True):
  nonlocal tick,raw,corr,last_q
  now=float(env.tether.time);force=env.tether.force.copy();nominal=original_act(obs,env.standing,mean=True);q0=qtarget(nominal);command=env.standing;alpha=0.
  if record:
   if tick%5==0:
    active=bool(env.tether.active);gate.update(now,force);h=deploy_input(model,obs,env.history,env.keys)
    if controller in nets:raw=nets[controller](h).detach()
    elif controller=='rule':raw=rule
    desired=np.zeros(7)
    if controller=='expert' and active:
     point,_=env.tether.point_state();error=env.tether.anchor-point
     desired=arm_ik(env.tether.jac[:,vadr[arm]],error);nomq=q0[0].cpu().numpy();desired=np.clip(nomq[arm]+desired,limits[arm,0],limits[arm,1])-nomq[arm]
    corr=fade.update(now,active,desired) if controller=='expert' else np.zeros(7)
    decisions.append(dict(time=now,tether_time=now,active=active,force_xyz_n=force.tolist(),force_history=env.history.tolist(),bfm_obs={k:v.tolist() for k,v in obs.items()},correction_rad=corr.tolist(),raw=raw.tolist()))
   if controller in nets or controller=='rule':command=constrained(model,env.standing,raw,gate.value(now))[0]
   elif controller=='A2':
    hist=memory.append(force.tolist());alpha=float(a2(student_input(model,obs,[hist],env.keys)).item());command=spherical(env.standing,goal,alpha)
  action=original_act(obs,command,mean=True)
  if record and controller=='expert':
   cc=fade.value(now) if fade.release_start is not None else corr;assert max(abs(cc))<=.040001 and np.linalg.norm(cc)<=.080001
   da=torch.zeros_like(action);da[:,arm]=torch.as_tensor(cc,device=dev,dtype=action.dtype)/scale[:,arm];action=nominal+da
  if record:
   q=qtarget(action);point=env.last_point.copy();delta=q-q0
   assert torch.isfinite(action).all() and abs(float(command.norm())-16)<1e-4
   control.append(dict(time=now,tether_time=now,alpha=alpha,gate=gate.value(now),executed_z=command.tolist(),measured_force_xyz=force.tolist(),bfm_obs={k:v.tolist() for k,v in obs.items()},
    action=action[0].tolist(),nominal=nominal[0].tolist(),standing_action=nominal[0].tolist(),q_PD=q[0].tolist(),nominal_q_PD=q0[0].tolist(),standing_q_PD=q0[0].tolist(),delta_q_PD=delta[0].tolist(),
    pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),pre_wrist=point.tolist(),pre_anchor=env.tether.anchor.tolist(),pre_error=(env.tether.anchor-point).tolist(),
    action_saturation_fraction=float(((action*5).abs()>=contract['clip']).float().mean()),q_PD_step_max_rad=float(abs(q-last_q).max())))
   last_q=q.clone();tick+=1
  return action
 model.act=act;spec=protocol['track_B'];devcond=dict(spec['development']);devcond['enabled']=True
 if stage=='collect':
  neg=dict(devcond,name='dev_negative',enabled=False);jobs=[(devcond,k) for k in ['standing','expert','P0']]+[(neg,'P0')]
 else:
  held=dict(spec['heldout'],enabled=True);held['end']=8.96;neg=dict(spec['negative']);neg['end']=8.96
  jobs=[(c,ctl) for c in [devcond,held,neg] for ctl in spec['evaluation'][c['name']]]
 results=dict(status='running',episodes=[],student_optimizer_updates=0);baselines={}
 try:
  for cond,controller in jobs:
   check_deadline();label=cond['name']+'__'+controller;record=False;env.qualification_active=False;env.interruption_window=None
   env.cfg.enabled=False;env.cfg.activation_time=3.;env.cfg.ramp_duration=2.;env.cfg.release_time=6.6
   env.reset(seed=0,trajectory=(zero[0]['wrist'],[0,0,0]))
   for name in ['rew_buf','push_robot_counter','push_robot_plot_counter','command_counter','average_episode_length','torques','last_episode_length_buf']:
    v=getattr(env.base,name);v.copy_(torch.as_tensor(initial0['base_tensors'][name],device=v.device,dtype=v.dtype))
   env.base.common_step_counter=initial0['common_step_counter'];mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,env.sim.data);mujoco.mj_setState(m,env.sim.data,np.asarray(initial0['mj_integration']),mujoco.mjtState.mjSTATE_INTEGRATION)
   for _ in range(30):_,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True);assert not done
   parity={k:float(abs(np.asarray([r[k] for r in env.trace])-np.asarray([r[k] for r in zero[:600]])).max()) for k in ['qpos','qvel','wrist','root','force']};assert max(parity.values())==0;save(out,label+'-prefix.json',parity)
   env.base.rew_buf.copy_(torch.as_tensor(common['base_tensors']['rew_buf'],device=dev,dtype=env.base.rew_buf.dtype));state=snapshot(env);assert state==common
   initial=copy.deepcopy(env.trace[-1]);start=len(env.trace);control=[];decisions=[];tick=0;raw=torch.zeros_like(env.standing);gate=ContactGate();fade=ReleaseFade();corr=np.zeros(7);memory=ForceMemory(320);last_q=qtarget(env.previous_action).clone()
   row=zero[round(cond['onset']*200)-1];assert abs(row['time']-cond['onset'])<1e-8;anchor=np.asarray(row['wrist'])
   for cfg in (env.cfg,env.tether.cfg):cfg.activation_time=cond['onset'];cfg.ramp_duration=2.;cfg.release_time=cond['release'];cfg.enabled=cond['enabled']
   env.tether.origin=anchor.copy();env.tether.anchor=anchor.copy();env.tether.delta=np.asarray(cond['delta_xyz']);record=True;env.qualification_active=True;boundary=None
   steps=round((cond['end']-3)*50)
   for i in range(steps):
    check_deadline()
    try:
     action=model.act(env.obs,env.standing,mean=True);env.obs,_,terminated,truncated,_=env.wrapped.step(action,to_numpy=False);env.steps+=1;env.previous_action=action.clone();env.history=np.roll(env.history,-1,axis=0);env.history[-1]=env.tether.force
     if bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any()) or env.failed:boundary=dict(kind='native_failure',time=env.tether.time);break
    except UnsafePhysics as exc:boundary=dict(kind='unsafe_physics',physics=exc.row);break
   record=False;env.qualification_active=False
   ep=dict(label=label,controller=controller,condition=dict(cond,duration=cond['release']-cond['onset']),initial=initial,physics=copy.deepcopy(env.trace[start:]),control=control,decisions=decisions,boundary=boundary,common_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
   save(out,label+'.json',ep);entry=dict(label=label,seconds=len(ep['physics'])/200,complete=len(ep['physics'])==steps*4 and boundary is None,boundary=boundary);results['episodes'].append(entry);save(out,'physical-result.json',results);print('M0_PHYSICS',entry,flush=True)
   if not entry['complete']:results['status']='physical-stopped';return results
   key=cond['name']
   if key not in baselines:baselines[key]=ep
   else:assert [r['anchor'] for r in ep['physics']]==[r['anchor'] for r in baselines[key]['physics']]
  if stage=='collect':
   data=dataset(env,model,contract,[(RUNS/spec['old_expert_data'],True),(out/'dev__expert.json',True),(out/'dev__P0.json',False),(out/'dev_negative__P0.json',False)],out)
   results['dataset_decisions']=len(data['h'])
  assert all(tensor_digest(nets[k])==h for k,h in digests.items());results['status']='completed';save(out,'physical-result.json',results);return results
 finally:model.act=original_act;env.qualification_active=False
