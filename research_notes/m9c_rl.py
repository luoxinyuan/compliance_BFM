"""Privileged high-level residual expert acquisition on frozen BFM, original PD.
PPO is an acquisition optimizer, not a direct-PPO versus LPP study.
Raw Gaussian likelihoods precede bounded legal-z transformation.
"""
import os,time,json,copy,gzip,math
from pathlib import Path
import numpy as np,torch
from m9c_stage import P,RUN,completed
from m9b_stage import RUN as PREVIOUS
from m8_stage import reset_common,advance
from m9_cyclic import CyclicFeedback,sample
from m9c_contact import install_partner
from locomotion_rollout import spherical,UnsafePhysics
from contact_projector import check_deadline
from m1_stage import save
from m9_learning import read,sha

class Adapter(torch.nn.Module):
 def __init__(self,n,dim=19):
  super().__init__();self.actor=torch.nn.Sequential(torch.nn.Linear(n,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,dim));self.critic=torch.nn.Sequential(torch.nn.Linear(n,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,1));torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias);self.register_buffer('std',torch.full((dim,),.25))
 def distribution(self,x):return torch.distributions.Normal(self.actor(x),self.std)

def basis_for(env,model,refs):
 # Existing motion variation plus action-sensitive upper-body directions. No random-z search.
 z=env.standing.detach().clone().requires_grad_(True)
 with torch.enable_grad():
  a=model.act(env.obs,z,mean=True);grad=[torch.autograd.grad(a[0,j],z,retain_graph=True)[0][0] for j in range(15,29)]
 J=torch.stack(grad);J=J/J.norm(dim=1,keepdim=True).clamp_min(1e-6)
 motion=torch.cat([r['tensor'][::4] for r in refs.values()])-env.standing;motion=motion/motion.norm(dim=1,keepdim=True).clamp_min(1e-6)
 _,_,V=torch.linalg.svd(torch.cat([motion,J.repeat(4,1)]),full_matrices=False);return V[:16].detach()

def features(env,model,state,prev):
 norm=model._normalize(env.obs);pro=torch.cat([norm[k].reshape(1,-1) for k in env.keys],-1).clamp(-10,10)
 R=env.sim.data.xmat[1].reshape(3,3);err=R.T@(env.tether.anchor-env.last_point);vel=R.T@getattr(env.tether,'partner_velocity',np.zeros(3));force=R.T@env.tether.force;rv=R.T@env.sim.data.qvel[:3]
 x=np.r_[np.clip(err/.3,-4,4),vel/.3,force/5,rv/.5,env.sim.data.qpos[2],R[2,:2],np.sin(2*np.pi*state['phase']),np.cos(2*np.pi*state['phase']),state['alpha'],state['turn'],prev.cpu().numpy()[0]]
 return torch.cat([pro,torch.as_tensor(x,device=pro.device,dtype=pro.dtype)[None]],-1)

def command(model,standing,refs,state,raw,basis):
 b=torch.tanh(raw);turn=float(np.clip(state['turn']+.25*float(b[0,17]),-1,1));alpha=float(np.clip(state['alpha']+.15*float(b[0,18])*state['alpha'],0,1));native=spherical(sample(refs['straight'],state['phase']),sample(refs['left' if turn>=0 else 'right'],state['phase']),abs(turn));base=spherical(standing,native,alpha)
 d=b[:,:16]@basis;d=d-(d*base).sum(-1,keepdim=True)/256*base;d=d*torch.clamp(4/d.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1);z=model.project_z(base+alpha*d)
 return z,float(np.exp(.25*float(b[0,16])))

def reward(rows,oldq,newq):
 r=rows[-1];F=np.asarray([x['force'] for x in rows]);E=np.asarray([np.asarray(x['anchor'])-x['wrist'] for x in rows]);v=np.asarray([x['qvel'][:3] for x in rows]);vp=np.asarray([x['anchor_velocity'] for x in rows]);R=np.array(r['pelvis_matrix']);guide=vp[-1]+.5*E[-1];heading=np.arctan2(R[1,0],R[0,0]);desired=np.arctan2(guide[1],max(guide[0],.02));active=any(x['active'] for x in rows)
 wrist=np.array([x['wrist'] for x in rows]);root=np.array([x['root'] for x in rows]);relvel=np.diff(wrist-root,axis=0)/.005
 terms=dict(force=1.5*float((F**2).sum(1).mean()/25),yz=.5*float((F[:,1:]**2).sum(1).mean()/25),tracking=.5*float(np.minimum((E**2).sum(1)/.04,16).mean()) if active else 0.,velocity=.3*float(np.minimum(((v[:,:2]-vp[:,:2])**2).sum(1)/.04,16).mean()),heading=.5*(1-math.cos(heading-desired)) if active else 0.,wrist=.1*float(np.minimum((relvel**2).sum(1)/.25,16).mean()),qpd=.05*float(torch.clamp((newq-oldq).square().mean()/.09,max=16)),slip=.1*min((max(x['contact_tangent_speed_max_mps'] for x in rows)/.1)**2,16),tilt=.1*min((r['tilt_deg']/15)**2,16))
 return -sum(terms.values())/10,terms

def rollout(env,model,contract,refs,net,basis,spec,label,out,stochastic=False,save_full=True):
 orig=model.act;reset_common(env,model,orig);rotation=None
 if spec.get('heading_deg'):
  from m9_heading import rotate_ready
  rotation,pivot,_=rotate_ready(env,model,spec['heading_deg'])
 free=read(P/'bridge-v2-owned/runs/m8-contact-body-20261101A/qualification-attempt-01/outputs/standing.json.gz')['physics'];rr=min(free,key=lambda r:abs(r['time']-spec['onset']));anchor=np.array(rr['wrist']);anchor=anchor if rotation is None else pivot+rotation@(anchor-pivot)
 for c in [env.cfg,env.tether.cfg]:c.enabled=spec['enabled'];c.activation_time=spec['onset'];c.release_time=spec['release'];c.ramp_duration=spec['ramp']
 oldmeasure=env.tether.measure;install_partner(env.tether,spec,anchor);initial=copy.deepcopy(env.trace[-1]);initial['wrist']=env.last_point.tolist();start=len(env.trace);env.qualification_active=True;fb=CyclicFeedback({k:r['period'] for k,r in refs.items()},True);prev=torch.zeros((1,19),device=env.standing.device);raw=prev.clone();records=[];cs=[];costs=[];boundary=None;block=None;rate=1.;oldq=None;scale=torch.tensor(contract['q_PD_scale'],device=prev.device);off=torch.tensor(contract['q_PD_offset'],device=prev.device);qof=lambda a:off+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5;oldq=qof(env.previous_action);blockq=oldq.clone();blockstart=len(env.trace)
 try:
  for tick in range(round((spec['end']-float(env.tether.time))*50)):
   check_deadline();t=float(env.tether.time);R=env.sim.data.xmat[1].reshape(3,3).copy();guide=getattr(env.tether,'partner_velocity',np.zeros(3))+.5*(env.tether.anchor-env.last_point);before=fb.phase;state=fb.step(env.tether.force,R,guide=guide);fb.phase=before+(fb.phase-before)*rate;state['phase']=fb.phase
   if tick%5==0:
    x=features(env,model,state,prev);dist=net.distribution(x);raw=dist.sample() if stochastic else dist.mean;lp=dist.log_prob(raw).sum(-1);val=net.critic(x).flatten();block=dict(x=x.detach(),raw=raw.detach(),logprob=lp.detach(),value=val.detach(),mean=dist.mean.detach());blockstart=len(env.trace);blockq=oldq.clone();prev=raw.detach()
   z,rate=command(model,env.standing,refs,state,raw,basis);a=orig(env.obs,z,mean=True);q=qof(a)
   if save_full:cs.append(dict(time=t,controller=label,expert_state=state,expert_raw=raw.tolist(),expert_input=block['x'].tolist(),executed_z=z.tolist(),oracle_z=z.tolist(),oracle_q_PD=q[0].tolist(),action=a[0].tolist(),q_PD=q[0].tolist(),q_PD_step_max_rad=float((q-oldq).abs().max()),pre_qpos=env.sim.data.qpos.tolist(),pre_qvel=env.sim.data.qvel.tolist(),measured_force_xyz=env.tether.force.tolist(),bfm_obs={k:v.tolist() for k,v in env.obs.items()},action_saturation_fraction=float(((a*5).abs()>=contract['clip']).float().mean())))
   oldq=q.clone()
   try:
    if advance(env,a):boundary=dict(kind='native_termination',time=float(env.tether.time))
   except UnsafePhysics as exc:boundary=dict(kind='physics_review',physics=exc.row)
   if tick%5==4 or boundary:
    re,terms=reward(env.trace[blockstart:],blockq,q);re-=5 if boundary else 0;block.update(reward=re,done=boundary is not None);records.append(block);costs.append(terms)
   if boundary:break
 finally:env.qualification_active=False;env.tether.measure=oldmeasure
 if save_full:
  ep=dict(label=label,initial=initial,physics=copy.deepcopy(env.trace[start:]),control=cs,boundary=boundary,condition=spec,anchor=anchor.tolist(),phases={'pre':[3,spec['onset']],'move':[spec['onset'],spec['onset']+spec['ramp']],'hold':[spec['onset']+spec['ramp'],spec['release']],'recover':[spec['release'],spec['end']]});save(out,label+'.json',ep)
 stats=dict(label=label,boundary=boundary,control_samples=len(env.trace[start:])//4,physical_steps=len(env.trace[start:]),rl_samples=len(records),reward_mean=float(np.mean([r['reward'] for r in records])),cost_terms={k:float(np.mean([r[k] for r in costs])) for k in costs[0]})
 # Finite episodic horizon is terminal, not a continuing-task bootstrap.
 if records:records[-1]['done']=True
 return records,stats

def update(net,opt,episodes):
 xs=[];aa=[];ll=[];vv=[];mm=[];ad=[];ret=[]
 for rows in episodes:
  gae=0.;nxt=0.;adv=[]
  for r in rows[::-1]:
   v=float(r['value']);d=r['reward']+.99*nxt*(not r['done'])-v;gae=d+.99*.95*gae*(not r['done']);adv.append(gae);nxt=v
  adv=adv[::-1]
  for r,g in zip(rows,adv):xs.append(r['x']);aa.append(r['raw']);ll.append(r['logprob']);vv.append(r['value']);mm.append(r['mean']);ad.append(g);ret.append(g+float(r['value']))
 x=torch.cat(xs);a=torch.cat(aa);old=torch.cat(ll);ov=torch.cat(vv);om=torch.cat(mm);adv=torch.tensor(ad,device=x.device);returns=torch.tensor(ret,device=x.device);adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
 d=net.distribution(x);pre=float((d.log_prob(a).sum(-1)-old).abs().max());assert pre<2e-5
 logs=[];updates=0
 with torch.enable_grad():
  for epoch in range(4):
   order=torch.randperm(len(x),device=x.device)
   for ix in order.split(128):
    dist=net.distribution(x[ix]);lp=dist.log_prob(a[ix]).sum(-1);ratio=(lp-old[ix]).exp();loss=-torch.minimum(ratio*adv[ix],ratio.clamp(.9,1.1)*adv[ix]).mean()+.5*(net.critic(x[ix]).flatten()-returns[ix]).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step();updates+=1
   with torch.no_grad():
    mu=net.actor(x);kl=float(((mu-om).square()/(2*net.std.square())).sum(-1).mean());ratio=(net.distribution(x).log_prob(a).sum(-1)-old).exp();clip=float((abs(ratio-1)>.1).float().mean());logs.append(dict(epoch=epoch,analytic_joint_kl=kl,clip_fraction=clip))
   if kl>.015:break
 newv=net.critic(x).flatten();ev=lambda p:float(1-(returns-p).var()/returns.var().clamp_min(1e-8))
 return dict(updates=updates,preupdate_logprob_max_error=pre,preupdate_ratio_max_error=float((d.log_prob(a).sum(-1)-old).exp().sub(1).abs().max()),old_value_EV=ev(ov),post_value_EV=ev(newv),logs=logs)

def run(env,model,contract,out,refs,stage):
 plan=read(Path(__file__).with_name('m9c_protocol.json'));torch.manual_seed(plan['seed']);reset_common(env,model,model.act);basis=basis_for(env,model,refs);dummy=CyclicFeedback({k:r['period'] for k,r in refs.items()},True).step(np.zeros(3),np.eye(3),guide=np.zeros(3));x=features(env,model,dummy,torch.zeros(1,19,device=env.standing.device));net=Adapter(x.shape[1]).to(x.device);logs=[];samples=0;steps=0;updates=0;started=time.monotonic()
 if stage=='rl':
  opt=torch.optim.Adam(net.parameters(),lr=3e-5);settings=plan['rl'];torch.save(dict(state_dict=net.state_dict(),basis=basis,input_dim=x.shape[1]),out/'initial.pt')
  for iteration in range(settings['iterations']):
   episodes=[]
   for j,spec in enumerate(settings['development']):
    rows,stats=rollout(env,model,contract,refs,net,basis,spec,f'train-{iteration:02d}-{j}',out,True,True);episodes.append(rows);samples+=stats['rl_samples'];steps+=stats['physical_steps'];logs.append(stats);print('RL_ROLLOUT',iteration,j,stats['reward_mean'],stats['boundary'] and stats['boundary']['kind'],flush=True)
   u=update(net,opt,episodes);updates+=u['updates'];logs.append(dict(iteration=iteration,optimization=u));torch.save(dict(state_dict=net.state_dict(),basis=basis,input_dim=x.shape[1],iteration=iteration,rl_samples=samples,physical_steps=steps),out/f'iteration-{iteration:02d}.pt');(out/'progress.json').write_text(json.dumps(dict(logs=logs,rl_samples=samples,physical_steps=steps,updates=updates,wall_s=time.monotonic()-started),indent=2));print('RL_UPDATE',iteration,u,flush=True)
  ck=out/'expert-final.pt';torch.save(dict(state_dict=net.state_dict(),basis=basis,input_dim=x.shape[1],rl_samples=samples,physical_steps=steps,fixed_final=True),ck);result=dict(status='completed',rl_samples=samples,physical_steps=steps,optimizer_updates=updates,checkpoint=str(ck),sha256=sha(ck),logs=logs,student_optimizer_updates=0)
 else:
  ck=completed('rl')/'expert-final.pt';loaded=torch.load(ck,map_location=x.device,weights_only=True);net.load_state_dict(loaded['state_dict']);basis=loaded['basis'];assert sha(ck)==plan['expert_final_sha256']
  for name,spec in plan['rl_evaluation'].items():
   _,st=rollout(env,model,contract,refs,net,basis,spec,name+'__rl',out,False,True);logs.append(st);print('RL_EVAL',name,st,flush=True)
  result=dict(status='completed',episodes=logs,student_optimizer_updates=0)
 (out/'rl-results.json').write_text(json.dumps(result,indent=2));return result
