"""Matched independent-frame vs own-command recurrence on real physical sequences.
D2 unrolls only policy command memory. BFM physical observations/action histories
remain measured recordings; this is explicitly NOT differentiable physics.
"""
import os,time,gzip,json,subprocess
import transfer_student_validated as runner

def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def train(env,model,task,load,c2path,ds,out,cfg):
 import torch,numpy as np
 from clean_policy import direct_command
 from locomotion_rollout import spherical
 arm=os.environ['M38C_ARM'];manifest=json.loads((ds/'dataset.json').read_text())
 from pathlib import Path
 paths=[Path(manifest['base_path']),Path(manifest['old_visited_path']),ds/'visited-data.json.gz']
 for p,key in zip(paths,['base_sha256','old_visited_sha256','visited_sha256']):assert runner.sha(p)==manifest[key]
 corpora=[read(p) for p in paths];assert corpora[2], 'No physically valid C2 recovery data; D1/D2 cannot be claimed'
 data=sum(corpora,[]);n=len(data);device=env.standing.device
 torch.manual_seed(cfg['seed']);net=load(c2path);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=cfg['lr'])
 x=torch.tensor([r['x'] for r in data],device=device);target=torch.tensor([r['z'] for r in data],device=device);prev=torch.tensor([r['previous_z'] for r in data],device=device)
 seq={k:torch.tensor([[c['bfm_obs'][k][0] for c in r['sequence']] for r in data],device=device) for k in env.keys};sh={k:v.shape[2:] for k,v in seq.items()}
 seqnorm=model._normalize({k:v.reshape(-1,*sh[k]) for k,v in seq.items()});seqnorm={k:v.detach().reshape(n,5,*v.shape[1:]) for k,v in seqnorm.items()}
 mask=torch.arange(5,device=device)[None,:]<torch.tensor([r['valid_steps'] for r in data],device=device)[:,None]
 qt=torch.tensor([[c['q_PD'] for c in r['sequence']] for r in data],device=device)
 def interp(p,z):
  a=torch.arange(1,6,device=device,dtype=z.dtype)[None,:,None]/5
  theta=torch.acos(((p*z).sum(-1)/256).clamp(-1+1e-7,1-1e-7))[:,None,None]
  zz=(torch.sin((1-a)*theta)*p[:,None]+torch.sin(a*theta)*z[:,None])/torch.sin(theta)
  zz=zz*(16/zz.norm(dim=-1,keepdim=True));return torch.cat([zz[:,:4],z[:,None]],1)
 def qpred(z,ix,p):
  zz=interp(p,z);ob={k:v[ix].reshape(-1,*v.shape[2:]) for k,v in seqnorm.items()}
  return task.qof(model._actor(ob,zz.reshape(-1,256),model.cfg.actor_std).mean.float()).reshape(-1,5,29)
 # Build contiguous five-high-level-decision blocks, never cross episodes.
 groups=[];offset=0
 for corpus in corpora:
  g={}
  for e in sorted(set(r['episode'] for r in corpus)):
   ids=[i+offset for i,r in enumerate(corpus) if r['episode']==e];valid=[]
   for j in range(len(ids)-4):
    ii=ids[j:j+5]
    if all(abs(data[b]['time']-data[a]['time']-.1)<1e-5 for a,b in zip(ii,ii[1:])):valid.append(ii)
   if valid:g[e]=torch.tensor(valid,device=device)
  groups.append(g);offset+=len(corpus)
 assert all(groups)
 # Verify actual recording continuity and replay; fail before optimizer updates.
 block=next(iter(groups[2].values()))[0];ix=block
 memory_error=float((prev[ix[1:]]-target[ix[:-1]]).abs().max());assert memory_error<1e-5,memory_error
 replay_error=float((qpred(target[ix],ix,prev[ix])-qt[ix])[mask[ix]].abs().max());assert replay_error<1e-5,replay_error
 intererr=max(float((interp(prev[ix],target[ix])[j,k]-spherical(prev[ix[j]][None],target[ix[j]][None],(k+1)/5)[0]).abs().max()) for j in range(5) for k in range(5));assert intererr<1e-4
 # Teacher-forced sequence vs independent-frame numerical parity.
 forced=torch.cat([direct_command(net.distribution(x[ix[k:k+1]]).mean,env.standing) for k in range(5)])
 batch=direct_command(net.distribution(x[ix]).mean,env.standing);parity=float((forced-batch).abs().max());assert parity<1e-4
 runner.write(out/'sequence-validation.json',dict(actual_previous_z_max_error=memory_error,teacher_qpd_replay_max_rad=replay_error,slerp_max=intererr,teacher_forced_batch_parity=parity,length=5,physical_observations='fixed real recorded observations, not differentiable rollout',sampler='D0 26base+6oldvisit blocks; D1/D2 20base+6oldvisit+6newvisit; 32x5=160 frames; identical D1/D2 blocks'))
 gen=torch.Generator(device=device).manual_seed(cfg['seed']);keys=[sorted(g) for g in groups]
 def sample(group,count,step):
  rows=[]
  for j in range(count):
   g=groups[group][keys[group][(j+step)%len(keys[group])]];rows.append(g[torch.randint(len(g),(1,),device=device,generator=gen)])
  return torch.cat(rows)
 curve=[]
 for step in range(3000):
  if time.monotonic()-runner.START>runner.ARGS.max_seconds-180:raise TimeoutError('reserve evaluation time')
  blocks=torch.cat([sample(0,26 if arm=='D0' else 20,step),sample(1,6,step)]+([] if arm=='D0' else [sample(2,6,step)]))
  with torch.enable_grad():
   if arm=='D2':
    pp=prev[blocks[:,0]];zs=[];qs=[]
    for k in range(5):
     ii=blocks[:,k];xx=x[ii].clone();xx[:,565:821]=pp/16
     zz=direct_command(net.distribution(xx).mean,env.standing);qs.append(qpred(zz,ii,pp));zs.append(zz);pp=zz
    zz=torch.stack(zs,1).reshape(-1,256);qq=torch.stack(qs,1).reshape(-1,5,29)
    if step==0:
     future_grad=torch.autograd.grad(qs[-1].square().mean(),zs[0],retain_graph=True)[0];assert torch.isfinite(future_grad).all() and future_grad.norm()>0
     runner.write(out/'own-history-gradient.json',dict(last_qpd_loss_gradient_to_first_command=float(future_grad.norm()),physical_graph=False))
   else:
    ii=blocks.reshape(-1);zz=direct_command(net.distribution(x[ii]).mean,env.standing);qq=qpred(zz,ii,prev[ii])
   ii=blocks.reshape(-1);lq=((qq-qt[ii])[mask[ii]]/np.deg2rad(1)).square().mean();lz=(zz-target[ii]).square().mean();loss=lq+.1*lz
   opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));assert np.isfinite(gn);opt.step()
  runner.write(out/'progress.json',dict(updates=step+1,arm=arm))
  if step%100==0 or step==2999:
   curve.append(dict(step=step+1,qpd_loss=float(lq),z_loss=float(lz),gradient=gn,frames=160));runner.write(out/'learning-curve.json',curve);print('M38C_FIT',arm,curve[-1],flush=True)
 net.eval().requires_grad_(False);path=out/(arm+'.pt');runtime=subprocess.check_output(['git','-C',str(runner.ROOT),'rev-parse','HEAD'],text=True).strip()
 torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=1088,kind='direct',updates=3000,arm=arm,data=manifest,runtime=runtime,initial_sha=runner.sha(c2path)),path)
 runner.write(out/'model.json',dict(path=str(path),sha256=runner.sha(path),arm=arm,updates=3000,runtime=runtime,initial_sha=runner.sha(c2path),data=manifest))
 return net
