"""v2 on-policy joint conditional PPO; no action Q or expert replay in ratios."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from .ei_v2_agent import SIGMA

def device_obs(o,device):return {k:v.to(device) for k,v in o.items()}
def stack_obs(rows,key='observation'):return {k:torch.cat([r[key][k] for r in rows],0) for k in rows[0][key]}

def segments(rows):
    out=[]
    for i,r in enumerate(rows):
        if not out or r['macro_slot']==0 or r['episode']!=rows[out[-1][-1]]['episode']:out.append([])
        out[-1].append(i)
    assert all(1<=len(g)<=5 for g in out)
    return out

@torch.no_grad()
def advantages(agent,rows,groups):
    device=agent.device;rewards=[];vs=[];ns=[];discount=[];trace=[]
    gamma=.99**(1/5);lam=.95**(1/5)
    for g in groups:
        rr=0.;d=1.;duration=0.
        for idx in g:
            row=rows[idx];dt=float(row['duration']);rr+=d*float(row['aux_rewards']['contact_physical']);d*=gamma**(dt/.02);duration+=dt
        first,last=rows[g[0]],rows[g[-1]]
        term=bool(last['next']['terminated']);trunc=bool(last['next']['truncated'])
        v=float(agent.v(device_obs(first['observation'],device)))
        nv=0. if term else float(agent.v(device_obs(last['next']['observation'],device)))
        rewards.append(rr);vs.append(v);ns.append(nv);discount.append(d);trace.append(0. if term or trunc else d*lam**(duration/.02))
    for i in range(len(groups)-1):
        last=rows[groups[i][-1]];nxt=rows[groups[i+1][0]]
        if last['episode']==nxt['episode']:
            assert torch.equal(agent.value_input(device_obs(last['next']['observation'],device)),agent.value_input(device_obs(nxt['observation'],device))), 'state-value input changed just by selecting the next action'
    adv=np.zeros(len(groups),dtype=np.float64);carry=0.
    for i in reversed(range(len(groups))):
        if i==len(groups)-1 or rows[groups[i][-1]]['episode']!=rows[groups[i+1][0]]['episode']:carry=0.
        carry=rewards[i]+discount[i]*ns[i]-vs[i]+trace[i]*carry;adv[i]=carry
    ret=adv+np.asarray(vs);normal=(adv-adv.mean())/(adv.std()+1e-8)
    return torch.tensor(normal,device=device,dtype=torch.float32),torch.tensor(ret,device=device,dtype=torch.float32),dict(raw_adv=adv.tolist(),values=vs,next_values=ns,macro_rewards=rewards,discounts=discount,trace_discounts=trace)

def current(agent,rows,groups,indices,grad=True):
    # Batch-one means and actual recorded z/history, never regenerate counterfactual z.
    lp=[];old=[]
    for gi in indices:
        g=groups[gi];new_total=torch.zeros((),device=agent.device);old_total=torch.zeros_like(new_total)
        for idx in g:
            row=rows[idx];o=device_obs(row['observation'],agent.device);z=row['z'].to(agent.device);op=torch.tensor([row['option']],device=agent.device)
            mu=agent.mean(o,z,op);new_total=new_total+agent.logp(row['raw_u'].to(agent.device),mu).sum();old_total=old_total+row['old_logp'].to(agent.device).sum()
            if row['cat_boundary']:
                co=device_obs(row['selector_obs'],agent.device);zs=row['candidate_first_z'].to(agent.device)
                logits=agent.logits(co,zs);new_total=new_total+F.log_softmax(logits,-1)[0,row['option']];old_total=old_total+F.log_softmax(row['old_logits'].to(agent.device),-1)[0,row['option']]
        lp.append(new_total);old.append(old_total)
    return torch.stack(lp),torch.stack(old)

@torch.no_grad()
def kl(agent,rows,groups):
    gaussian=0.;categorical=0.;sample=0.
    for g in groups:
        ng=0.;nc=0.;oldlp=0.;newlp=0.
        for idx in g:
            r=rows[idx];o=device_obs(r['observation'],agent.device);z=r['z'].to(agent.device);op=torch.tensor([r['option']],device=agent.device)
            mu=agent.mean(o,z,op);oldmu=r['old_mu'].to(agent.device)
            ng+=float(((oldmu-mu).square()/(2*SIGMA**2)).sum());oldlp+=float(r['old_logp'].sum());newlp+=float(agent.logp(r['raw_u'].to(agent.device),mu).sum())
            if r['cat_boundary']:
                a=F.log_softmax(r['old_logits'].to(agent.device),-1);b=F.log_softmax(agent.logits(device_obs(r['selector_obs'],agent.device),r['candidate_first_z'].to(agent.device)),-1)
                nc+=float((a.exp()*(a-b)).sum());oldlp+=float(a[0,r['option']]);newlp+=float(b[0,r['option']])
        gaussian+=ng;categorical+=nc;sample+=oldlp-newlp
    n=max(len(groups),1)
    return dict(gaussian=gaussian/n,categorical=categorical/n,joint=(gaussian+categorical)/n,sample_joint_old_minus_new=sample/n)

def retention(agent,by_parent,eps):
    vals=[];parent_v=[]
    for parent,rows in by_parent.items():
        if not rows:continue
        v=[]
        for r in rows:
            o=device_obs(r['observation'],agent.device);z=r['z'].to(agent.device);op=torch.tensor([r['option']],device=agent.device)
            from .schema import robot
            with torch.no_grad():old=agent.base._actor(agent.base._normalize(robot(o)),z,agent.base.cfg.actor_std).mean
            new=agent.mean(o,z,op)
            scale=torch.as_tensor(eps['qpd_scale'],device=agent.device);clip=eps['original_action_clip']
            delta=((new*5).clamp(-clip,clip)-(old*5).clamp(-clip,clip))*scale/5
            norm=torch.maximum(delta.square().mean(-1).sqrt()/eps['eps_rms'],delta.abs().max(-1).values/eps['eps_p95'])
            v.append(norm[0])
        vv=torch.stack(v);vals.append(vv.square().mean());parent_v.append(torch.quantile(vv,.9))
    if not vals:raise RuntimeError('No qualified parent-balanced retention data')
    return torch.stack(vals).mean(),torch.stack(parent_v).mean()

def update(agent,rows,keep,eps,aux_batch,phase):
    assert rows and len(rows)<=1024
    assert all(r.get('eligible_for_joint_PPO',False) and r['collection_role'] in ('formal_on_policy','engineering_on_policy') for r in rows), 'non-Gaussian/reference/qualification data cannot enter PPO'
    groups=segments(rows);indices=list(range(len(groups)));device=agent.device
    for p in agent.selector.parameters():p.requires_grad_(phase=='P3')
    with torch.no_grad():
        new,old=current(agent,rows,groups,indices);err=float((new-old).abs().max());assert err<=3e-4,('original joint logprob assertion',err)
    adv,ret,gae=advantages(agent,rows,groups);metrics={'old_logprob_max_error':err,'groups':len(groups),'rows':len(rows),'GAE':gae,'steps':[]}
    lam=float(agent.retention_lambda)
    with torch.enable_grad():
        for epoch in range(2):
            order=torch.randperm(len(groups)).tolist()
            for begin in range(0,len(order),64):
                ix=order[begin:begin+64];saved=copy.deepcopy(agent.modules.state_dict());savedopt=copy.deepcopy(agent.opt.state_dict())
                new,old=current(agent,rows,groups,ix);ratio=(new-old).exp();aa=adv[ix]
                task=-torch.minimum(ratio*aa,ratio.clamp(.8,1.2)*aa).mean()
                keep_loss,violation=retention(agent,keep,eps);loss=task+lam*keep_loss
                agent.opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(agent.modules.parameters(),1.));assert math.isfinite(grad)
                agent.opt.step();k=kl(agent,rows,groups)
                accept=math.isfinite(k['joint']) and k['joint']<=.02
                if accept:agent.accepted+=1
                else:agent.modules.load_state_dict(saved);agent.opt.load_state_dict(savedopt);agent.rejected+=1
                # Value optimizer is separate and never affects Actor probabilities.
                ob=stack_obs([rows[groups[i][0]] for i in ix]);v=agent.v(device_obs(ob,device));vl=F.mse_loss(v,ret[ix]);agent.vopt.zero_grad(set_to_none=True);vl.backward();vg=float(torch.nn.utils.clip_grad_norm_(agent.value.parameters(),1.));assert math.isfinite(vg);agent.vopt.step()
                metrics['steps'].append(dict(epoch=epoch,accepted=accept,KL=k,policy_loss=float(task.detach()),keep_loss=float(keep_loss.detach()),keep_violation=float(violation.detach()),lambda_keep=lam,actor_grad=grad,value_loss=float(vl.detach()),clip_fraction=float(((ratio-1).abs()>.2).float().mean())))
        for _ in range(4):
            batch=aux_batch()
            if batch is None:raise RuntimeError('No valid current-policy auxiliary horizons; cannot silently omit prescribed updates')
            o,z,op,y,mask=batch;saved=copy.deepcopy(agent.modules.state_dict());savedopt=copy.deepcopy(agent.opt.state_dict())
            al,terms=agent.auxiliary(o,z,op,y,mask);agent.opt.zero_grad(set_to_none=True);(.1*al).backward()
            assert all(p.grad is None for p in list(agent.tail.parameters())+list(agent.film.parameters())+list(agent.selector.parameters())), 'future-supervision gradient entered execution blocks'
            ag=float(torch.nn.utils.clip_grad_norm_(agent.modules.parameters(),1.));assert math.isfinite(ag);agent.opt.step();k=kl(agent,rows,groups);accept=math.isfinite(k['joint']) and k['joint']<=.02
            if not accept:agent.modules.load_state_dict(saved);agent.opt.load_state_dict(savedopt);agent.rejected+=1
            else:agent.accepted+=1
            metrics['steps'].append(dict(auxiliary=True,accepted=accept,KL=k,losses=terms,grad=ag))
    with torch.no_grad():
        _,violation=retention(agent,keep,eps);agent.retention_lambda=float(np.clip(lam+.05*(float(violation)-1),0,3))
    metrics['next_lambda']=agent.retention_lambda
    return metrics
