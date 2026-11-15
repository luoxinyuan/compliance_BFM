"""OR1 engineering only: original macro GAE/PPO with signed Adam parameter backtracking."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from .ei_v2_agent import ContactGoalAgent
from .ei_v2_ppo import segments,advantages,current,retention,kl,device_obs,stack_obs
from .ei_v2_or1_optimizer import one_proposal

class OR1Agent(ContactGoalAgent):
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.actor_accepted=0;self.actor_rejected=0;self.aux_accepted=0;self.aux_rejected=0
    def state(self):
        s=super().state();s['OR1_counts']=dict(actor_accepted=self.actor_accepted,actor_rejected=self.actor_rejected,aux_accepted=self.aux_accepted,aux_rejected=self.aux_rejected);return s
    def load(self,s):
        super().load(s)
        for k,v in s.get('OR1_counts',dict(actor_accepted=0,actor_rejected=0,aux_accepted=0,aux_rejected=0)).items():setattr(self,k,v)

def update(agent,rows,keep,eps,aux_batch,phase):
    assert rows and len(rows)<=1024
    assert all(r.get('eligible_for_joint_PPO',False)and r['collection_role']=='engineering_on_policy' for r in rows)
    groups=segments(rows);ids=list(range(len(groups)));device=agent.device
    agent.selector.requires_grad_(phase=='P3')
    with torch.no_grad():
        new,old=current(agent,rows,groups,ids);err=float(abs(new-old).max());assert err<=3e-4,('original joint logprob assertion',err)
    adv,ret,gae=advantages(agent,rows,groups);lam=float(agent.retention_lambda);metrics=dict(old_logprob_max_error=err,groups=len(groups),rows=len(rows),GAE=gae,steps=[])
    with torch.enable_grad():
        for epoch in range(2):
            order=torch.randperm(len(groups)).tolist()
            for begin in range(0,len(order),64):
                from . import benchmark
                if benchmark.STOP_REQUESTED:raise RuntimeError('OR1_STOP_BEFORE_NEXT_ADAM_TRANSACTION')
                ix=order[begin:begin+64]
                def objective():
                    new,old=current(agent,rows,groups,ix);ratio=(new-old).exp();aa=adv[ix];task=-torch.minimum(ratio*aa,ratio.clamp(.8,1.2)*aa).mean();keep_loss,_=retention(agent,keep,eps);return task+lam*keep_loss
                def evaluate(alpha):
                    new,old=current(agent,rows,groups,ids);r=(new-old).exp();task=-torch.minimum(r*adv,r.clamp(.8,1.2)*adv);kk,vv=retention(agent,keep,eps)
                    return dict(KL=kl(agent,rows,groups),objective=float(task[ix].mean()+lam*kk),minibatch_task_loss=float(task[ix].mean()),full_block_task_loss=float(task.mean()),keep_loss=float(kk),keep_violation=float(vv),lambda_keep=lam,clip_fraction=float(((r-1).abs()>.2).float().mean()),ratio_min=float(r.min()),ratio_max=float(r.max()))
                result=one_proposal(agent.modules,agent.opt,objective,evaluate,commit=True,evaluate_all=False)
                if result['accepted']:agent.accepted+=1;agent.actor_accepted+=1
                else:agent.rejected+=1;agent.actor_rejected+=1
                ob=stack_obs([rows[groups[i][0]]for i in ix]);v=agent.v(device_obs(ob,device));vl=F.mse_loss(v,ret[ix]);agent.vopt.zero_grad(set_to_none=True);vl.backward();vg=float(torch.nn.utils.clip_grad_norm_(agent.value.parameters(),1.));assert math.isfinite(vg);agent.vopt.step()
                result.update(epoch=epoch,minibatch_groups=ix,value_loss=float(vl.detach()));metrics['steps'].append(result)
        for _ in range(4):
            batch=aux_batch();assert batch is not None,'no valid actual auxiliary successor'
            o,z,op,y,mask=batch;saved=copy.deepcopy(agent.modules.state_dict());savedopt=copy.deepcopy(agent.opt.state_dict());al,terms=agent.auxiliary(o,z,op,y,mask)
            agent.opt.zero_grad(set_to_none=True);(.1*al).backward()
            assert all(p.grad is None for p in list(agent.tail.parameters())+list(agent.film.parameters())+list(agent.selector.parameters()))
            ag=float(torch.nn.utils.clip_grad_norm_(agent.modules.parameters(),1.));assert math.isfinite(ag);agent.opt.step();k=kl(agent,rows,groups);ok=math.isfinite(k['joint'])and k['joint']<=.02
            if ok:agent.accepted+=1;agent.aux_accepted+=1
            else:agent.modules.load_state_dict(saved);agent.opt.load_state_dict(savedopt);agent.rejected+=1;agent.aux_rejected+=1
            metrics['steps'].append(dict(auxiliary=True,accepted=ok,KL=k,losses=terms,grad=ag))
    with torch.no_grad():
        _,v=retention(agent,keep,eps);agent.retention_lambda=float(np.clip(lam+.05*(float(v)-1),0,3))
    metrics['next_lambda']=agent.retention_lambda;metrics['actual_actor_accepted_count']=agent.actor_accepted;return metrics
