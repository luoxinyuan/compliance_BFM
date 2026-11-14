"""Issue63 isolated partial-Actor execution, independent of old 29D adapter.
No shared BFM parameter is ever optimized. Calibration lock required for learning.
"""
import copy, math
import torch
from torch import nn
from torch.nn import functional as F
from .schema import deploy,privileged,robot,to_device
from .agent import mlp,norm

class ExecutionAgent:
    def __init__(self,base,obs,goal,scale,arm,u_state,lock):
        assert arm in ('E','I')
        self.base=base; self.goal=goal.clone(); self.scale=scale.clone(); self.arm=arm
        self.lock=copy.deepcopy(lock)
        base.requires_grad_(False) if isinstance(base,nn.Module) else None
        base._actor.requires_grad_(False)
        assert len(base._actor.policy)==7,'Architecture contract differs'
        self.tail=copy.deepcopy(base._actor.policy[5:]).requires_grad_(True)
        self.target_tail=copy.deepcopy(self.tail).requires_grad_(False)
        ds=deploy(obs).shape[-1]; dp=privileged(obs).shape[-1]
        self.nets=nn.ModuleDict(dict(u=mlp(ds+256,256,True),goal_delta=mlp(32,256,True),
            q1=mlp(dp+29+256+32,1),q2=mlp(dp+29+256+32,1),goal_encoder=mlp(16,32)))
        self.nets['u'].load_state_dict(u_state,strict=True)
        if arm=='I':
            self.nets['bi']=mlp(16,32);self.nets['fi']=mlp(dp+29+256+32,32)
        self.targets=copy.deepcopy(self.nets).requires_grad_(False)
        self.opts={k:torch.optim.Adam(v.parameters(),lr=lock['lr']) for k,v in self.nets.items()}
        self.opts['actor_tail']=torch.optim.Adam(self.tail.parameters(),lr=lock['lr'])
        self.prompt=torch.zeros(1,16,device=goal.device)
        self.exploration_std=.01;self.exploration_clip=.06;self.envelope=None;self.last_action_audit=None
        self.updates=0;self.z_buffer=[];self.audit=[]
        self.retention_lambda=float(lock['retention_lambda'])
    def zi(self,n,target=False):
        nets=self.targets if target else self.nets
        key='bi' if self.arm=='I' else 'goal_encoder'
        return norm(nets[key](self.prompt)).expand(n,-1)
    def command(self,obs,target=False):
        nets=self.targets if target else self.nets;x=deploy(obs);g=self.goal.expand(len(x),-1)
        return (g+nets['u'](torch.cat([x,g],-1))+nets['goal_delta'](self.zi(len(x),target)))/16
    def mean_at_z(self,obs,z,target=False):
        actor=self.base._actor;o=actor.input_filter(self.base._normalize(robot(obs)))
        h=torch.cat([actor.embed_s(o),actor.embed_z(torch.cat([o,z],-1))],-1)
        for layer in actor.policy[:5]:h=layer(h)
        return torch.tanh((self.target_tail if target else self.tail)(h))
    def action(self,obs,target=False,sample=False,z_override=None):
        from clean_policy import direct_command
        from locomotion_rollout import spherical
        raw=self.command(obs,target);new=direct_command(raw,self.goal);zz=[]
        for i in range(len(raw)):
            slot=int(obs['_slot'][i,0]);tgt=new[i:i+1] if slot==0 else obs['_target_z'][i:i+1]
            zz.append(spherical(obs['_previous_z'][i:i+1],tgt,(slot+1)/5))
        z=torch.cat(zz) if z_override is None else z_override
        mean=self.mean_at_z(obs,z,target);a=mean
        if sample:a=(mean+(self.exploration_std*torch.randn_like(mean)).clamp(-self.exploration_clip,self.exploration_clip)).clamp(-1,1)
        return a,z,self.zi(len(raw),target)
    def inp(self,o,a,z,zi):return torch.cat([privileged(o),a,z,zi],-1)
    def state(self):
        return dict(version='R2-execution-v1',arm=self.arm,lock=self.lock,tail=self.tail.state_dict(),target_tail=self.target_tail.state_dict(),nets=self.nets.state_dict(),targets=self.targets.state_dict(),optimizers={k:v.state_dict() for k,v in self.opts.items()},goal=self.goal,scale=self.scale,prompt=self.prompt,updates=self.updates,z_buffer=self.z_buffer,audit=self.audit,retention_lambda=self.retention_lambda,normalizers=self.base._obs_normalizer.state_dict(),exploration=dict(std=.01,clip=.06))
    def load(self,s):
        assert s['version']=='R2-execution-v1' and s['arm']==self.arm and s['lock']==self.lock
        self.tail.load_state_dict(s['tail']);self.target_tail.load_state_dict(s['target_tail'])
        self.nets.load_state_dict(s['nets']);self.targets.load_state_dict(s['targets'])
        for k,v in s['optimizers'].items():self.opts[k].load_state_dict(v)
        for k in ('goal','scale','prompt'):setattr(self,k,s[k].cuda())
        for k in ('updates','z_buffer','audit','retention_lambda'):setattr(self,k,s[k])
        for k,v in s['normalizers'].items():assert torch.equal(v.cuda(),self.base._obs_normalizer.state_dict()[k])
    def _step(self,keys,loss):
        for k in keys:self.opts[k].zero_grad(set_to_none=True)
        loss.backward();stats={}
        for k in keys:
            params=self.tail.parameters() if k=='actor_tail' else self.nets[k].parameters()
            grads=[p.grad for p in params if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            stats[k]=sum(float(g.square().sum()) for g in grads)**.5
            self.opts[k].step()
        return stats
    def update(self,rows,native,actor_update=False):
        assert self.lock['status'] in ('engineering_only','locked_training')
        from dual_track_fb_online import stack
        ix=torch.randint(len(rows),(self.lock['batch'],));b=stack([rows[i] for i in ix])
        o=to_device(b['observation']);no=to_device(b['next']['observation']);a=b['action'].cuda();z=b['z'].cuda();zi=b['zi'].cuda()
        d=(.99**(b['duration'].cuda()/.1))*(~b['next']['terminated'].cuda());r=b['aux_rewards']['contact_physical'].cuda()
        with torch.no_grad():
            na,nz,ni=self.action(no,True)
            target=r+d*torch.minimum(self.targets['q1'](self.inp(no,na,nz,ni)),self.targets['q2'](self.inp(no,na,nz,ni)))
        losses=[F.mse_loss(self.nets[k](self.inp(o,a,z,zi)),target) for k in ('q1','q2')]
        grads=self._step(['q1','q2'],sum(losses));rec=dict(update=self.updates+1,indices=ix.tolist(),q_losses=[float(x.detach()) for x in losses],actor_updated=actor_update,gradient_norms=grads)
        if self.arm=='I':
            # Interaction-only successor density, no stale frozen F_R term.
            y=no['contact_privileged_v1'][:,6:22]
            with torch.no_grad():
                tb=norm(self.targets['bi'](y));tf=self.targets['fi'](self.inp(no,na,nz,ni));t=d*(tf@tb.T/32)
            bi=norm(self.nets['bi'](y));fi=self.nets['fi'](self.inp(o,a,z,zi));density=fi@bi.T/32
            off=~torch.eye(len(a),device=a.device,dtype=torch.bool)
            fb=.5*(density-t)[off].square().mean()-density.diag().mean()
            orth=(bi.T@bi/len(a)-torch.eye(32,device=a.device)).square().mean()
            rec['representation_loss']=float((fb+.1*orth).detach())
            rec['gradient_norms'].update(self._step(['fi','bi'],fb+.1*orth))
        if actor_update:
            assert native,'No eligible same-z retention data'
            ex=torch.randint(len(native),(self.lock['batch'],));e=stack([{k:native[i][k] for k in ('observation','z')} for i in ex]);eo=to_device(e['observation']);ez=e['z'].cuda()
            for k in ('q1','q2','fi'):
                if k in self.nets:self.nets[k].requires_grad_(False)
            aa,zz,ii=self.action(o)
            val=torch.minimum(self.nets['q1'](self.inp(o,aa,zz,ii)),self.nets['q2'](self.inp(o,aa,zz,ii))).mean()
            with torch.no_grad():old=self.base._actor(self.base._normalize(robot(eo)),ez,self.base.cfg.actor_std).mean
            delta=(self.mean_at_z(eo,ez)-old)*self.scale
            keep=delta.square().mean()
            sf=(self.nets['fi'](self.inp(o,aa,zz,ii))*ii).sum(-1).mean()/32 if self.arm=='I' else val*0
            objective=-val+self.retention_lambda*keep-self.lock['sf_coefficient']*sf
            keys=['actor_tail','u','goal_delta']+(['bi'] if self.arm=='I' else ['goal_encoder'])
            rec['gradient_norms'].update(self._step(keys,objective))
            rec.update(actor_loss=float(objective.detach()),q_mean=float(val.detach()),retention_qpd_rms=float(keep.detach().sqrt()),retention_indices=ex.tolist())
            for k in ('q1','q2','fi'):
                if k in self.nets:self.nets[k].requires_grad_(True)
        with torch.no_grad():
            for k in self.nets:
                for t,p in zip(self.targets[k].parameters(),self.nets[k].parameters()):t.lerp_(p,.01)
            for t,p in zip(self.target_tail.parameters(),self.tail.parameters()):t.lerp_(p,.01)
        self.updates+=1;self.audit.append(rec);return rec
