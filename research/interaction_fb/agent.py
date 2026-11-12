"""Independent residual successor-density factorization; original BFM immutable.
Six optimizers: B_I, F_I, Guidance, execution adapter, physical Q, robot CPR discriminator.
"""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from .schema import deploy,privileged,interaction,robot,to_device

def mlp(i,o,zero=False):
    m=nn.Sequential(nn.Linear(i,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,o))
    if zero:nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
    return m.cuda()
def norm(x):return math.sqrt(x.shape[-1])*F.normalize(x,dim=-1)
class Agent:
    def __init__(self,original,obs,goal,scale):
        self.base=original;self.goal=goal.clone();self.scale=scale.clone()
        self.fr=copy.deepcopy(original._forward_map)
        self.fr._forward_pre_hooks.clear();self.fr.cuda().eval().requires_grad_(False)
        ds=deploy(obs).shape[-1];dp=privileged(obs).shape[-1];di=interaction(obs).shape[-1]
        self.nets=nn.ModuleDict(dict(bi=mlp(di,32),fi=mlp(dp+29+256+32,32),
            u=mlp(ds+256,256,True),adapter=mlp(ds+256+32,29,True),
            q=mlp(dp+29+256+32,1),disc=mlp(obs['state'].shape[-1]+256,1)))
        self.targets=copy.deepcopy(self.nets).requires_grad_(False)
        self.opts={k:torch.optim.Adam(v.parameters(),lr=1e-4) for k,v in self.nets.items()}
        # Declared public goal prototype, NEVER current privileged state.
        self.prompt=interaction(obs).detach().clone();self.prompt[:,-16:]=0;self.prompt[:,-1]=1
        self.updates=0;self.z_buffer=[];self.audit=[]
    def zi(self,n,target=False):
        nets=self.targets if target else self.nets
        with torch.no_grad():return norm(nets['bi'](self.prompt)).expand(n,-1)
    def command(self,obs,target=False):
        nets=self.targets if target else self.nets;x=deploy(obs);g=self.goal.expand(len(x),-1)
        return norm(g+nets['u'](torch.cat([x,g],-1)))
    def action(self,obs,target=False,sample=False,z_override=None):
        nets=self.targets if target else self.nets;x=deploy(obs);g=self.goal.expand(len(x),-1)
        new=self.command(obs,target)
        from locomotion_rollout import spherical
        zz=[]
        for i in range(len(x)):
            slot=int(obs['_slot'][i,0]);tgt=new[i:i+1] if slot==0 else obs['_target_z'][i:i+1]
            zz.append(spherical(obs['_previous_z'][i:i+1],tgt,(slot+1)/5))
        z=torch.cat(zz);z=z if z_override is None else z_override
        zi=self.zi(len(x),target)
        # Use underlying frozen actor, preserving gradient back to guidance.
        a=self.base._actor(self.base._normalize(robot(obs)),z,self.base.cfg.actor_std).mean
        a=(a+nets['adapter'](torch.cat([x,z,zi],-1))).clamp(-1,1)
        if sample:a=(a+(.05*torch.randn_like(a)).clamp(-.3,.3)).clamp(-1,1)
        return a,z,zi
    def inp(self,o,a,z,zi):return torch.cat([privileged(o),a,z,zi],-1)
    def stepopt(self,name,loss):
        op=self.opts[name];op.zero_grad(set_to_none=True);loss.backward()
        gradients=[p.grad for p in self.nets[name].parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients),(name,'nonfinite gradient')
        size=sum(float(g.square().sum()) for g in gradients)**.5
        assert size>0,(name,'zero dummy gradient')
        op.step();return size
    def update(self,rows,native,batch=32):
        from dual_track_fb_online import stack
        ix=torch.randint(len(rows),(batch,));ex=torch.randint(len(native),(batch,))
        b=stack([rows[i] for i in ix]);e=stack([native[i] for i in ex])
        o=to_device(b['observation']);no=to_device(b['next']['observation']);eo=to_device(e['observation'])
        a=b['action'].cuda();z=b['z'].cuda();zi=b['zi'].cuda()
        d=(.99**(b['duration'].cuda()/.1))*(~b['next']['terminated'].cuda())
        rw=b['aux_rewards']['contact_physical'].cuda()
        with torch.no_grad():
            na,nz,nzi=self.action(no,True)
            tb=norm(self.targets['bi'](interaction(no)))
            tf=self.targets['fi'](self.inp(no,na,nz,nzi))
            br=self.base.backward_map(robot(no))/16
            fr=self.fr(self.base._normalize(robot(o)),z,a).mean(0)/16
            nfr=self.fr(self.base._normalize(robot(no)),nz,na).mean(0)/16
            target=d*(nfr@br.T+tf@tb.T/32)
        fi=self.nets['fi'](self.inp(o,a,z,zi));bi=norm(self.nets['bi'](interaction(no)))
        density=fr@br.T+fi@bi.T/32
        off=~torch.eye(batch,device='cuda',dtype=torch.bool)
        residual=density-target
        # Discounted successor-density Bellman residual + diagonal occupancy source.
        fb=.5*residual[off].square().mean()-density.diag().mean()
        gram=bi.T@bi/batch
        orth=(gram-torch.eye(32,device='cuda')).square().mean()
        loss=fb+.1*orth
        self.opts['fi'].zero_grad(set_to_none=True);self.opts['bi'].zero_grad(set_to_none=True);loss.backward()
        grads={}
        for k in ('fi','bi'):
            gg=[p.grad for p in self.nets[k].parameters() if p.grad is not None]
            assert gg and all(torch.isfinite(g).all() for g in gg)
            grads[k]=sum(float(g.square().sum()) for g in gg)**.5
            assert grads[k]>0;self.opts[k].step()
        with torch.no_grad():qt=rw+d*self.targets['q'](self.inp(no,na,nz,nzi))
        qloss=F.mse_loss(self.nets['q'](self.inp(o,a,z,zi)),qt)
        grads['q']=self.stepopt('q',qloss)
        ez=e['z'].cuda()
        logits_e=self.nets['disc'](torch.cat([eo['state'],ez],-1))
        logits_t=self.nets['disc'](torch.cat([o['state'],z],-1))
        # CPR adversarial objective robot-only; not a contact reward head.
        dl=F.softplus(-logits_e).mean()+F.softplus(logits_t).mean()
        grads['disc']=self.stepopt('disc',dl)
        for k in ('fi','q','disc'):self.nets[k].requires_grad_(False)
        aa,zz,ii=self.action(o);ea,_,_=self.action(eo)
        with torch.no_grad():old=self.base.act(robot(eo),ez,True)
        cpr=(((ea-old)*self.scale)**2).mean()/.01
        val=self.nets['q'](self.inp(o,aa,zz,ii)).mean()
        sf=(self.nets['fi'](self.inp(o,aa,zz,ii))*ii).sum(-1).mean()/32
        adv=F.softplus(-self.nets['disc'](torch.cat([o['state'],zz],-1))).mean()
        actor=-val-.01*sf+.1*cpr+.01*adv
        for k in ('u','adapter'):self.opts[k].zero_grad(set_to_none=True)
        actor.backward()
        for k in ('u','adapter'):
            gg=[p.grad for p in self.nets[k].parameters() if p.grad is not None]
            assert gg and all(torch.isfinite(g).all() for g in gg)
            grads[k]=sum(float(g.square().sum()) for g in gg)**.5
            assert grads[k]>0;self.opts[k].step()
        for k in ('fi','q','disc'):self.nets[k].requires_grad_(True)
        with torch.no_grad():
            for key in self.nets:
                for t,p in zip(self.targets[key].parameters(),self.nets[key].parameters()):t.lerp_(p,.01)
        self.updates+=1
        self.z_buffer.append(zi.detach().cpu());self.z_buffer=self.z_buffer[-64:]
        metric=dict(update=self.updates,fb=float(loss.detach()),q=float(qloss.detach()),actor=float(actor.detach()),
            cpr=float(cpr.detach()),disc=float(dl.detach()),gradient_norms=grads,
            replay_indices=ix.tolist(),expert_indices=ex.tolist())
        self.audit.append(metric);return metric
    def state(self):
        return dict(nets=self.nets.state_dict(),targets=self.targets.state_dict(),
            optimizers={k:v.state_dict() for k,v in self.opts.items()},prompt=self.prompt,goal=self.goal,
            scale=self.scale,z_buffer=self.z_buffer,updates=self.updates,audit=self.audit,
            normalizers=self.base._obs_normalizer.state_dict())
    def load(self,s):
        self.nets.load_state_dict(s['nets']);self.targets.load_state_dict(s['targets'])
        for k,v in s['optimizers'].items():self.opts[k].load_state_dict(v)
        for k in ('prompt','goal','scale'):setattr(self,k,s[k].cuda())
        self.z_buffer=s['z_buffer'];self.updates=s['updates'];self.audit=s['audit']
        for k,v in s['normalizers'].items():assert torch.equal(v.cuda(),self.base._obs_normalizer.state_dict()[k])
