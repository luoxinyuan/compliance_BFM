"""Approved E/I Q-only warm-up. Actor path remains gated separately."""
import copy
import torch
from torch.nn import functional as F
from .approved_agent import ApprovedAgent
from .schema import to_device
from .agent import norm

class ApprovedLearner(ApprovedAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.q_steps=0;self.representation_steps=0;self.actor_steps=0
        self.draws={k:torch.Generator(device='cpu').manual_seed(self.lock['initialization_seed']+v)
                    for k,v in [('q',101),('representation',102),('actor',103)]}
        self.gates={'q':False,'retention':False,'storage':False,'sf':False}
    def state(self):
        s=super().state();s.update(version='R2-approved-learning-v1',q_steps=self.q_steps,
            representation_steps=self.representation_steps,actor_steps=self.actor_steps,
            draws={k:v.get_state() for k,v in self.draws.items()},gates=copy.deepcopy(self.gates))
        return s
    def load(self,s):
        assert s['version']=='R2-approved-learning-v1' and s['arm']==self.arm and s['lock']==self.lock
        self.tail.load_state_dict(s['tail'],strict=True);self.target_tail.load_state_dict(s['target_tail'],strict=True)
        self.nets.load_state_dict(s['nets'],strict=True);self.targets.load_state_dict(s['targets'],strict=True)
        for k,v in s['optimizers'].items():self.opts[k].load_state_dict(v)
        for k in ('goal','scale','prompt'):setattr(self,k,s[k].cuda())
        for k in ('updates','z_buffer','audit','retention_lambda','u_retention_lambda','sf_coefficient','q_steps','representation_steps','actor_steps','gates'):setattr(self,k,copy.deepcopy(s[k]))
        for k,v in s['draws'].items():self.draws[k].set_state(v.cpu())
        for k,v in s['normalizers'].items():assert torch.equal(v.cuda(),self.base._obs_normalizer.state_dict()[k])
    def batch(self,rows,kind):
        from dual_track_fb_online import stack
        ix=torch.randint(len(rows),(64,),generator=self.draws[kind]);b=stack([rows[i] for i in ix])
        return ix,b
    def ema(self,keys):
        with torch.no_grad():
            for k in keys:
                for t,p in zip(self.targets[k].parameters(),self.nets[k].parameters()):t.lerp_(p,.01)
    def q_update(self,rows):
        ix,b=self.batch(rows,'q');o=to_device(b['observation']);no=to_device(b['next']['observation'])
        a=b['action'].cuda();z=b['z'].cuda();zi=b['zi'].cuda()
        r=b['aux_rewards']['contact_physical'].cuda();assert (r<=0).all()
        d=(.99**(b['duration'].cuda()/.1))*(~b['next']['terminated'].cuda())
        with torch.no_grad():
            na,nz,ni=self.action(no,target=True)
            target=r+d*torch.minimum(self.targets['q1'](self.inp(no,na,nz,ni)),self.targets['q2'](self.inp(no,na,nz,ni)))
        pred=[self.nets[k](self.inp(o,a,z,zi)) for k in ('q1','q2')]
        assert all((p<=0).all() for p in pred) and (target<=0).all()
        losses=[F.mse_loss(p,target) for p in pred]
        grads=self._step(['q1','q2'],sum(losses));self.ema(['q1','q2']);self.q_steps+=1;self.updates+=1
        rec=dict(kind='q',step=self.q_steps,indices=ix.tolist(),losses=[float(v.detach()) for v in losses],
                 q_mean=[float(v.mean().detach()) for v in pred],target_mean=float(target.mean()),
                 disagreement=float((pred[0]-pred[1]).abs().mean().detach()),gradient_norms=grads)
        self.audit.append(rec);return rec
    def representation_update(self,rows):
        assert self.arm=='I'
        ix,b=self.batch(rows,'representation');o=to_device(b['observation']);no=to_device(b['next']['observation'])
        a=b['action'].cuda();z=b['z'].cuda();zi=b['zi'].cuda()
        d=(.99**(b['duration'].cuda()/.1))*(~b['next']['terminated'].cuda())
        y=no['contact_privileged_v1'][:,6:22]
        with torch.no_grad():
            na,nz,ni=self.action(no,target=True)
            tb=norm(self.targets['bi'](y));tf=self.targets['fi'](self.inp(no,na,nz,ni));target=d*(tf@tb.T/32)
        bi=norm(self.nets['bi'](y));fi=self.nets['fi'](self.inp(o,a,z,zi));density=fi@bi.T/32
        off=~torch.eye(len(a),device=a.device,dtype=torch.bool)
        fb=.5*(density-target)[off].square().mean()-density.diag().mean()
        orth=(bi.T@bi/len(a)-torch.eye(32,device=a.device)).square().mean()
        grad=self._step(['bi','fi'],fb+.1*orth);self.ema(['bi','fi']);self.representation_steps+=1
        rec=dict(kind='representation',step=self.representation_steps,indices=ix.tolist(),
                 successor=float(fb.detach()),anticollapse=float(orth.detach()),coefficient=.1,gradient_norms=grad)
        self.audit.append(rec);return rec
    def update_block_q_only(self,rows,block):
        assert len(rows)==block*250
        out=[self.q_update(rows) for _ in range(4)]
        if self.arm=='I' and block>1:out.extend(self.representation_update(rows) for _ in range(2))
        assert self.actor_steps==0
        return out
