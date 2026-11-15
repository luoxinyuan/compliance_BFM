"""Approved Contact-Goal E/I v2 policy: finite commands, contact FiLM, native tail.
No historical high-level weights, scalar action Q or external teacher.
"""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from .schema import deploy,privileged,robot

SIGMA=.01
SCALES=(.1,)*9+(10.,)*3+(.5,)*6+(.35,.45,15.,1.,1.,.02)

def mlp(inp,width,out,zero=False):
    m=nn.Sequential(nn.Linear(inp,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,out))
    if zero:nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
    return m

class ContactGoalAgent:
    def __init__(self,base,ob,arm,device):
        assert arm in ('E','I');self.base=base;self.arm=arm;self.device=device;self.scales=torch.tensor(SCALES,device=device)[None]
        for name in ('_actor','_forward_map','_backward_map','_critic','_aux_critic','_target_forward_map','_target_backward_map','_target_critic','_target_aux_critic','_discriminator'):
            module=getattr(base,name,None)
            if isinstance(module,nn.Module):module.requires_grad_(False)
        base._obs_normalizer.requires_grad_(False);base._obs_normalizer.eval()
        assert len(base._actor.policy)==7
        self.tail=copy.deepcopy(base._actor.policy[5:]).to(device).requires_grad_(True)
        self.ds=deploy(ob).shape[-1];self.dp=self.value_input(ob).shape[-1]-50
        with torch.no_grad():
            v=self.frozen_features(ob,ob['_previous_z']);self.width=v.shape[-1]
        # Draw every common component before the differing prediction output.
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(660301)
            self.goal_encoder=nn.Sequential(nn.Linear(24,32),nn.SiLU(),nn.Linear(32,16)).to(device)
            self.film=mlp(self.ds+50+16+16,128,2*self.width,True).to(device)
            self.selector=mlp(self.ds+50+4*16+5,128,4,True).to(device)
            self.value=nn.Sequential(nn.Linear(self.dp+50,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1)).to(device)
            self.predictor=mlp(self.ds+50+4+256,128,3*(24 if arm=='E' else 16)).to(device)
            self.decoder=nn.Sequential(nn.Linear(16,32),nn.SiLU(),nn.Linear(32,24)).to(device) if arm=='I' else None
        self.modules=nn.ModuleDict(dict(tail=self.tail,goal_encoder=self.goal_encoder,film=self.film,selector=self.selector,predictor=self.predictor))
        if self.decoder is not None:self.modules['decoder']=self.decoder
        self.opt=torch.optim.Adam(self.modules.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.vopt=torch.optim.Adam(self.value.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.retention_lambda=0.;self.accepted=0;self.rejected=0;self.scales=torch.tensor(SCALES,device=device)[None]
    def goal(self,o):
        return o['_goal_y']/self.scales,o['_goal_mask'],o['_goal_mode']
    def public(self,o):return torch.cat(self.goal(o),-1)
    def frozen_features(self,o,z):
        a=self.base._actor;x=a.input_filter(self.base._normalize(robot(o)))
        h=torch.cat((a.embed_s(x),a.embed_z(torch.cat((x,z),-1))),-1)
        for layer in a.policy[:5]:h=layer(h)
        return h
    def context(self,o,z,option):
        x=torch.cat((deploy(o),self.public(o),F.one_hot(option,4).to(z.dtype),z),-1)
        pred=self.predictor(x).reshape(len(z),3,-1)
        ctx=self.goal_encoder(pred.reshape(-1,24)).reshape(len(z),3,16).mean(1) if self.arm=='E' else pred.mean(1)
        return pred,ctx
    def mean(self,o,z,option):
        # Same FP32 batch-one execution used by the real single-env sampler.
        # Preserve autograd; do not loosen logprob assertions to use a different numerical forward.
        if len(z)>1:
            return torch.cat([self.mean({k:v[i:i+1] for k,v in o.items()},z[i:i+1],option[i:i+1]) for i in range(len(z))],0)
        h=self.frozen_features(o,z);gy,mask,mode=self.goal(o)
        _,ctx=self.context(o,o.get('_first_interpolated_z',z),option)
        cond=torch.cat((deploy(o),self.public(o),self.goal_encoder(gy),ctx),-1)
        gamma,beta=self.film(cond).chunk(2,-1)
        return torch.tanh(self.tail((1.+gamma)*h+beta))
    def logits(self,o,candidate_first_z):
        contexts=[]
        for j in range(4):
            opt=torch.full((len(candidate_first_z),),j,device=self.device,dtype=torch.long)
            contexts.append(self.context(o,candidate_first_z[:,j],opt)[1])
        own=o['_option_memory'];x=torch.cat((deploy(o),self.public(o),*contexts,own),-1)
        raw=self.selector(x);follow=o['_goal_mode'][:,0:1]
        pf=torch.tensor((.10,.10,.70,.10),device=self.device);ph=torch.tensor((.97,.01,.01,.01),device=self.device)
        return raw+torch.log(follow*pf+(1.-follow)*ph)
    def value_input(self,o):
        # State V sees current C-P and public Goal, never the just-selected target z/slot.
        # The legacy action-Q helper privileged() includes those action-dependent fields.
        c=o['contact_privileged_v1']
        return torch.cat((deploy(o),o['privileged_state'],c[:,9:15],c[:,18:22],self.public(o)),-1)
    def v(self,o):return self.value(self.value_input(o)).squeeze(-1)
    def sample(self,o,z,option,stochastic=True):
        mu=self.mean(o,z,option);u=mu+SIGMA*torch.randn_like(mu) if stochastic else mu
        return u,u.clamp(-1.,1.),mu,self.logp(u,mu)
    @staticmethod
    def logp(u,mu):return (-.5*((u-mu)/SIGMA).square()-math.log(SIGMA)-.5*math.log(2*math.pi)).sum(-1)
    def auxiliary(self,o,z,option,y,mask):
        # y is already standardized 3x24, mask contains real horizon validity.
        pred,_=self.context(o,o.get('_first_interpolated_z',z),option);w=mask.to(y.dtype)
        def avg(x):return (x*w).sum()/w.sum().clamp_min(1)
        if self.arm=='E':return avg((pred-y).square().mean(-1)),{'outcome':float(avg((pred-y).square().mean(-1)).detach())}
        b=self.goal_encoder(y.reshape(-1,24)).reshape(len(y),3,16)
        decoded=self.decoder(pred);recon=self.decoder(b)
        physical=avg((decoded-y).square().mean(-1));successor=avg((pred-b.detach()).square().mean(-1));reconstruct=avg((recon-y).square().mean(-1))
        active=b[mask.bool()]
        if len(active)>1:
            std=active.std(0,unbiased=False);var=F.relu(1-std).square().mean();center=active-active.mean(0);cov=center.T@center/(len(active)-1);off=~torch.eye(16,device=y.device,dtype=torch.bool);covloss=cov[off].square().mean()
        else:var=b.sum()*0;covloss=b.sum()*0
        loss=physical+successor+reconstruct+.1*(var+covloss)
        return loss,{k:float(v.detach()) for k,v in dict(decoded=physical,successor=successor,reconstruct=reconstruct,variance=var,covariance=covloss).items()}
    def state(self):
        return dict(version='contact-goal-ei-v2',arm=self.arm,modules=self.modules.state_dict(),value=self.value.state_dict(),optimizer=self.opt.state_dict(),value_optimizer=self.vopt.state_dict(),retention_lambda=self.retention_lambda,accepted=self.accepted,rejected=self.rejected,normalizer=self.base._obs_normalizer.state_dict())
    def load(self,s):
        assert s['version']=='contact-goal-ei-v2' and s['arm']==self.arm
        self.modules.load_state_dict(s['modules'],strict=True);self.value.load_state_dict(s['value'],strict=True);self.opt.load_state_dict(s['optimizer']);self.vopt.load_state_dict(s['value_optimizer'])
        for k in ('retention_lambda','accepted','rejected'):setattr(self,k,s[k])
        for k,v in self.base._obs_normalizer.state_dict().items():assert torch.equal(v,s['normalizer'][k].to(v.device))
