"""Regression: default new guard is numerically identical; invalid probability fails."""
import os,copy,torch
os.environ['P4000_CANONICAL_ROW_ACTOR']='1'
from clean_policy import Expert,update as old_update
from dual_track_ppo import update
from dual_track_runtime import PPOContract,load_ppo_contract

def main():
 assert load_ppo_contract()==PPOContract()
 torch.set_num_threads(1);torch.manual_seed(5510)
 n=Expert(1639,'direct').cuda();n.actor[-1].bias.data.normal_(std=.01);nn=copy.deepcopy(n)
 x=torch.randn(12,1639,device='cuda');d=n.distribution(x);a=d.sample();lp=d.log_prob(a).sum(-1).detach();v=n.value(x).detach()
 b=dict(obs=x,action=a,logp=lp,advantages=torch.randn(12,device='cuda'),returns=v+torch.randn_like(v),value=v,raw_adv_std=torch.ones((),device='cuda'))
 def opt(net):return torch.optim.Adam([{'params':net.actor.parameters(),'lr':3e-5},{'params':net.critic.parameters(),'lr':3e-4}])
 torch.manual_seed(5511);o=old_update(n,opt(n),b,epochs=1,minibatch=12)
 torch.manual_seed(5511);r=update(nn,opt(nn),b,epochs=1,minibatch=12,contract=PPOContract())
 assert all(torch.equal(t,nn.state_dict()[k]) for k,t in n.state_dict().items())
 assert o==r
 bad=dict(b,logp=lp+1.)
 try:update(nn,opt(nn),bad,epochs=1,minibatch=12)
 except AssertionError:pass
 else:raise AssertionError('probability mismatch accepted')
 from p4000_critic import update as old_separate
 from dual_track_critic import update as new_separate
 left=Expert(1639,'direct').cuda();right=copy.deepcopy(left)
 with torch.no_grad():
  dist=left.distribution(x);aa=dist.sample();vv=left.value(x)
 bb=dict(b,action=aa,logp=dist.log_prob(aa).sum(-1).detach(),value=vv,returns=vv+torch.ones_like(vv))
 torch.manual_seed(5512);ll=old_separate(left,opt(left),bb,epochs=1,minibatch=12,critic_shuffle_seed=5513)
 torch.manual_seed(5512);rr=new_separate(right,opt(right),bb,epochs=1,minibatch=12,critic_shuffle_seed=5513,contract=PPOContract())
 assert all(torch.equal(t,right.state_dict()[k]) for k,t in left.state_dict().items())
 assert ll==rr
 print('PASS independent-critic default parity',flush=True)
 print('PASS default numeric parity and original logprob rejection',flush=True)
if __name__=='__main__':main()
