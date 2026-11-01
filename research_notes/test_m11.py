import torch
from m11_policy import Expert,legal_command,update
from humanoidverse.force_compliance.ppo import advantages

def test_raw_probability_and_legal_projection():
 class Frozen:
  def project_z(self,z):return 16*z/z.norm(dim=-1,keepdim=True)
 torch.manual_seed(2);net=Expert(7);x=torch.randn(128,7);d=net.distribution(x);a=d.sample();lp=d.log_prob(a).sum(-1).detach();base=Frozen().project_z(torch.randn(128,256))
 z,_=legal_command(Frozen(),base,a);assert torch.allclose(z.norm(dim=-1),torch.full((128,),16.),atol=3e-6)
 z0,_=legal_command(Frozen(),base,torch.zeros_like(a));assert torch.allclose(z0,base,atol=1e-6)
 assert torch.isfinite(z).all();assert torch.allclose(net.distribution(x).log_prob(a).sum(-1),lp)
 # Forced transformed coordinates differ; PPO must use original raw samples.
 assert not torch.allclose(net.distribution(x).log_prob(z).sum(-1),lp)
 b=dict(obs=x,action=a,logp=lp,value=net.value(x).detach(),advantages=torch.randn(128),returns=torch.randn(128),raw_adv_std=torch.tensor(1.))
 m=update(net,torch.optim.Adam(net.parameters(),lr=1e-5),b,epochs=1)
 assert m['pre_ratio_max_error']<1e-5 and m['epochs'][0]['joint_gaussian_KL']>=0

def test_GAE_never_crosses_terminal_or_worker():
 r=torch.tensor([[1.,10.],[2.,20.],[3.,30.]]);v=torch.zeros_like(r);done=torch.tensor([[False,False],[True,False],[False,True]])
 a,ret=advantages(r,v,v,torch.ones_like(r),done,1.)
 assert torch.equal(a,torch.tensor([[3.,60.],[2.,50.],[3.,30.]]));assert torch.equal(a,ret)

if __name__=='__main__':
 test_raw_probability_and_legal_projection();test_GAE_never_crosses_terminal_or_worker();print('M11 mock/unit tests passed')
