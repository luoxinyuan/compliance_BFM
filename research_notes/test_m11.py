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

def test_history_same_physical_coordinates():
 import numpy as np
 from scipy.spatial.transform import Rotation
 from m9_history import ContactHistory
 from m11_history import FastHistory
 a=ContactHistory();b=FastHistory();rng=np.random.default_rng(41)
 for i in range(400):
  f=rng.normal(size=3) if i%5 else np.zeros(3);R=Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
  args=(f,'right_wrist_yaw_link' if i%2 else 'torso_link',[.1,0,.2],R,i%7!=0)
  a.append(*args);b.append(*args)
  if i in [0,10,319,399]:
   x=a.features(R);y=b.features(R)
   assert np.allclose(x,y,atol=1e-15)
   assert np.array_equal(x.astype(np.float32),y.astype(np.float32))

if __name__=='__main__':
 test_history_same_physical_coordinates();test_raw_probability_and_legal_projection();test_GAE_never_crosses_terminal_or_worker();print('M11 mock/unit tests passed')
