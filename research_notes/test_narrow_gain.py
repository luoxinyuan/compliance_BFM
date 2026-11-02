"""Pure tensor contract test; no BFM or physics execution."""
import torch
from narrow_policy import legal_command
class Sphere:
 def project_z(self,x):return 16*torch.nn.functional.normalize(x,dim=-1)
torch.manual_seed(101)
b=Sphere().project_z(torch.randn(64,256));raw=torch.randn_like(b)*.02
z1,_=legal_command(Sphere(),b,raw)
z16,_=legal_command(Sphere(),b,raw,16)
zscaled,_=legal_command(Sphere(),b,raw*16)
assert torch.equal(z16,zscaled)
assert torch.equal(legal_command(Sphere(),b,torch.zeros_like(b),16)[0],b)
old=raw-(raw*b).sum(-1,keepdim=True)*b/256
old=old*torch.clamp(16/old.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1)
assert torch.equal(z1,Sphere().project_z(b+old))
for x in [raw,raw*1000]:
 z,_=legal_command(Sphere(),b,x,16)
 assert torch.allclose(z.norm(dim=-1),torch.full((64,),16.),atol=1e-5)
 assert ((z*b).sum(-1)/256 >= 2**-.5-1e-5).all()
d=torch.distributions.Normal(torch.zeros_like(raw),torch.full_like(raw,.02))
logp=d.log_prob(raw).sum(-1).clone()
legal_command(Sphere(),b,raw,16)
assert torch.equal(logp,d.log_prob(raw).sum(-1))
print('PASS gain1 backward compatibility; gain16 physical equivalence; zero identity; sphere/cap; raw probability unchanged')
