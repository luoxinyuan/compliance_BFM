"""NVIDIA-only bounded recurrence of original batched FP32 probability contract."""
import argparse,copy,json,time
from pathlib import Path

def main():
 import torch
 from native_policy import Expert
 from clean_policy import update
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(4800)
 assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert 'NVIDIA' in torch.cuda.get_device_name(0)
 rows=[]
 for scale in [.03,.3,1.]:
  n=Expert(1088,'direct').cuda()
  if scale==.3:
   dirs=torch.linalg.qr(torch.randn(256,4,device='cuda'))[0].T;n.set_covariance(dirs)
  with torch.no_grad():
   n.actor[-1].weight.normal_(0,.01);n.actor[-1].bias.normal_(0,.05)
   x=torch.randn(768,1088,device='cuda')*scale
   ds=[n.distribution(t[None]) for t in x];raw=torch.cat([d.sample() for d in ds]);old=torch.cat([d.log_prob(r[None]).sum(-1) for d,r in zip(ds,raw)])
   d=n.distribution(x);err=float((d.log_prob(raw).sum(-1)-old).abs().max());assert err<3e-4,(scale,err)
   values=n.value(x);ret=values+torch.randn_like(values);adv=torch.randn_like(values);adv=(adv-adv.mean())/adv.std()
  opt=torch.optim.Adam([{'params':n.actor.parameters(),'lr':1e-5},{'params':n.critic.parameters(),'lr':3e-4}])
  u=update(n,opt,{'obs':x,'action':raw,'logp':old,'advantages':adv,'returns':ret,'value':values,'raw_adv_std':adv.std()},epochs=2)
  # Probe the pre-existing rejection threshold rather than raising it.
  with torch.no_grad():
   before={k:v.clone() for k,v in n.state_dict().items()};adam_before=copy.deepcopy(opt.state_dict());om=n.distribution(x).mean.clone();st=n.log_std.exp().clone();n.actor[-1].bias.add_(.05)
   kl=torch.distributions.kl_divergence(torch.distributions.Normal(om,st),n.distribution(x)).sum(-1).mean();assert kl>.02
   n.load_state_dict(before);opt.load_state_dict(adam_before);assert all(torch.equal(v,before[k]) for k,v in n.state_dict().items())
  rows.append({'input_scale':scale,'batch_one_vs_batched_joint_logprob_max':err,'update':u,'forced_rejection_KL':float(kl),'rollback_parameters_equal':True,'original_logprob_threshold':3e-4,'original_joint_KL_limit':.02})
 a.output.write_text(json.dumps({'status':'passed','device':torch.cuda.get_device_name(0),'torch':torch.__version__,'FP32':True,'TF32':False,'PPU_rowwise_path_used':False,'rows':rows},indent=2));print('NVIDIA_PROBABILITY_PASS',flush=True)
if __name__=='__main__':main()
