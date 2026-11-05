"""Same-state exploration directions plus finite stochastic physical baseline."""
import json,gzip,time
from pathlib import Path
import transfer_student_validated as runner
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105')
def study(env,model,contract,out):
 import torch,numpy as np
 from adapter_evaluate import setup,conditions,episode
 from adapter_policy import load
 from clean_policy import direct_command
 from m39b_replay import interp
 task,I,goal=setup(env,model,contract);torch.set_grad_enabled(False)
 cfg=json.loads(Path(__file__).with_name('recovery_protocol.json').read_text());c=cfg['models']['C0'] if 'C0' in cfg['models'] else dict(path=str(R.parent/'prior-recovery-20261105/correct-C0/C0.pt'),sha256='ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c')
 assert runner.sha(c['path'])==c['sha256'];C=load(torch.load(c['path'],map_location='cuda',weights_only=True),'cuda');rows=[];torch.manual_seed(4600)
 scale=torch.tensor(contract['q_PD_scale'],device='cuda');off=torch.tensor(contract['q_PD_offset'],device='cuda')
 for name in ['new9','new65','stop','continue']:
  e=json.load(gzip.open(R/'final-reference'/('I__'+name+'.json.gz'),'rt'))
  for t in [4.9,5.5,6.5,8.,10.]:
   j=min(range(len(e['decisions'])),key=lambda j:abs(e['decisions'][j]['time']-.1-t));d=e['decisions'][j];cs=e['control'][5*j:5*j+5]
   if len(cs)!=5:continue
   x=torch.tensor([d['obs']],device='cuda');p=x[:,565:821]*16;im=I.distribution(x).mean;cm=C.distribution(x).mean;iz=direct_command(im,env.standing);cz=direct_command(cm,env.standing);eps=torch.randn((256,256),device='cuda');rr=im+.008*eps;zz=direct_command(rr,env.standing)
   def acts(z):
    ob={k:torch.tensor([c['bfm_obs'][k][0] for c in cs],device='cuda') for k in contract['bfm_inputs']};ob={k:v.repeat(len(z),1) for k,v in ob.items()};a=model.act(ob,interp(p.expand(len(z),-1),z).reshape(-1,256),mean=True);q=off+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5;return q.reshape(len(z),-1)
   qi=acts(iz);qc=acts(cz);qn=acts(zz);delta=qc-qi;noise=qn-qi;cos=(noise*delta).sum(-1)/(noise.norm(dim=-1)*delta.norm()).clamp_min(1e-10);frac=(noise*delta).sum(-1)/delta.square().sum().clamp_min(1e-10)
   nearest=(im*cz).sum(-1,keepdim=True).clamp_min(1e-6)*cz/256
   rows.append(dict(condition=name,time=cs[0]['time'],I_C0_z_angle=float(torch.rad2deg(torch.acos((iz*cz).sum()/256)).nan_to_num()),I_C0_qpd_rms_deg=float(torch.rad2deg(delta).square().mean().sqrt()),minimum_raw_direction_distance_std=float((nearest-im).norm()/.008),noise_z_angle_mean_deg=float(torch.rad2deg(torch.acos((zz*iz).sum(-1).div(256).clamp(-1,1))).mean()),noise_qpd_rms_deg=float(torch.rad2deg(noise).square().mean().sqrt()),noise_alignment_mean=float(cos.mean()),noise_progress_fraction_max=float(frac.max()),fraction_at_least_half_action_direction=float((frac>=.5).float().mean()),scope='C0 evaluated on actual I state/history; not proof C0 recovery on this state'))
 runner.write(out/'exploration.json',dict(std=.008,samples_per_state=256,states=rows,scope='finite Monte Carlo command/action coverage, not probability of multistep skill discovery'))
 summaries=[]
 for name in ['new9','new65']:
  spec=conditions()[name];task.reset(spec,record=True,label='I_sample__'+name);g=torch.Generator(device='cuda').manual_seed(4600)
  while not task.done:
   dist=I.distribution(task.features());raw=dist.mean+dist.stddev*torch.randn(dist.mean.shape,device='cuda',generator=g);task.step(raw)
  summaries.append(task.save(out));runner.write(out/'summaries.json',summaries)
 return dict(status='completed',updates=0,rl_training_decisions=0,episodes=len(summaries),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
