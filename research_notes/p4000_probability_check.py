"""Check canonical FP32 actor probabilities, gradients and separate critic on saved data."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',P4000_CANONICAL_ROW_ACTOR='1')
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from clean_policy import Expert,update as original_update
from p4000_critic import update,actor_adam,equal_tree
from humanoidverse.force_compliance.parallel import merge_rollouts

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 torch.set_num_threads(1);torch.set_grad_enabled(False);s=Path(__file__).resolve().parents[2];out=[];start=time.monotonic()
 for seed,version,cp in [(5201,0,'initial.pt'),(5202,30,'latest.pt')]:
  source=s/f'runs/seed{seed}-32';checkpoint=torch.load(source/cp,map_location='cuda',weights_only=True);results=[]
  for w in range(3):
   with np.load(source/f'worker-{w}/rollout-{version:04d}.npz') as z:results.append(dict(batch={k:z[k].copy() for k in z.files}))
  batch={k:v.cuda() for k,v in merge_rollouts(results,.95).items()};batch['raw_adv_std']=(batch['returns']-batch['value']).std();nets=[];opts=[]
  for _ in range(2):
   net=Expert(1088,'direct').cuda();net.load_state_dict(checkpoint['state_dict']);opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=3e-5),dict(params=net.critic.parameters(),lr=3e-4)]);nets.append(net);opts.append(opt)
  # The failed model's saved optimizer is irrelevant to the numerical replay;
  # same fresh optimizer on both implementations isolates their actor algebra.
  d=nets[0].distribution(batch['obs']);err=float((d.log_prob(batch['action']).sum(-1)-batch['logp']).abs().max());assert err<3e-4
  shapes={}
  for chunk in [1,16,128,256,768]:
   mu=torch.cat([nets[0].distribution(v).mean for v in batch['obs'].split(chunk)]);assert torch.equal(mu,d.mean);shapes[str(chunk)]=True
  # A mismatched action must still violate the original guard.
  wrong=dict(batch);wrong['action']=batch['action'].clone();wrong['action'][0,0]+=.008
  rejected=False
  try:original_update(nets[0],opts[0],wrong,epochs=0)
  except AssertionError:rejected=True
  assert rejected
  torch.cuda.synchronize();before=time.monotonic();torch.manual_seed(9701);old=original_update(nets[0],opts[0],batch);torch.cuda.synchronize();old_s=time.monotonic()-before
  torch.manual_seed(9701);before=time.monotonic();new=update(nets[1],opts[1],batch,critic_shuffle_seed=52010000);torch.cuda.synchronize();new_s=time.monotonic()-before
  ae=all(torch.equal(v,nets[1].actor.state_dict()[k]) for k,v in nets[0].actor.state_dict().items());oe=equal_tree(actor_adam(opts[0],nets[0]),actor_adam(opts[1],nets[1]));assert ae and oe
  assert old['optimizer_updates']==new['optimizer_updates'] and new['critic_fit']['optimizer_updates']==18
  assert new['critic_fit']['post']['mse']<new['critic_fit']['pre']['mse']
  out.append(dict(seed=seed,version=version,checkpoint_sha256=hashlib.sha256((source/cp).read_bytes()).hexdigest(),old_logprob_max_error=err,all_batch_shapes_bitwise_equal=shapes,wrong_action_rejected=rejected,actor_equal=ae,actor_adam_equal=oe,old_update_s=old_s,new_update_s=new_s,old=old,new=new))
 result=dict(status='completed',actor_equal=True,rows=out,canonical_actor='FP32 batch-one actor for both sampling and scoring; actual batch-one controls unchanged',unchanged_thresholds=dict(logprob=3e-4,joint_KL_accept=.02,joint_KL_stop=.015),new_physics_steps=0,RL_decisions=0,elapsed_s=time.monotonic()-start)
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
