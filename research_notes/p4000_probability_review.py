"""Saved CUDA probability review. No new physics, sampling, optimizer or threshold change."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
import torch
from clean_policy import Expert

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 torch.set_num_threads(1);torch.set_grad_enabled(False);s=Path(__file__).resolve().parents[2];d=s/'runs/seed5202-32';checkpoint=d/'latest.pt';ck=torch.load(checkpoint,map_location='cuda',weights_only=True);assert ck['iteration']==30
 net=Expert(1088,'direct').cuda().eval();net.load_state_dict(ck['state_dict']);rows=[];oldmeans=[];data=[]
 for worker in range(3):
  with np.load(d/f'worker-{worker}/rollout-0030.npz') as f: data.append({k:f[k].copy() for k in f.files})
 obs=torch.as_tensor(np.stack([x['obs'] for x in data],axis=1).reshape(768,-1),device='cuda');raw=torch.as_tensor(np.stack([x['action'] for x in data],axis=1).reshape(768,-1),device='cuda');old=torch.as_tensor(np.stack([x['logp'] for x in data],axis=1).reshape(-1),device='cuda')
 means={};logs={}
 for chunk in [1,16,128,256,768]:
  dist=[net.distribution(obs[i:i+chunk]) for i in range(0,len(obs),chunk)];means[chunk]=torch.cat([z.mean for z in dist]);logs[chunk]=torch.cat([z.log_prob(raw[i:i+chunk]).sum(-1) for i,z in zip(range(0,len(obs),chunk),dist)])
  rows.append(dict(chunk=chunk,old_logprob_max_error=float((logs[chunk]-old).abs().max()),old_logprob_mismatch_count=int(((logs[chunk]-old).abs()>=3e-4).sum()),mean_max_difference_from_row=None))
 for row in rows:row['mean_max_difference_from_row']=float((means[row['chunk']]-means[1]).abs().max())
 std=net.log_std.exp();standardized=[]
 for w,x in enumerate(data):
  gen=torch.Generator(device='cuda').manual_seed(5202*1000003+w*1001+30*100003);noise=torch.cat([torch.randn((1,256),device='cuda',generator=gen) for _ in range(256)])
  actual=torch.as_tensor(x['action'],device='cuda');reconstructed=means[1][w::3]+std*noise
  standardized.append(dict(worker=w,raw_reconstruction_max_error=float((actual-reconstructed).abs().max()),raw_exact=bool(torch.equal(actual,reconstructed))))
 double_lp=lambda mu:torch.distributions.Normal(mu.double(),std.double()).log_prob(raw.double()).sum(-1)
 comparisons={str(chunk):dict(high_precision_logprob_max_difference_from_row=float((double_lp(means[chunk])-double_lp(means[1])).abs().max())) for chunk in means}
 bad=(logs[768]-old).abs().argmax();i=int(bad)
 result=dict(status='completed',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint_iteration=ck['iteration'],runtime=ck['runtime'],saved_failed_batch=30,probability_threshold_unchanged=3e-4,torch_version=torch.__version__,allow_tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),rows=rows,raw_action_reconstruction=standardized,high_precision_comparison=comparisons,worst=dict(index=i,worker=i%3,worker_timestep=i//3,old_logprob=float(old[i]),row_logprob=float(logs[1][i]),batched_logprob=float(logs[768][i]),mean_difference_max=float((means[768][i]-means[1][i]).abs().max())),new_RL_decisions=0,new_physics_steps=0,optimizer_updates=0)
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
