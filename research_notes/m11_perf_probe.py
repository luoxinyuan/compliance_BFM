"""Bounded GPU-only profiling of existing sampling, no PPO updates or new task."""
import m11_ipc
import os,time,json,cProfile,pstats,io
from pathlib import Path
import m10_stage,m10_physics

def study(e,m,c,out):
 import torch
 from m11_env import Task
 from m11_policy import Expert
 task=Task(e,m,c,1001)
 spec=dict(distance=.82,ramp=7.8,onset=4.25,release=14.05,end=20.05,heading_deg=0,curvature=0,enabled=True,family='walking')
 obs=task.reset(spec,record=True,label='profile-baseline')
 ck=torch.load('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-privileged-expert-20261101/train-attempt-01/outputs/job/best.pt',map_location='cuda',weights_only=True)
 net=Expert(ck['input_dim']).cuda();net.load_state_dict(ck['state_dict']);gen=torch.Generator(device='cuda').manual_seed(101)
 pr=cProfile.Profile();torch.cuda.synchronize();start=time.monotonic();pr.enable()
 for i in range(64):
  d=net.distribution(obs);a=d.mean+d.stddev*torch.randn(d.mean.shape,device='cuda',generator=gen)
  obs,_,_,_=task.step(a)
  if task.done:break
 torch.cuda.synchronize();pr.disable();elapsed=time.monotonic()-start
 pr.dump_stats(out/'profile.pstats');s=io.StringIO();pstats.Stats(pr,stream=s).strip_dirs().sort_stats('cumtime').print_stats(45);(out/'profile.txt').write_text(s.getvalue());print(s.getvalue())
 task.save(out)
 return dict(status='completed',decisions=i+1,seconds=elapsed,decisions_per_second=(i+1)/elapsed,profile='baseline',policy_updated=False)
m10_stage.run_study=study
if __name__=='__main__':m10_physics.main()
