"""Replay both high functions on each actually visited BFM substep observation.
The BFM observations are fixed to the real owner's path, not a physical rollout.
"""
import csv,gzip,json,time
from pathlib import Path
import numpy as np
import torch
from dual_track_high import build,write
from clean_policy import direct_command
from locomotion_rollout import spherical

def run(e,m,contract,out,args):
 start=time.monotonic();D=out.parents[1];folder=D/'runs/d1-01'
 old=D.parent/'dual-track-formal-20261109/runs/expert-kl002';nets=[]
 for i in (0,48):
  n=build();n.load_state_dict(torch.load(old/f'candidate-{i}.pt',map_location='cpu',weights_only=False)['model']);n.eval().requires_grad_(False);nets.append(n)
 rows=[];archives=[];max_self=0.
 for p in sorted(folder.glob('*-complete.pt')):
  label=p.name.removesuffix('-complete.pt');d=torch.load(p,map_location='cpu',weights_only=False)
  ep=json.load(gzip.open(folder/(label+'.json.gz'),'rt'));ci=0
  for h in d['high']:
   x=h['input'].cuda();previous=h['own_last_z'].cuda();targets=[direct_command(n.distribution(x).mean,e.standing) for n in nets]
   part=[]
   for j in range(5):
    if ci>=len(ep['control']) or abs(ep['control'][ci]['time']-(h['time']+.02*j))>1e-7:break
    c=ep['control'][ci];ci+=1;obs={k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in c['bfm_obs'].items()}
    aa=[m.act(obs,spherical(previous,z,(j+1)/5),mean=True) for z in targets]
    qq=[torch.tensor(contract['q_PD_offset'],device='cuda')+a.clamp(-contract['clip']/contract['normalize_ratio'],contract['clip']/contract['normalize_ratio'])*torch.tensor(contract['q_PD_scale'],device='cuda') for a in aa]
    owner=0 if label.endswith('-i0') else 1
    self_error=float((aa[owner][0]-torch.tensor(c['action'],device='cuda')).abs().max());max_self=max(max_self,self_error)
    assert self_error<1e-6,('Actual owner action replay mismatch',label,c['time'],self_error)
    delta=torch.rad2deg(qq[1]-qq[0])[0]
    row=dict(label=label,high_time=h['time'],native_time=c['time'],substep=j+1,own_action_max_error=self_error,
      action_delta_rms=float((aa[1]-aa[0]).square().mean().sqrt()),qpd_delta_max_deg=float(delta.abs().max()),qpd_delta_rms_deg=float(delta.square().mean().sqrt()))
    rows.append(row);part.append(dict(**row,qpd_delta_each_joint_deg=delta.cpu(),actions=[a.cpu() for a in aa],q_PD=[q.cpu() for q in qq]))
   archives.append(dict(label=label,high_time=h['time'],substeps=part))
  assert ci==len(ep['control'])
 with (out/'actual-five-substep-replay.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 torch.save(archives,out/'actual-five-substep-replay.pt')
 return dict(status='completed',actual_substep_rows=len(rows),self_action_max_error=max_self,
  physics_samples=0,optimizer_updates=0,wall_s=time.monotonic()-start,
  scope='Both high commands on owner actual successive BFM observations; owner prev-z and original interpolation. Not counterfactual physics.')
