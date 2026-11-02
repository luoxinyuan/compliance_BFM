import json,gzip,sys,hashlib
from pathlib import Path
import numpy as np,torch
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/clean-contact-expert-20261102'
sys.path.insert(0,str(P/'bridge-v2-owned/trees/clean-direct-fixed/repo/research_notes'))
from clean_policy import Expert
out=R/'evaluation-analysis';out.mkdir(exist_ok=True)
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def metrics(e):
 rows=e['physics'];c=e['control'];result={}
 for name,(lo,hi) in e['phases'].items():
  rs=[r for r in rows if lo<=r['time']<hi]
  if not rs:continue
  F=np.array([r['force'] for r in rs]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in rs]);root=np.array([r['root'] for r in rs]);w=np.array([r['wrist'] for r in rs]);v=np.array([r['qvel'][:3] for r in rs]);active=np.array([r['active'] for r in rs]);support=np.array([r['support'] for r in rs]);foot=np.array([r['foot_normal_force_n'] for r in rs]);cs=[r for r in c if lo<=r['time']<hi]
  m=dict(force_all_rms=float(np.sqrt(np.mean(np.sum(F*F,1)))),force_xyz_rms=np.sqrt(np.mean(F*F,0)).tolist(),force_xyz_mean=F.mean(0).tolist(),active_samples=int(active.sum()),samples=len(rs),root_delta=(root[-1]-root[0]).tolist(),wrist_delta=(w[-1]-w[0]).tolist(),wrist_relative_root_delta=((w[-1]-root[-1])-(w[0]-root[0])).tolist(),root_speed_peak=float(np.linalg.norm(v,axis=1).max()),root_height_min=float(root[:,2].min()),tilt_peak=max(r['tilt_deg'] for r in rs),joint_limit_max=max(r['joint_limit_excess_max_rad'] for r in rs),torque_saturation_max=float(np.array([r['torque_saturation'] for r in rs]).max()),foot_normal_min=foot.min(0).tolist(),no_support_fraction=float(np.mean(np.max(support,axis=1)<=0)) if support.ndim==2 else float(np.mean(support<=0)),slip_max=max(r['contact_tangent_speed_max_mps'] for r in rs))
  if active.any():m.update(error_cm=float(100*np.sqrt(np.mean(np.sum(E[active]**2,1)))),signed_error_xyz_cm=(100*E[active].mean(0)).tolist(),error_xyz_rms_cm=(100*np.sqrt(np.mean(E[active]**2,0))).tolist(),active_force_rms=float(np.sqrt(np.mean(np.sum(F[active]**2,1)))),cap_fraction=float(np.mean(np.linalg.norm(F[active],axis=1)>=4.999)))
  if cs:m['qpd_step_peak_deg']=float(np.rad2deg(max(x['q_PD_step_max_rad'] for x in cs)))
  if name=='recover':
   m['speed_half_second_rms']=[float(np.sqrt(np.mean(np.sum(v[(np.array([r['time'] for r in rs])>=lo+t)&(np.array([r['time'] for r in rs])<lo+t+.5)]**2,1)))) for t in np.arange(0,hi-lo,.5)]
   m['end_root_speed_rms']=float(np.sqrt(np.mean(np.sum(v[-200:]**2,1))))
  result[name]=m
 return dict(boundary=e['boundary'],condition=e['condition'],phases=result)
def main():
 paths={'direct_initial':R/'direct-train-08/job/dev-0000/single__mean.json.gz','direct15':R/'direct-train-08/job/dev-0015/single__mean.json.gz','direct20':R/'direct-train-08/job/dev-0020/single__mean.json.gz','structured20':R/'structured-train-06/job/dev-0020/single__mean.json.gz'}
 result={'physical':{k:metrics(read(p)) for k,p in paths.items()}}
 torch.set_num_threads(1);nets={}
 for it in [15,20]:
  ck=torch.load(R/f'direct-train-08/job/candidate-{it:04d}.pt',map_location='cpu',weights_only=True);n=Expert(ck['input_dim'],'direct');n.load_state_dict(ck['state_dict']);n.eval();nets[it]=n
 result['same_observation_policy_comparison']={}
 with torch.no_grad():
  for name in ['direct15','direct20']:
   e=read(paths[name]);x=torch.tensor([d['obs'] for d in e['decisions']]);a=nets[15].distribution(x);b=nets[20].distribution(x);za=torch.nn.functional.normalize(a.mean,dim=-1);zb=torch.nn.functional.normalize(b.mean,dim=-1);angle=torch.rad2deg(torch.acos((za*zb).sum(-1).clamp(-1,1)));kl=torch.distributions.kl_divergence(a,b).sum(-1)
   result['same_observation_policy_comparison'][name]=dict(mean_command_angle_deg=float(angle.mean()),max_command_angle_deg=float(angle.max()),joint_KL_mean=float(kl.mean()),joint_KL_max=float(kl.max()),raw_delta_rms=float((a.mean-b.mean).square().mean().sqrt()),scope='high-level actor only on CPU, no frozen BFM CPU execution; accumulated5updates, not per-batch trust statistic')
 a=read(paths['direct15'])['physics'];b=read(paths['direct20'])['physics'];assert len(a)==len(b)
 d=np.linalg.norm(np.array([r['wrist'] for r in a])-np.array([r['wrist'] for r in b]),axis=1)
 idx=next((i for i in range(len(d)-20) if (d[i:i+20]>=.01).all()),None)
 result['first_1cm_wrist_divergence_sustained100ms']=None if idx is None else dict(time=a[idx]['time'],wrist_gap=float(d[idx]),force15=a[idx]['force'],force20=b[idx]['force'],scope='descriptive trajectory divergence, not same-state physical causal intervention')
 result['training']={}
 for name in ['structured-train-06','direct-train-08']:
  logs=[json.loads(l) for l in (R/name/'job/training.jsonl').read_text().splitlines()]
  result['training'][name]=dict(decisions=logs[-1]['rl_decisions'],physical_steps=sum(x['physical_steps'] for x in logs),control_steps=sum(x['control_steps'] for x in logs),optimizer_updates=logs[-1]['optimizer_updates'],max_accepted_joint_KL=max(x['optimization']['epochs'][-1]['joint_gaussian_KL'] for x in logs),max_logprob_reproduction_error=max(x['optimization']['pre_logprob_max_error'] for x in logs))
 (out/'development-diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
