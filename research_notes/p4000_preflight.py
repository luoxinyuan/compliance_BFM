"""Bounded P4000 parity, actual exploration and complete-state short controls; no training."""
import argparse,json,os,sys,hashlib,gzip,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
PLAN=json.loads(Path(__file__).with_name('p4000_protocol.json').read_text());S=ROOT.parent

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def study(e,m,c,out):
 import torch,numpy as np
 from p4000_env import Task,install
 from clean_policy import Expert,direct_command
 from native_transfer import phase_report
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from rolling_snapshot import BranchState
 torch.set_num_threads(1);torch.set_grad_enabled(False);install();write(out/'runtime-speed.json',speed(e));write(out/'backend.json',backend(e,m,S.parent/'native-transfer-20261107/compatibility-prefix'))
 ckpath=Path(PLAN['initial']['path']);assert hashlib.sha256(ckpath.read_bytes()).hexdigest()==PLAN['initial']['sha256'];ck=torch.load(ckpath,map_location='cuda',weights_only=True);net=Expert(1088,'direct').cuda();net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False);task=Task(e,m,c,5200);summaries=[];inputs=[];physics=0
 def run(label,s,noise=0.):
  nonlocal physics
  x=task.reset(s,record=True,label=label);g=torch.Generator(device=x.device).manual_seed(5200);angles=[];norms=[]
  while not task.done:
   inputs.append(x.clone());mu=net.distribution(x).mean;raw=mu+noise*torch.randn(mu.shape,device=x.device,generator=g) if noise else mu;z=direct_command(raw,e.standing);zm=direct_command(mu,e.standing);angles.append(float(torch.rad2deg(torch.acos(((z*zm).sum()/256).clamp(-1,1)))));norms.append(float(mu.norm()));x,_,_,st=task.step(raw);physics+=st['physics_steps']
  row=task.save(out);row.update(phase_report(task),noise_std=noise,z_noise_angle_mean=float(np.mean(angles)),z_noise_angle_max=max(angles),raw_mean_norm_min=min(norms),raw_mean_norm_mean=float(np.mean(norms)));summaries.append(row);write(out/'summaries.json',summaries);print('P4000_PREFLIGHT',label,row['all']['error_cm'],row['all']['force_rms'],row['boundary'],flush=True);return row
 means={}
 for k in ['short08','fast72','normal72']:
  means[k]=run(k+'__mean',PLAN['dev'][k])
  if k in PLAN['preflight']['mean_parity']:
   ref=PLAN['preflight']['mean_parity'][k];assert abs(means[k]['all']['error_cm']-ref[0])<1e-5 and abs(means[k]['all']['force_rms']-ref[1])<1e-5,('mean parity mismatch',k,means[k]['all'],ref)
 def bad(r,k):return bool(r['boundary']) or r['all']['error_cm']>means[k]['all']['error_cm']+max(10.,.5*means[k]['all']['error_cm'])
 base={k:run(k+'__sample008',PLAN['dev'][k],.008) for k in PLAN['preflight']['sample_cases']};chosen=.008;small=None
 if any(bad(v,k) for k,v in base.items()):
  small={k:run(k+'__sample004',PLAN['dev'][k],.004) for k in base}
  cost=lambda rr:sum(v['all']['force_rms']+.03*v['all']['error_cm']+v['all']['cap_fraction']+20*bool(v['boundary']) for v in rr.values())
  if not any(v['boundary'] for v in small.values()) and cost(small)<cost(base):chosen=.004
  elif any(v['boundary'] for v in base.values()):raise RuntimeError('Both registered noise choices lack safe improvement; no PPO launch')
 # New actor/normalizer batch shape only: original numerical threshold unchanged.
 x=torch.cat(inputs[::max(1,len(inputs)//96)])[:128];d=net.distribution(x);g=torch.Generator(device=x.device).manual_seed(5200);raw=d.mean+chosen*torch.randn(d.mean.shape,device=x.device,generator=g);net.log_std.fill_(np.log(chosen));batch=net.distribution(x).log_prob(raw).sum(-1);one=torch.cat([net.distribution(t[None]).log_prob(a[None]).sum(-1) for t,a in zip(x,raw)]);err=float((batch-one).abs().max());assert err<3e-4,('original logprob assertion',err)
 write(out/'probability.json',dict(batch_one_FP32_logprob_max_error=err,threshold=3e-4,raw_Gaussian=True,PPU_path_used=False));write(out/'sampling-decision.json',dict(selected_std=chosen,default=.008,small_common_correction_tested=small is not None,rule=PLAN['preflight']['noise_rule'],no_final_data=True))
 # Exact real Teacher state prefixes, then alternative first command and feedback continuation.
 old=S.parent/'native-rolling-h100-20261107/runs/rolling-02';ep=json.loads(gzip.open(old/'rolling_continue.json.gz','rt').read());qs=[q for q in json.loads((old/'searches.json').read_text()) if q['label']=='rolling_continue'];branches=[]
 for at in PLAN['preflight']['branches']['times']:
  task.reset(ep['condition'],record=True,label='branch_prefix');ix=0
  while float(e.tether.time)<at-1e-7:
   q=qs[ix];assert abs(q['time']-float(e.tether.time))<1e-6;task.step(torch.tensor(q['selected_first_z'],device='cuda')[None]/16);ix+=1
  ss=BranchState(task);g=torch.Generator(device='cuda').manual_seed(5200+round(at*10));x=task.features();mu=net.distribution(x).mean
  firsts={'mean':mu,'sample':mu+chosen*torch.randn(mu.shape,device='cuda',generator=g),'search_first':torch.tensor(qs[ix]['selected_first_z'],device='cuda')[None]/16}
  for name,first in firsts.items():
   ss.restore();assert BranchState(task).fingerprint==ss.fingerprint;start=len(task.physics);total=0.;boundary=None
   for j in range(12):
    raw=first if j==0 else net.distribution(task.features()).mean;_,r,done,st=task.step(raw);physics+=st['physics_steps'];total+=r
    if done:boundary=task.boundary;break
   rr=task.physics[start:];active=[r for r in rr if r['active']];f=np.array([r['force'] for r in active]);a=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);branches.append(dict(time=at,first=name,reward=total,force_rms=float(np.sqrt((f*f).sum(-1).mean())),error_cm=float(100*np.sqrt((a*a).sum(-1).mean())),boundary=boundary,source_complete_state=ss.fingerprint,steps=len(rr),scope='one first-command intervention, then P4000 feedback for1.2s; search first-z is not PPO sample'))
  ss.restore()
 write(out/'same-state-branches.json',branches);return dict(status='completed',selected_std=chosen,mean_reproduced=True,original_probability_passed=True,diagnostic_physics_steps=physics,PPO_decisions=0,student_optimizer_updates=0)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);os.environ['BFM_RECOVERY_OUTPUT']=str(a.output);os.chdir(a.output)
 import m10_physics,m10_stage
 m10_physics.OUT=a.output;m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
