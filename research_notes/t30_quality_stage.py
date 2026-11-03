"""Saved-state decomposition and one predeclared 0.1s high-level command filter; frozen BFM/PD."""
from pathlib import Path
import json,gzip,time
import transfer_student_validated as runner
from stop_env import install
PLAN=json.loads(Path(__file__).with_name('t30_protocol.json').read_text());P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');OLD=P/'bridge-v2-owned/runs/stop-sensing-20261103'
def study(env,model,contract,out):
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 from locomotion_rollout import spherical
 from m11_runtime_speed import install as speed
 install();from clean_env import Task
 runner.write(out/'runtime-speed.json',speed(env));torch.set_num_threads(1);torch.set_grad_enabled(False);task=Task(env,model,contract,805);task.reset();device=env.standing.device
 identity=PLAN['teachers']['t30'];assert runner.sha(identity['path'])==identity['sha256'];ck=torch.load(identity['path'],map_location=device,weights_only=True);net=Expert(ck['input_dim'],'direct').to(device);net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False)
 checks=[]
 for name in ['stop_unseen_early','stop_unseen_late','normal_new','local']:
  with gzip.open(OLD/'E-final-08/job'/('selected_'+name+'.json.gz'),'rt') as f:ep=json.load(f)
  cs=ep['control'];q=np.array([r['q_PD'] for r in cs]);delta=np.abs(np.diff(q,axis=0));i,j=np.unravel_index(np.argmax(delta),delta.shape);i+=1
  prev,now=cs[i-1],cs[i];op={k:torch.tensor(v,device=device) for k,v in prev['bfm_obs'].items()};on={k:torch.tensor(v,device=device) for k,v in now['bfm_obs'].items()};zp=torch.tensor(prev['executed_z'],device=device);zn=torch.tensor(now['executed_z'],device=device)
  qp=task.qof(model.act(op,zp,mean=True));qn=task.qof(model.act(on,zn,mean=True));qz=task.qof(model.act(op,zn,mean=True));qo=task.qof(model.act(on,zp,mean=True));err=max(float((qp-torch.tensor(prev['q_PD'],device=device)).abs().max()),float((qn-torch.tensor(now['q_PD'],device=device)).abs().max()));assert err<1e-4,err
  checks.append(dict(condition=name,time=now['time'],joint=contract['dof_names'][j],reconstruction_max_rad=err,actual_jump_deg=float(torch.rad2deg(qn-qp)[0,j]),z_only_at_previous_obs_deg=float(torch.rad2deg(qz-qp)[0,j]),obs_only_at_previous_z_deg=float(torch.rad2deg(qo-qp)[0,j]),remaining_obs_after_z_deg=float(torch.rad2deg(qn-qz)[0,j]),remaining_z_after_obs_deg=float(torch.rad2deg(qn-qo)[0,j]),scope='frozen computational same-observation decomposition, not independent physical causality'))
 runner.write(out/'jump-decomposition.json',checks);rows=[]
 for name,spec in PLAN['dev'].items():
  task.reset(spec,record=True,label='filter01_'+name)
  while not task.done:
   raw=net.distribution(task.features()).mean;z=direct_command(raw,env.standing);filtered=spherical(task.last_command,z,1-np.exp(-.1/.1));task.step(filtered/16)
  row=task.save(out);rows.append(row);runner.write(out/'summaries.json',rows);print('QUALITY_FILTER',name,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
  if row['boundary'] is not None:break
 return dict(status='completed',decisions=0,student_optimizer_updates=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
