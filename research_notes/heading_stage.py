"""Bounded same-weight turning qualification and Direct initialization; no online rule in learned execution."""
from pathlib import Path
import copy,json,os,time,gzip
import transfer_student_validated as runner
from heading_env import install
PLAN=json.loads(Path(__file__).with_name('heading_protocol.json').read_text())
def study(env,model,contract,out):
 import numpy as np,torch
 from clean_policy import Expert,direct_command
 from m11_runtime_speed import install as speed
 install();from clean_env import Task
 runner.write(out/'runtime-speed.json',speed(env));torch.set_num_threads(1);torch.set_grad_enabled(False);task=Task(env,model,contract,906);device=env.standing.device
 identity=PLAN['teachers']['t30'];assert runner.sha(identity['path'])==identity['sha256'];ck=torch.load(identity['path'],map_location=device,weights_only=True);net=Expert(ck['input_dim'],'direct').to(device);net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False);summary=[];datasets=[]
 def run(label,spec,mode,policy):
  task.reset(spec,record=True,label=label,query_reference=mode=='m10_expert')
  while not task.done:
   x=task.features();raw=policy.distribution(x).mean;task.step(raw,mode=mode)
  row=task.save(out);summary.append(row);runner.write(out/'summaries.json',summary)
  ph=task.physics;moving=[r for r in ph if r['active'] and np.linalg.norm(r['anchor_velocity'][:2])>.02]
  angles=[np.arctan2(np.array(r['pelvis_matrix'])[1,0],np.array(r['pelvis_matrix'])[0,0]) for r in moving];target=[np.arctan2(r['anchor_velocity'][1],r['anchor_velocity'][0]) for r in moving];err=np.arctan2(np.sin(np.array(angles)-target),np.cos(np.array(angles)-target))
  row['moving_heading_rms_deg']=float(np.degrees(np.sqrt(np.mean(err**2)))) if moving else None;row['heading_end_move_deg']=float(np.degrees(angles[-1])) if moving else None;row['desired_heading_end_move_deg']=float(np.degrees(target[-1])) if moving else None
  runner.write(out/'summaries.json',summary);print('HEADING_RUN',label,row['error_cm'],row['force_rms'],row['moving_heading_rms_deg'],row['boundary'],flush=True)
  return row,copy.deepcopy(task.bootstrap_rows)
 for name,spec in PLAN['heading_sources'].items():
  row,labels=run('M10_'+name,spec,'m10_expert',net)
  if row['boundary'] is None:datasets.extend(dict(r,episode=name) for r in labels)
  if name in PLAN['heading_dev']:run('T30_'+name,spec,'learned',net);run('standing_'+name,spec,'standing',net)
 runner.write(out/'qualification.json',dict(rows=summary,labels=len(datasets),source='M10 task-adjusted motion + existing projected wrist feedback, not learned general heading expert',rule='No automatic rejection from single metric; unsafe source labels excluded. All source and learned physical outcomes retained.'))
 if not datasets:raise RuntimeError('No safe source trajectories; do not train on invalid teacher')
 with gzip.open(out/'heading-source.json.gz','wt') as f:json.dump(datasets,f)
 # Retain frozen Teacher30 normalization and initialize a new heading actor from it.
 x=torch.tensor([r['obs'] for r in datasets],device=device);z=torch.tensor([r['teacher']['z'] for r in datasets],device=device)
 torch.manual_seed(906);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=1e-4);curve=[]
 groups=[torch.tensor([i for i,r in enumerate(datasets) if r['episode']==name],device=device) for name in PLAN['heading_sources']];groups=[g for g in groups if len(g)]
 with torch.enable_grad():
  for step in range(4000):
   g=groups[step%len(groups)];ix=g[torch.randint(len(g),(128,),device=device)];pred=direct_command(net.distribution(x[ix]).mean,env.standing);loss=(pred-z[ix]).square().mean();opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
   if step%200==0:curve.append(dict(step=step,loss=float(loss.detach()),grad=gn))
 net.eval().requires_grad_(False);f=out/'heading-initial.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},input_dim=len(net.center),kind='direct',source_student_online=False,iteration=0,decisions=0,seed=906),f)
 runner.write(out/'heading-training.json',dict(updates=4000,role='Expert initialization, not Student updates or RL',samples=len(datasets),source_sha=runner.sha(out/'heading-source.json.gz'),sha256=runner.sha(f),curve=curve,alignment='Decision input preceding original 5x50Hz source interval -> endpoint legal z; source used privileged current feedback through interval, so endpoint approximation explicitly retained; physical learned test determines viability.'))
 for name,spec in {**PLAN['heading_dev'],**{k:v for k,v in PLAN['heading_sources'].items() if k.startswith('change')},**PLAN['regression']}.items():run('DirectInit_'+name,spec,'learned',net)
 result=dict(status='completed',decisions=0,student_optimizer_updates=0,expert_supervised_updates=4000,elapsed_s=time.monotonic()-runner.START);runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
