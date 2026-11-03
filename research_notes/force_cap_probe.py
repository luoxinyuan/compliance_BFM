"""Authorized frozen-policy force-cap diagnostic. No training or reward comparison."""
from pathlib import Path
import json,time,copy,gzip
import transfer_student_validated as runner
from stop_env import PLAN,install
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/stop-sensing-20261103'
def study(env,model,contract,out):
 import torch,numpy as np
 from clean_policy import Expert
 from m11_runtime_speed import install as speed
 install();from clean_env import Task
 runner.write(out/'runtime-speed.json',speed(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 task=Task(env,model,contract,707);task.reset();device=env.standing.device
 identities={};nets={}
 for name,folder in [('B505','A-prepare-03'),('N505','C-final-05'),('N506','C-final-05')]:
  root=RUN/folder/'job';identity=json.loads((root/(name+'-training.json')).read_text());path=root/(name+'.pt');assert runner.sha(path)==identity['sha256'];ck=torch.load(path,map_location=device,weights_only=True)
  net=Expert(ck['input_dim'],'direct').to(device);net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False);nets[name]=net;identities[name]=dict(path=str(path),sha256=identity['sha256'])
 spec=copy.deepcopy(PLAN['final']['stop_unseen_late']);rows=[]
 runner.write(out/'protocol.json',dict(caps=[5.,7.5,10.],models=['standing','B505','N505','N506'],identities=identities,condition=spec,scope='previously exposed condition; diagnostic not new blind test; frozen models; no proprio noise; no cap-dependent input rescaling',reward='legacy reward and cap fields invalid for cross-cap ranking; use recomputed physical metrics',stop='existing physical review boundary stops the entire probe; no gains or cap retuning'))
 for cap in [5.,7.5,10.]:
  for label in ['standing','B505','N505','N506']:
   for c in [env.cfg,env.tether.cfg]:c.max_force=cap
   task.sensor_mode='ideal';task.reset(spec,record=True,label=f'{label}_cap{cap:g}')
   while not task.done:
    task.prepare();R=env.sim.data.xmat[1].reshape(3,3);force=task.sensor.features(R)[-100:,:3].reshape(-1)
    x=torch.cat([task.pro.clamp(-10,10),torch.tensor(force,device=device,dtype=torch.float32)[None],task.last_command/16,torch.tensor([[1.,0,0,0]],device=device)],-1)
    if label.startswith('N'):x=torch.cat([x[:,:465],x[:,765:]],-1)
    raw=torch.zeros((1,256),device=device) if label=='standing' else nets[label].distribution(x).mean
    task.step(raw,mode='standing' if label=='standing' else 'learned')
   row=task.save(out);row['max_force_N']=cap;row['legacy_reward_not_comparable']=True
   active=[r for r in task.physics if r['active']];row['cap_fraction']=float(np.mean([np.linalg.norm(r['force'])>=cap-1e-3 for r in active]));row['peak_force_N']=max(float(np.linalg.norm(r['force'])) for r in task.physics)
   for name,(a,b) in json.loads(gzip.decompress((out/(task.label+'.json.gz')).read_bytes()))['phases'].items():
    segment=[r for r in task.physics if a<=r['time']<b]
    if segment:row['phases'][name]['cap_fraction']=float(np.mean([np.linalg.norm(r['force'])>=cap-1e-3 for r in segment]))
   rows.append(row);runner.write(out/'summaries.json',rows);print('CAP_PROBE',task.label,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
   assert row['peak_force_N']<=cap+1e-5
   if row['boundary'] is not None:raise RuntimeError('Physical review boundary; stop rather than continue caps')
 runner.write(out/'identities.json',identities)
 return dict(status='completed',decisions=0,student_optimizer_updates=0,episodes=len(rows),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
