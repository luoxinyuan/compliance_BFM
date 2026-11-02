"""Paired full physical episodes: legacy vs resolved static config, identical RNG."""
import m11_ipc
import json,time,os,hashlib
from pathlib import Path
import m10_stage,m10_physics

def study(e,m,c,out):
 import torch,numpy as np
 from m11_env import Task
 from m11_policy import Expert
 from m11_runtime_speed import install
 source=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-privileged-expert-20261101/train-attempt-01/outputs/job/best.pt')
 ck=torch.load(source,map_location='cuda',weights_only=True);net=Expert(ck['input_dim']).cuda();net.load_state_dict(ck['state_dict'])
 plan=json.loads(Path(__file__).with_name('m11_continuation_protocol.json').read_text())
 refs={};results=[]
 for mode in ['legacy','resolved']:
  if mode=='resolved':(out/'adapter.json').write_text(json.dumps(install(e),indent=2))
  for case in ['straight','left','local','no_contact']:
   torch.manual_seed(901);np.random.seed(901);task=Task(e,m,c,1001);obs=task.reset(plan['dev'][case],record=True,label=mode+'-'+case);gen=torch.Generator(device='cuda').manual_seed(202)
   torch.cuda.synchronize();start=time.monotonic();decisions=0
   while not task.done:
    d=net.distribution(obs);raw=d.mean+d.stddev*torch.randn(d.mean.shape,device='cuda',generator=gen);obs,_,_,_=task.step(raw);decisions+=1
   torch.cuda.synchronize();elapsed=time.monotonic()-start;summary=task.save(out)
   physical={k:np.asarray([r[k] for r in task.physics]) for k in ['qpos','qvel','force','anchor','support']};controls={k:np.asarray([r[k] for r in task.controls]) for k in ['action','q_PD','executed_z']}
   data=dict(physical=physical,controls=controls,rewards=np.asarray([r['reward'] for r in task.stats]))
   if mode=='legacy':refs[case]=data;parity=None
   else:
    prior=refs[case];parity={group:{k:float(np.max(np.abs(v-prior[group][k]))) for k,v in data[group].items()} for group in ['physical','controls']};parity['reward_max_abs']=float(np.max(abs(data['rewards']-prior['rewards'])))
    assert all(v==0 for group in ['physical','controls'] for v in parity[group].values()) and parity['reward_max_abs']==0,parity
   row=dict(mode=mode,case=case,elapsed_s=elapsed,decisions=decisions,rate=decisions/elapsed,summary=summary,parity=parity);results.append(row);(out/'parity.json').write_text(json.dumps(results,indent=2));print('PERF_PARITY',mode,case,elapsed,parity,flush=True)
 return dict(status='completed',results=results,policy_updated=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
m10_stage.run_study=study
if __name__=='__main__':m10_physics.main()
