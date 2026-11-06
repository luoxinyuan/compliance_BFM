"""Frozen final evaluation, paired common prefixes and full real qpos traces."""
import copy,json,hashlib,subprocess
from pathlib import Path
import transfer_student_validated as runner

def study(env,model,contract,out):
 import torch,numpy as np
 from native_env import Task,install,spec,RUN
 from native_policy import Expert
 from clean_policy import Expert as OldExpert
 from geometry_policy import expand
 from m11_runtime_speed import install as speed
 install();runner.write(out/'runtime-speed.json',speed(env));torch.set_num_threads(1);torch.set_grad_enabled(False);task=Task(env,model,contract,4810);nets={};identities={}
 for arm in ['ordinary','guided']:
  for tag in ['best','last']:
   p=RUN/f'ppo-{arm}'/(tag+'.pt');c=torch.load(p,map_location='cuda',weights_only=True);n=Expert(c['input_dim']).cuda();n.load_state_dict(c['state_dict']);nets[arm+'_'+tag]=n.eval().requires_grad_(False);identities[arm+'_'+tag]={'path':str(p),'sha256':runner.sha(p),'iteration':c['iteration'],'runtime':c['runtime']}
 p=RUN/'native-initial.pt';c=torch.load(p,map_location='cuda',weights_only=True);n=Expert(c['input_dim']).cuda();n.load_state_dict(c['state_dict']);nets['native_initial']=n.eval().requires_grad_(False);identities['native_initial']={'path':str(p),'sha256':runner.sha(p),'runtime':c['runtime']}
 P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
 for name,path,sha in [('I',P/'m39a-demand-20261104/B-ppo-03/job/best.pt','72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a'),('C0',P/'prior-recovery-20261105/correct-C0/C0.pt','ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c')]:
  assert runner.sha(path)==sha;c=torch.load(path,map_location='cuda',weights_only=True);n=OldExpert(1088,'direct').cuda();expand(n,c);nets[name]=n.eval().requires_grad_(False);identities[name]={'path':str(path),'sha256':sha,'scope':'frozen evaluation reference only, never native training source'}
 runner.write(out/'models.json',identities)
 cases={'new_near':spec(.068,1.9,5.17),'new_mid':spec(.215,3.2,5.17),'new_long':spec(.69,8.3,5.17),'new_stop':spec(.69,8.3,5.17,stop=8.27),'new_continue':spec(.69,8.3,5.17),'new_arm':spec(.215,3.2,6.4,'goal_arm5',5.7),'new_moving':spec(.44,5.8,6.7,'motion_walk_task',6.1),'no_contact':spec(0.,4.,5.17,enabled=False),'zero_connected':spec(0.,4.,5.17),'light5':spec(.068,1.9,5.17,cap=5.)}
 runner.write(out/'conditions.json',cases);summ=[];prefixes={};comparisons=[]
 for name,net in nets.items():
  for case,s in cases.items():
   # Exact duplicate continue/long condition is intentionally a prefix check,
   # not an additional independent heldout physical condition.
   task.reset(s,record=True,label=name+'__'+case);commands=[]
   while not task.done:
    x=task.features();raw=net.raw_mean(x) if hasattr(net,'raw_mean') else net.distribution(x).mean;commands.append({'time':float(env.tether.time),'x':x.cpu().tolist(),'raw':raw.cpu().tolist()});task.step(raw)
   row=task.save(out);row.update(model=name,case=case);summ.append(row);runner.write(out/'summaries.json',summ);print('NATIVE_FINAL',name,case,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
   if case in ['new_stop','new_continue']:
    pre=[r for r in task.physics if r['force_time']<8.27-1e-8];cmd=[r for r in commands if r['time']<8.27-1e-8];prefixes[(name,case)]={'physics':pre,'command':cmd}
  a=prefixes[(name,'new_stop')];b=prefixes[(name,'new_continue')]
  check={}
  for key in ['qpos','qvel','force','wrist','anchor']:
   aa=np.array([r[key] for r in a['physics']]);bb=np.array([r[key] for r in b['physics']]);check[key+'_max']=float(np.abs(aa-bb).max()) if aa.shape==bb.shape else None
  for key in ['x','raw']:
   aa=np.array([r[key] for r in a['command']]);bb=np.array([r[key] for r in b['command']]);check[key+'_max']=float(np.abs(aa-bb).max()) if aa.shape==bb.shape else None
  comparisons.append({'model':name,'prefix_end':8.27,**check});runner.write(out/'common-prefix.json',comparisons)
 # Simple physical references use source calls only, clearly not learned control.
 for mode in ['standing','goal_arm5','motion_walk_task']:
  for case in ['new_near','new_long']:
   task.reset(cases[case],record=True,label='native_'+mode+'__'+case)
   while not task.done:task.step(task.native(mode,float(env.tether.time))/16)
   row=task.save(out);row.update(model='native_'+mode,case=case);summ.append(row);runner.write(out/'summaries.json',summ)
 return dict(status='completed',models=len(nets),episodes=len(summ),rl_decisions=0,supervised_updates=0,source='frozen final; no checkpoint reselection',old_actors='evaluation references only')
runner.study=study
if __name__=='__main__':runner.main()
