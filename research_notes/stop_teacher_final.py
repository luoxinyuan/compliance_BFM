"""Frozen preselected teacher final evaluation; no updates or model reselection."""
from pathlib import Path
import json,time
import transfer_student_validated as runner
from stop_env import PLAN,install
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/stop-sensing-20261103'
def study(env,model,contract,out):
 import torch
 from clean_policy import Expert
 from m11_runtime_speed import install as speed
 install();from clean_env import Task
 runner.write(out/'runtime-speed.json',speed(env));torch.set_grad_enabled(False);torch.set_num_threads(1);task=Task(env,model,contract,707);task.reset();device=env.standing.device
 attempts=json.loads((RUN/'compute-ledger.json').read_text())['attempts'];train=next(Path(r['path'])/'job' for r in attempts if r['arm']=='B1' and r['stage']=='train' and r['status']=='finished');selection=json.loads((train/'selection.json').read_text());selected=dict(path=str(train/'best.pt'),sha256=selection['sha256']);cases={'initial':PLAN['teachers']['initial'],'selected':selected};rows=[]
 for label,identity in cases.items():
  path=Path(identity['path']);assert runner.sha(path)==identity['sha256'];ck=torch.load(path,map_location=device,weights_only=True);net=Expert(ck['input_dim'],'direct').to(device);net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False)
  for name,spec in {**PLAN['final'],**PLAN['regression']}.items():
   task.reset(spec,record=True,label=label+'_'+name)
   while not task.done:task.step(net.distribution(task.features()).mean)
   row=task.save(out);rows.append(row);runner.write(out/'summaries.json',rows);print('TEACHER_FINAL',row['label'],row['error_cm'],row['force_rms'],flush=True)
 runner.write(out/'identities.json',dict(selection=selection,cases=cases));result=dict(status='completed',decisions=0,student_optimizer_updates=0,elapsed_s=time.monotonic()-runner.START);runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
