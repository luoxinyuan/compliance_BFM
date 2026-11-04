"""Final frozen-model physical evaluation; no training or dataset generation."""
from pathlib import Path
import json,time
import transfer_student_validated as runner
from m39b_env import PLAN,Task,install_frame
from m39b_metrics import extra

def study(env,model,contract,out):
 import torch
 from clean_policy import Expert
 from geometry_policy import expand
 from m11_runtime_speed import install
 assert runner.ARGS.stage=='final'
 install_frame();runner.write(out/'runtime-speed.json',install(env));torch.set_num_threads(1);torch.set_grad_enabled(False)
 task=Task(env,model,contract,runner.ARGS.seed);summary=[]
 config=json.loads(Path(runner.ARGS.resume).read_text())
 for name,ident in config['models'].items():
  assert runner.sha(ident['path'])==ident['sha256']
  ck=torch.load(ident['path'],map_location=env.standing.device,weights_only=True)
  net=Expert(1088,'direct').to(env.standing.device);expand(net,ck);net.eval().requires_grad_(False)
  for key in config['keys']:
   task.reset(PLAN['sealed'][key],record=True,label=name+'_'+key)
   while not task.done:task.step(net.distribution(task.features()).mean)
   row=task.save(out);row.update(extra(task));summary.append(row);runner.write(out/'summaries.json',summary)
   print('M39B_FINAL',row['label'],row['error_cm'],row['sum_force_norm_rms'],row['actual_torso_turn_deg'],row['boundary'],flush=True)
 return dict(status='completed',decisions=0,expert_supervised_updates=0,student_optimizer_updates=0,episodes=len(summary),elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
