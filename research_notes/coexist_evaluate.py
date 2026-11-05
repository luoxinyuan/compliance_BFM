"""Frozen final evaluation, never used for checkpoint selection."""
import json,time,hashlib
from pathlib import Path
import transfer_student_validated as runner

def study(env,model,contract,out):
 import torch
 from coexist_env import Task,install_frame
 from m11_runtime_speed import install
 from adapter_policy import load
 install_frame();install(env);torch.set_num_threads(1);torch.set_grad_enabled(False)
 plan=json.loads(Path(__file__).with_name('coexist_protocol.json').read_text());manifest=json.loads(Path(runner.ARGS.resume).read_text());task=Task(env,model,contract,4600);rows=[];checks=[]
 for name,identity in manifest.items():
  assert runner.sha(identity['path'])==identity['sha256'];net=load(torch.load(identity['path'],map_location='cuda',weights_only=True),'cuda');prefix={}
  for k,s in plan['final'].items():
   if time.monotonic()-runner.START>runner.ARGS.max_seconds-45:raise TimeoutError('final budget')
   task.reset(s,record=True,label=name+'__'+k)
   while not task.done:task.step(net.distribution(task.features()).mean)
   row=task.save(out);rows.append(row);runner.write(out/'summaries.json',rows)
   if k in ['holdout_stop','holdout_continue']:
    end=plan['final']['holdout_stop']['stop_time'];payload=dict(physics=[r for r in task.physics if r['force_time']<end-1e-8],decisions=[r for r in task.decision_rows if r['time']<=end+1e-8]);prefix[k]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
   print('COEXIST_FINAL',name,k,row['error_cm'],row['force_rms'],row['boundary'],flush=True)
  assert prefix['holdout_stop']==prefix['holdout_continue'];checks.append(dict(model=name,prefix_hash=prefix['holdout_stop'],same=True));runner.write(out/'prefix-parity.json',checks)
 runner.write(out/'models.json',manifest)
 return dict(status='completed',episodes=len(rows),updates=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
