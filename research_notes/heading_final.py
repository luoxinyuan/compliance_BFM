"""Frozen final physical evaluation; no training or selection on sealed outcomes."""
from pathlib import Path
import json,gzip,os,time
import transfer_student_validated as runner
from heading_env import install
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/heading-transfer-20261103';PLAN=json.loads(Path(__file__).with_name('heading_protocol.json').read_text())
def study(env,model,contract,out):
 import numpy as np,torch
 from clean_policy import Expert
 from m11_runtime_speed import install as speed
 install();from clean_env import Task
 runner.write(out/'runtime-speed.json',speed(env));torch.set_num_threads(1);torch.set_grad_enabled(False);task=Task(env,model,contract,906);summary=[]
 manifest=json.loads((RUN/'heading-final-selection.json').read_text());runner.write(out/'frozen-selection.json',manifest)
 nets={}
 for key,id in manifest.items():
  assert runner.sha(id['path'])==id['sha256'];ck=torch.load(id['path'],map_location='cuda',weights_only=True);net=Expert(ck['input_dim'],'direct').cuda();net.load_state_dict(ck['state_dict']);nets[key]=net.eval().requires_grad_(False)
 for key,net in nets.items():
  for name,spec in {**PLAN['heading_final'],**PLAN['regression']}.items():
   label=key+'_'+name;task.reset(spec,record=True,label=label)
   while not task.done:task.step(net.distribution(task.features()).mean)
   row=task.save(out);summary.append(row);runner.write(out/'summaries.json',summary);print('HEADING_FINAL',label,row['error_cm'],row['force_rms'],row['moving_heading_rms_deg'],row['boundary'],flush=True)
  # Same-prefix physical causal check, not merely same partner prescription.
  def read(name):
   with gzip.open(out/(key+'_'+name+'.json.gz'),'rt') as f:return json.load(f)
  left,right=read('prefix_left'),read('prefix_right');change=left['condition']['change_time']
  a=[r for r in left['physics'] if r['time']<change];b=[r for r in right['physics'] if r['time']<change];assert len(a)==len(b)
  delta={k:float(np.abs(np.array([r[k] for r in a])-np.array([r[k] for r in b])).max()) for k in ['qpos','qvel','force','anchor']};assert max(delta.values())==0
  runner.write(out/(key+'-prefix-parity.json'),dict(change_time=change,maximum_difference=delta,source='identical complete physics prefix before real partner path diverges; learned model has no time/direction label'))
 result=dict(status='completed',decisions=0,student_optimizer_updates=0,elapsed_s=time.monotonic()-runner.START);runner.write(out/'result.json',result);return result
runner.study=study
if __name__=='__main__':runner.main()
