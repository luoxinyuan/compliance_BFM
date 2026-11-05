"""Supplemental source-meaning check, not a new blind test or training condition.
Predeclared6.5s was actually backward step preparation; preserve it and add real8s walking.
"""
import json,time
from pathlib import Path
import transfer_student_validated as runner
from adapter_evaluate import setup,episode,conditions

def study(env,model,contract,out):
 import torch
 from adapter_policy import load
 task,I,goal=setup(env,model,contract);manifest=json.loads(Path(runner.ARGS.resume).read_text());spec=dict(conditions()['new65'],learned_handoff=8.0);rows=[]
 for name in ['I','F4501','R4501']:
  d=manifest[name];assert runner.sha(d['path'])==d['sha256'];n=load(torch.load(d['path'],map_location=env.standing.device,weights_only=True),env.standing.device);r,_=episode(task,I,goal,n,spec,name+'__walking8',out);rows.append(r);runner.write(out/'summaries.json',rows)
 runner.write(out/'scope.json',dict(reason='I saved new65 source6.5s root vx=-.0402m/s,8s vx=.1606m/s; preserve6.5 and explicitly add8s development probe',models='fixed seed4501, no outcome selection',no_training=True))
 return dict(status='completed',episodes=3,updates=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
