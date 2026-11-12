import json,sys,math
from pathlib import Path
import torch,numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--original-run',type=Path,required=True)
args=parser.parse_args();S=args.stage;OLD=args.original_run
def diff(a,b,path='root'):
 if torch.is_tensor(a) or torch.is_tensor(b):
  return None if torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes() else path
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
  return None if a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() else path
 if isinstance(a,dict):
  if a.keys()!=b.keys():return path+'.keys'
  for k in a:
   d=diff(a[k],b[k],path+'.'+str(k))
   if d:return d
  return None
 if isinstance(a,(list,tuple)):
  if len(a)!=len(b):return path+'.length'
  for i,(x,y) in enumerate(zip(a,b)):
   d=diff(x,y,path+f'[{i}]')
   if d:return d
  return None
 if isinstance(a,float):return None if isinstance(b,float) and a.hex()==b.hex() else path
 return None if a==b else path
old=torch.load(OLD/'uncommitted-stop.pt',map_location='cpu',weights_only=False)
traces={};results={}
for name in ('sample-r3a','sample-r3b','mean-r3','ankle-r4'):
 p=S/'runs'/name
 if not (p/'probe.json').exists():continue
 j=json.loads((p/'probe.json').read_text());t=torch.load(p/'probe-trace.pt',map_location='cpu',weights_only=False);traces[name]=t
 rows=t['physics'];ctrl=t['controls'];after=[c for c in ctrl if c['time']>=rows[0]['time']-.02]
 results[name]={k:v for k,v in j.items() if k not in ('physics','error')}
 results[name]['physical_summary']=dict(t_start=rows[0]['time'],t_end=rows[-1]['time'],ticks=len(rows),
  q_right_ankle_min=min(r['qpos'][18] for r in rows),q_right_ankle_max=max(r['qpos'][18] for r in rows),
  excess_max=max(r['joint_limit_excess_max_rad'] for r in rows),
  qpd_right_ankle_min=min(c['q_PD'][11] for c in after),qpd_right_ankle_max=max(c['q_PD'][11] for c in after),
  error_last=rows[-1]['error'],support_last=rows[-1]['support'])
 if name.startswith('sample'):
  matched=[r for r in old['physics'] if r['time']>=rows[0]['time']]
  results[name]['original_physics_first_difference']=diff(rows,matched)
  results[name]['original_controls_first_difference']=diff(ctrl,old['controls'])
  results[name]['original_actions_first_difference']=diff(t['actions'],old['actions'])
  results[name]['original_commands_first_difference']=diff(t['commands'],old['commands'])
if 'sample-r3a' in traces and 'sample-r3b' in traces:
 results['new_process_repeat_first_difference']=diff(traces['sample-r3a'],traces['sample-r3b'])
(S/'diagnosis.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a!='boundary'} if isinstance(v,dict) else v for k,v in results.items()},indent=2))
