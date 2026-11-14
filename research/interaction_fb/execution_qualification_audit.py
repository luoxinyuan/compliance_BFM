"""No physics: independently check saved qualification yaw and original-Actor actions."""
import gzip,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import torch,numpy as np
from humanoidverse.agents import load_utils
P=ROOT.parents[2];S=P/'experiments/r2-online-20261114';D=S/'runs/initial-qualification-01'
t0=time.monotonic();torch.set_num_threads(1)
m=load_utils.load_model_from_checkpoint_dir(str(P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint'),device='cpu');m.eval().requires_grad_(False)
rows=[]
for path in sorted(D.glob('case-*/actual-0.json.gz')):
 with gzip.open(path,'rt') as f:t=json.load(f)
 ph=t['physics'];angles=np.unwrap([np.arctan2(np.asarray(p['pelvis_matrix']).reshape(3,3)[1,0],np.asarray(p['pelvis_matrix']).reshape(3,3)[0,0]) for p in ph])
 errors=[]
 with torch.no_grad():
  for c in t['controls'][::10]:
   o={k:torch.tensor(v,dtype=torch.float32) for k,v in c['bfm_obs'].items()};z=torch.tensor(c['executed_z'],dtype=torch.float32).reshape(1,-1)
   a=m._actor(m._normalize(o),z,m.cfg.actor_std).mean
   actual=torch.tensor(c['action'],dtype=torch.float32).reshape(1,-1)
   errors.append(float((a-actual).abs().max()))
 rows.append(dict(case=path.parent.name,family=t['spec']['demand_family'],controls_checked=len(errors),original_actor_cpu_vs_executed_gpu_action_max_abs=max(errors),yaw_unwrapped_delta_deg=float(np.degrees(angles[-1]-angles[0])),yaw_abs_final_deg=float(abs(np.degrees(angles[-1])%360) if False else abs(np.degrees(np.arctan2(np.sin(angles[-1]),np.cos(angles[-1]))))),physics_ticks=len(ph)))
result=dict(rows=rows,wall_s=time.monotonic()-t0,new_physics=0,new_updates=0,cuda_initialized=torch.cuda.is_initialized(),scope='Original frozen Actor at recorded actual z; CPU/GPU numerical check, not a new tolerance or full U trajectory rerun')
(S/'analysis/qualification-readonly-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
