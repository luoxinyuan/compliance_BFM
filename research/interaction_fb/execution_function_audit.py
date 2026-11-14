"""Read-only CPU functional footprint of engineering updates; not calibration success."""
import copy,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import torch,numpy as np
from humanoidverse.agents import load_utils
from research.interaction_fb.journal import load_state
from research.interaction_fb.schema import robot
P=ROOT.parents[2];S=P/'experiments/r2-online-20261114'
t0=time.monotonic();torch.set_num_threads(1)
base=load_utils.load_model_from_checkpoint_dir(str(P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint'),device='cpu');base.eval().requires_grad_(False)
results=[]
for arm in ('e','i'):
 state=load_state(S/f'runs/{arm}-engineering-01/checkpoints/round-1');tail=copy.deepcopy(base._actor.policy[5:]);tail.load_state_dict(state['agent']['tail']);scale=state['agent']['scale']
 for label,rows in [('native',state['native']),('visited',state['rows'][::5])]:
  vals=[]
  with torch.no_grad():
   for row in rows:
    o=robot(row['observation']);z=row['z'];actor=base._actor;x=actor.input_filter(base._normalize(o));h=torch.cat([actor.embed_s(x),actor.embed_z(torch.cat([x,z],-1))],-1)
    for layer in actor.policy[:5]:h=layer(h)
    prior=torch.tanh(actor.policy[5:](h));new=torch.tanh(tail(h));dq=(new-prior)*scale
    vals.append(dq.flatten().numpy())
  arr=np.stack(vals)
  results.append(dict(arm=arm,domain=label,rows=len(rows),independent_parent_count=1,overall_qpd_rms_rad=float(np.sqrt(np.mean(arr**2))),abs_qpd_p95_rad=float(np.quantile(np.abs(arr),.95)),abs_qpd_max_rad=float(np.abs(arr).max()),per_joint_rms_rad=np.sqrt(np.mean(arr**2,axis=0)).tolist()))
out=S/'analysis';out.mkdir(exist_ok=True)
(out/'engineering-function-footprint.json').write_text(json.dumps(dict(results=results,wall_s=time.monotonic()-t0,cuda_initialized=torch.cuda.is_initialized(),new_physics=0,new_optimizer_steps=0,scope='One engineering parent each; not matched E/I physical comparison or global retention qualification'),indent=2))
print(json.dumps(results,indent=2))
