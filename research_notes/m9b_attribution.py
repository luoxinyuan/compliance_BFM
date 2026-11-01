"""At fixed z, decompose state-input change into physical state, last action and history."""
import json,gzip,itertools
import numpy as np,torch
from m9b_stage import OLD

def run(env,model,contract,out):
 sc=torch.tensor(contract['q_PD_scale'],device=env.standing.device);off=torch.tensor(contract['q_PD_offset'],device=env.standing.device);result={}
 for name in ['new_straight__expert','new_right__expert','new_heading__expert']:
  with gzip.open(OLD/'final-attempt-01/outputs'/(name+'.json.gz'),'rt') as f:e=json.load(f)
  cs=e['control'];ix=[int(i) for i in np.argsort([c['q_PD_step_max_rad'] for c in cs])[::-1] if i>0][:4];rr=[]
  for i in ix:
   c,p=cs[i],cs[i-1];z=torch.tensor(p['executed_z'],device=env.standing.device);obs=[{k:torch.tensor(r['bfm_obs'][k],device=env.standing.device) for k in env.keys} for r in [p,c]];values={}
   for bits in itertools.product([0,1],repeat=3):
    ob={k:obs[b][k] for k,b in zip(env.keys,bits)};a=model.act(ob,z,mean=True);values[bits]=(off+(a*5).clamp(-contract['clip'],contract['clip'])*sc/5)[0].cpu().numpy()
   contribution=np.zeros((3,29))
   for order in itertools.permutations(range(3)):
    bits=[0,0,0]
    for j in order:
     before=values[tuple(bits)];bits[j]=1;contribution[j]+=(values[tuple(bits)]-before)/6
   d=values[(1,1,1)]-values[(0,0,0)];j=int(np.argmax(abs(d)));rr.append(dict(time=c['time'],joint=contract['dof_names'][j],fixed_z_state_jump_deg=float(np.degrees(d[j])),shapley_group_deg={k:float(np.degrees(contribution[n,j])) for n,k in enumerate(env.keys)},reconstruction_max_rad=float(abs(contribution.sum(0)-d).max())))
  result[name]=rr
 (out/'state-input-attribution.json').write_text(json.dumps(result,indent=2));return result
