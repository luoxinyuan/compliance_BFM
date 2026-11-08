"""NumPy-only saved signal review; no model inference, GPU, physics or PPO updates."""
import argparse,json,time
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);s=Path(__file__).resolve().parents[2];root=s/'runs/single5201-independent-32';logs=[json.loads(x) for x in (root/'training.jsonl').read_text().splitlines()];allrows=[];checks=[];start=time.monotonic()
 for version in [1,7,15,23]:
  ds=[];ss=[]
  for w in range(3):
   with np.load(root/f'worker-{w}/rollout-{version:04d}.npz') as f:ds.append({k:f[k].copy() for k in ['reward','value','next_value','discount','done']})
   ss.append(json.loads((root/f'worker-{w}/rollout-{version:04d}-stats.json').read_text()))
  b={k:np.stack([d[k] for d in ds],axis=1) for k in ds[0]};adv=np.zeros_like(b['reward']);carry=np.zeros_like(b['reward'][0])
  for t in reversed(range(len(adv))):
   cont=(~b['done'][t]).astype(np.float32);delta=b['reward'][t]+b['discount'][t]*cont*b['next_value'][t]-b['value'][t];carry=delta+b['discount'][t]*np.float32(.95)*cont*carry;adv[t]=carry
  flat=adv.reshape(-1);normalized=(flat-flat.mean())/max(flat.std(),1e-6);rows=[]
  for t in range(256):
   for w in range(3):
    st=ss[w][t];i=t*3+w;assert abs(float(b['reward'][t,w])-st['reward'])<1e-6
    row=dict(phase=st['phase'],family=st['demand_family'],adv=float(normalized[i]),raw_adv=float(flat[i]),reward=float(b['reward'][t,w]),full_saturation=st['active_steps']>0 and st['cap_steps']==st['active_steps'],active=st['active_steps']>0,terms=st['terms']);rows.append(row);allrows.append(row)
  difference=max(abs(np.mean([r['adv'] for r in rows if r['phase']==phase])-logs[version]['groups']['phase:'+phase]['advantage_mean']) for phase in {r['phase'] for r in rows});assert difference<1e-5
  checks.append(dict(version=version,recomputed_phase_advantage_max_error=float(difference)))
 groups={}
 for phase in sorted({r['phase'] for r in allrows}):
  rr=[r for r in allrows if r['phase']==phase];active=[r for r in rr if r['active']];sat=[r for r in active if r['full_saturation']];terms={k:float(np.mean([r['terms'][k] for r in rr])) for k in rr[0]['terms']};cost=sum(terms.values());groups[phase]=dict(decisions=len(rr),fully_saturated_fraction_of_active=len(sat)/max(len(active),1),mean_reward=float(np.mean([r['reward'] for r in rr])),reward_std=float(np.std([r['reward'] for r in rr])),reward_terms=terms,force_cost_fraction=terms['force']/max(cost,1e-12),positive_adv_fraction=float(np.mean([r['adv']>0 for r in rr])),normalized_adv_std=float(np.std([r['adv'] for r in rr])),fully_saturated_reward_std=float(np.std([r['reward'] for r in sat])) if sat else None,fully_saturated_force_cost_range=[float(min(r['terms']['force'] for r in sat)),float(max(r['terms']['force'] for r in sat))] if sat else None)
 result=dict(status='completed',scope='four fixed saved batches x3workers; excludes batch0 value warmup; descriptive associations only',checks=checks,groups=groups,decisions=len(allrows),limitations=['Different states and future trajectories are mixed; phase mean advantages are not causal action rankings','No inference that low cost variance alone makes RL impossible; tracking/uncapped/velocity/heading remain available','No new reward/std/KL/curriculum change; only diagnose reward distinction and saved GAE consistency'],GPU_used=False,new_RL_decisions=0,new_physics_steps=0,elapsed_s=time.monotonic()-start)
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
