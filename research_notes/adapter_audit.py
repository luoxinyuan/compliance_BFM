"""Bounded shared-prefix label/observability audit; no optimization updates."""
import json,gzip,time
from pathlib import Path
import transfer_student_validated as runner
from adapter_fit import OLD,REC

def study(env,model,contract,out):
 import torch,numpy as np
 from clean_policy import Expert,direct_command
 from geometry_policy import expand
 from adapter_policy import ResidualExpert
 from m39b_replay import interp
 torch.set_num_threads(1);torch.set_grad_enabled(False);model.eval().requires_grad_(False)
 cfg=json.loads(Path(__file__).with_name('recovery_protocol.json').read_text());ck=torch.load(cfg['models']['I']['path'],map_location=env.standing.device,weights_only=True);I=Expert(1088,'direct').to(env.standing.device);expand(I,ck);I.requires_grad_(False)
 rows=[]
 for m in json.loads((OLD/'fit2-B/initialization.json').read_text())['manifest']:
  assert runner.sha(m['source'])==m['sha256']
  with gzip.open(m['source'],'rt') as f:rr=json.load(f)
  rows += [rr[i] for i in np.linspace(0,len(rr)-1,m['samples']).round().astype(int)]
 x=torch.tensor([r['x'] for r in rows],device=env.standing.device);z=torch.tensor([r['z'] for r in rows],device=x.device);xn=I.normalized(x);dist=torch.cdist(xn,xn)/1088**.5
 kinds=[r['kind'] for r in rows];mask=torch.tensor([[k==l for l in kinds] for k in kinds],device=x.device);dist.masked_fill_(mask,float('inf'));flat=dist.flatten().argsort()[:30];pairs=[]
 scale=torch.tensor(contract['q_PD_scale'],device=x.device);off=torch.tensor(contract['q_PD_offset'],device=x.device)
 for ix in flat.tolist():
  i,j=divmod(ix,len(rows))
  if j<=i:continue
  r=rows[i];obs={k:torch.tensor(r['sequence'][0]['bfm_obs'][k],device=x.device) for k in contract['bfm_inputs']};prev=torch.tensor(r['previous_z'],device=x.device)[None];zs=torch.cat([z[i:i+1],z[j:j+1]]);zz=interp(prev.expand(2,-1),zs)
  ob={k:v.expand(10,*v.shape[1:]) for k,v in obs.items()};a=model.act(ob,zz.reshape(10,256),mean=True);q=off+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5;q=q.reshape(2,5,29)
  pairs.append(dict(i=i,j=j,kind=[kinds[i],kinds[j]],time=[rows[i]['time'],rows[j]['time']],input_rms=float(dist[i,j]),z_angle_deg=float(torch.rad2deg(torch.acos((z[i]*z[j]).sum().div(256).clamp(-1,1)))),same_state_five_qpd_rms_deg=float(torch.rad2deg(q[0]-q[1]).square().mean().sqrt())))
 contradictions=[r for r in pairs if r['input_rms']<1e-6 and r['same_state_five_qpd_rms_deg']>.05]
 torch.manual_seed(4501);R=ResidualExpert(ck).to(x.device);R.requires_grad_(False);R.actor.requires_grad_(True)
 equal=torch.equal(R.distribution(x).mean,I.distribution(x).mean);assert equal
 with torch.enable_grad():
  loss=(direct_command(R.distribution(x).mean,env.standing)-z).square().mean();loss.backward();grads={n:float(p.grad.norm()) for n,p in R.actor.named_parameters()};assert grads['4.weight']>0 and grads['4.bias']>0
 assert not any(p.grad is not None for p in R.base.parameters());assert not any(p.grad is not None for p in model.parameters())
 runner.write(out/'audit.json',dict(samples=len(rows),pairs=pairs,exact_conflicts=contradictions,initial_I_parity=equal,adapter_initial_gradients=grads,first_layer_initial_zero_gradient_expected='zero last layer, hidden layers receive gradient after first update; not all-layer zero initialization',label_source_scope='arm5/15 share continuous current-displacement mapping within qualified source; walking uses I, not a general unified expert; near labels alone do not prove hidden future contradiction',decision='same corrected dataset both arms' if not contradictions else 'STOP explicit conflict requires common repair'))
 assert not contradictions,'Exact same-input action conflicts require shared data repair'
 return dict(status='completed',updates=0,elapsed_s=time.monotonic()-runner.START)
runner.study=study
if __name__=='__main__':runner.main()
